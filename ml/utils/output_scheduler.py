"""
Cross-Modal Output Scheduler
Manages frequency, intensity, and channel prioritization for accessibility outputs.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch


class OutputChannel(Enum):
    """Output channel types"""
    AUDIO = "audio"
    HAPTIC = "haptic"
    VISUAL = "visual"
    HYBRID = "hybrid"


class AlertFrequency(Enum):
    """Alert frequency levels"""
    LOW = "low"      # Only hazards
    MEDIUM = "medium"  # Hazards + important objects
    HIGH = "high"     # Continuous narration


@dataclass
class OutputConfig:
    """Configuration for output scheduling"""
    preferred_channel: OutputChannel = OutputChannel.AUDIO
    alert_frequency: AlertFrequency = AlertFrequency.MEDIUM
    audio_volume: float = 0.7
    haptic_intensity: float = 0.8
    visual_contrast: float = 0.9
    reaction_time_ms: float = 250.0
    uncertainty_threshold: float = 0.3  # Suppress alerts if uncertainty > threshold


@dataclass
class ScheduledOutput:
    """A scheduled output event"""
    channel: OutputChannel
    priority: int  # 0-100
    intensity: float  # 0-1
    frequency: float  # Hz
    duration: float  # seconds
    content: str  # Description for audio/narration
    spatial_position: Optional[Tuple[float, float]] = None  # For spatial audio/haptic


class CrossModalScheduler:
    """
    Schedules outputs across audio, haptic, and visual channels.
    Manages frequency, intensity, and prioritization based on user profile and model outputs.
    """
    
    def __init__(self, config: OutputConfig):
        self.config = config
        self.last_output_time: Dict[str, float] = {}
        self.output_history: List[ScheduledOutput] = []
        
    def schedule_outputs(
        self,
        detections: List[Dict],
        model_outputs: Dict[str, torch.Tensor],
        timestamp: float
    ) -> List[ScheduledOutput]:
        """
        Schedule outputs based on detections and model outputs.
        
        Args:
            detections: List of detection dictionaries with priority, findability, etc.
            model_outputs: Model outputs including uncertainty, navigation_difficulty, etc.
            timestamp: Current timestamp for rate limiting
        
        Returns:
            List of scheduled outputs
        """
        scheduled = []
        
        # Get uncertainty - suppress if too high
        uncertainty = model_outputs.get('uncertainty', torch.tensor(0.0))
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.item()
        
        if uncertainty > self.config.uncertainty_threshold:
            # High uncertainty - only output high-priority items
            priority_threshold = 90
        else:
            # Normal operation - use frequency-based threshold
            priority_threshold = self._get_priority_threshold()
        
        # Filter detections by priority and frequency settings
        filtered_detections = [
            d for d in detections
            if d.get('priority', 0) >= priority_threshold
        ]
        
        # Sort by priority (highest first)
        filtered_detections.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Limit number of outputs based on frequency
        max_outputs = self._get_max_outputs()
        filtered_detections = filtered_detections[:max_outputs]
        
        # Schedule each detection
        for det in filtered_detections:
            output = self._create_output_for_detection(det, model_outputs, timestamp)
            if output:
                scheduled.append(output)
                self.last_output_time[det.get('class_name', 'unknown')] = timestamp
        
        # Add scene-level outputs (navigation difficulty, glare warnings, etc.)
        scene_outputs = self._create_scene_outputs(model_outputs, timestamp)
        scheduled.extend(scene_outputs)
        
        # Store history
        self.output_history.extend(scheduled)
        if len(self.output_history) > 100:  # Keep last 100 outputs
            self.output_history = self.output_history[-100:]
        
        return scheduled
    
    def _get_priority_threshold(self) -> int:
        """Get priority threshold based on alert frequency"""
        thresholds = {
            AlertFrequency.LOW: 70,      # Only hazards + navigation
            AlertFrequency.MEDIUM: 40,    # + useful objects
            AlertFrequency.HIGH: 0       # All objects
        }
        return thresholds.get(self.config.alert_frequency, 40)
    
    def _get_max_outputs(self) -> int:
        """Get maximum number of outputs per frame based on frequency"""
        limits = {
            AlertFrequency.LOW: 3,
            AlertFrequency.MEDIUM: 5,
            AlertFrequency.HIGH: 10
        }
        return limits.get(self.config.alert_frequency, 5)
    
    def _create_output_for_detection(
        self,
        detection: Dict,
        model_outputs: Dict[str, torch.Tensor],
        timestamp: float
    ) -> Optional[ScheduledOutput]:
        """Create output for a single detection"""
        priority = detection.get('priority', 0)
        class_name = detection.get('class_name', 'object')
        box = detection.get('box', [0.5, 0.5, 0.1, 0.1])
        findability = detection.get('findability', 0.5)
        urgency = detection.get('urgency', 0)
        
        # Check rate limiting
        if self._should_suppress(class_name, timestamp, priority):
            return None
        
        # Determine channel
        channel = self._select_channel(priority, urgency)
        
        # Calculate intensity based on priority and findability
        intensity = self._calculate_intensity(priority, findability, urgency)
        
        # Calculate frequency (Hz) - higher priority = faster rhythm
        frequency = self._calculate_frequency(priority, urgency)
        
        # Calculate duration
        duration = self._calculate_duration(priority, urgency)
        
        # Generate content
        content = self._generate_content(detection, model_outputs)
        
        # Spatial position from bounding box center
        spatial_pos = (box[0], box[1]) if len(box) >= 2 else None
        
        return ScheduledOutput(
            channel=channel,
            priority=priority,
            intensity=intensity,
            frequency=frequency,
            duration=duration,
            content=content,
            spatial_position=spatial_pos
        )
    
    def _should_suppress(self, class_name: str, timestamp: float, priority: int) -> bool:
        """Check if output should be suppressed due to rate limiting"""
        if class_name not in self.last_output_time:
            return False
        
        last_time = self.last_output_time[class_name]
        time_since = timestamp - last_time
        
        # Rate limits based on priority (higher priority = more frequent)
        if priority >= 90:
            min_interval = 0.5  # 2 Hz max for hazards
        elif priority >= 70:
            min_interval = 1.0  # 1 Hz max for navigation
        else:
            min_interval = 2.0  # 0.5 Hz max for useful objects
        
        return time_since < min_interval
    
    def _select_channel(self, priority: int, urgency: int) -> OutputChannel:
        """Select output channel based on priority and user preference"""
        # High priority/urgency -> use preferred channel or hybrid
        if priority >= 90 or urgency >= 3:
            if self.config.preferred_channel == OutputChannel.HYBRID:
                return OutputChannel.HYBRID
            return self.config.preferred_channel
        
        # Medium priority -> use preferred channel
        if priority >= 70:
            return self.config.preferred_channel
        
        # Low priority -> use less intrusive channel
        if self.config.preferred_channel == OutputChannel.AUDIO:
            return OutputChannel.VISUAL  # Visual overlay instead of audio
        return self.config.preferred_channel
    
    def _calculate_intensity(self, priority: int, findability: float, urgency: int) -> float:
        """Calculate output intensity (0-1)"""
        # Base intensity from priority
        base_intensity = priority / 100.0
        
        # Adjust for findability (harder to find = higher intensity)
        findability_adjustment = (1.0 - findability) * 0.2
        
        # Adjust for urgency
        urgency_adjustment = urgency / 3.0 * 0.3
        
        intensity = base_intensity + findability_adjustment + urgency_adjustment
        
        # Apply channel-specific scaling
        if self.config.preferred_channel == OutputChannel.AUDIO:
            intensity *= self.config.audio_volume
        elif self.config.preferred_channel == OutputChannel.HAPTIC:
            intensity *= self.config.haptic_intensity
        else:
            intensity *= self.config.visual_contrast
        
        return min(1.0, max(0.0, intensity))
    
    def _calculate_frequency(self, priority: int, urgency: int) -> float:
        """Calculate output frequency in Hz"""
        # Higher priority/urgency = faster rhythm
        if priority >= 90 or urgency >= 3:
            return 10.0  # Fast rhythm for hazards
        elif priority >= 70:
            return 5.0   # Medium rhythm for navigation
        else:
            return 2.0   # Slow rhythm for useful objects
    
    def _calculate_duration(self, priority: int, urgency: int) -> float:
        """Calculate output duration in seconds"""
        # Higher priority = longer duration
        if priority >= 90 or urgency >= 3:
            return 0.5  # Longer for hazards
        elif priority >= 70:
            return 0.3  # Medium for navigation
        else:
            return 0.1  # Short for useful objects
    
    def _generate_content(self, detection: Dict, model_outputs: Dict) -> str:
        """Generate content description for output"""
        class_name = detection.get('class_name', 'object')
        distance = detection.get('distance', 'medium')
        priority = detection.get('priority', 0)
        
        # Simple content generation
        if priority >= 90:
            return f"Warning: {class_name} ahead"
        elif priority >= 70:
            return f"{class_name} {distance}"
        else:
            return class_name
    
    def _create_scene_outputs(
        self,
        model_outputs: Dict[str, torch.Tensor],
        timestamp: float
    ) -> List[ScheduledOutput]:
        """Create scene-level outputs (navigation difficulty, glare warnings, etc.)"""
        outputs = []
        
        # Navigation difficulty warning
        nav_difficulty = model_outputs.get('navigation_difficulty', None)
        if nav_difficulty is not None:
            if isinstance(nav_difficulty, torch.Tensor):
                nav_difficulty = nav_difficulty.item()
            if nav_difficulty > 0.7:  # High difficulty
                outputs.append(ScheduledOutput(
                    channel=self.config.preferred_channel,
                    priority=60,
                    intensity=0.6,
                    frequency=2.0,
                    duration=0.2,
                    content="Difficult navigation ahead"
                ))
        
        # Glare warning
        glare_level = model_outputs.get('glare_risk_level', None)
        if glare_level is not None:
            if isinstance(glare_level, torch.Tensor):
                glare_level = glare_level.item()
            if glare_level >= 2:  # Medium or high glare
                outputs.append(ScheduledOutput(
                    channel=OutputChannel.VISUAL,  # Visual overlay for glare
                    priority=50,
                    intensity=0.7,
                    frequency=1.0,
                    duration=0.3,
                    content="High glare detected"
                ))
        
        return outputs


def create_scheduler_from_profile(user_profile: Dict) -> CrossModalScheduler:
    """Create scheduler from user profile"""
    config = OutputConfig(
        preferred_channel=OutputChannel(user_profile.get('preferred_output_channel', 'audio')),
        alert_frequency=AlertFrequency(user_profile.get('alert_frequency', 'medium')),
        audio_volume=user_profile.get('accessibility_preferences', {}).get('audio_volume', 0.7),
        haptic_intensity=user_profile.get('accessibility_preferences', {}).get('haptic_intensity', 0.8),
        visual_contrast=user_profile.get('accessibility_preferences', {}).get('contrast_mode', 0.9),
        reaction_time_ms=user_profile.get('reaction_time_ms', 250.0)
    )
    return CrossModalScheduler(config)

