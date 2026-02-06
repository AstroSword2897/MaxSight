"""Degraded mode tracking for MaxSight Web Simulator.
Explicit failure modes instead of silent degradation."""
from enum import Enum
from typing import Dict, Set, Any
from dataclasses import dataclass, field
from threading import Lock


class DegradedMode(Enum):
    """Types of degraded operation."""
    NORMAL = "normal"
    VISION_UNSTABLE = "vision_unstable"  # Model inference issues
    AUDIO_UNAVAILABLE = "audio_unavailable"  # TTS/voice feedback down
    TEXT_DETECTION_OFFLINE = "text_detection_offline"  # OCR not working
    HAPTIC_UNAVAILABLE = "haptic_unavailable"  # Haptic feedback down
    MEMORY_FULL = "memory_full"  # Spatial memory at capacity
    PROCESSING_SLOW = "processing_slow"  # Performance degradation


@dataclass
class DegradedState:
    """Tracks degraded modes for a session."""
    active_modes: Set[DegradedMode] = field(default_factory=set)
    mode_reasons: Dict[DegradedMode, str] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)
    
    def set_degraded(self, mode: DegradedMode, reason: str = ""):
        """Mark a mode as degraded."""
        with self.lock:
            self.active_modes.add(mode)
            if reason:
                self.mode_reasons[mode] = reason
    
    def clear_degraded(self, mode: DegradedMode):
        """Clear a degraded mode."""
        with self.lock:
            self.active_modes.discard(mode)
            self.mode_reasons.pop(mode, None)
    
    def is_degraded(self, mode: DegradedMode) -> bool:
        """Check if a mode is degraded."""
        with self.lock:
            return mode in self.active_modes
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degraded status."""
        with self.lock:
            return {
                'is_degraded': len(self.active_modes) > 0,
                'active_modes': [mode.value for mode in self.active_modes],
                'reasons': dict(self.mode_reasons),
                'status_message': self._get_status_message()
            }
    
    def _get_status_message(self) -> str:
        """Get human-readable status message."""
        if not self.active_modes:
            return "All systems operational"
        
        messages = []
        if DegradedMode.VISION_UNSTABLE in self.active_modes:
            messages.append("Vision processing unstable")
        if DegradedMode.AUDIO_UNAVAILABLE in self.active_modes:
            messages.append("Audio feedback temporarily unavailable")
        if DegradedMode.TEXT_DETECTION_OFFLINE in self.active_modes:
            messages.append("Text detection offline")
        if DegradedMode.HAPTIC_UNAVAILABLE in self.active_modes:
            messages.append("Haptic feedback unavailable")
        if DegradedMode.MEMORY_FULL in self.active_modes:
            messages.append("Memory at capacity")
        if DegradedMode.PROCESSING_SLOW in self.active_modes:
            messages.append("Processing slower than normal")
        
        return "; ".join(messages) if messages else "Degraded operation"

