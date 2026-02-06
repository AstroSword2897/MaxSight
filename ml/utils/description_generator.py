"""Description generator - converts ML outputs to natural language descriptions."""

import torch
from typing import Dict, List, Optional, Tuple
import math


class DescriptionGenerator:
    """Generates natural language descriptions from model outputs."""
    
    DISTANCE_NAMES = ['near', 'medium', 'far']
    DIRECTION_ZONES = {
        'left': (-1.0, -0.33),
        'center_left': (-0.33, -0.1),
        'center': (-0.1, 0.1),
        'center_right': (0.1, 0.33),
        'right': (0.33, 1.0),
        'top': (-1.0, -0.33),
        'center_top': (-0.33, -0.1),
        'center_bottom': (0.1, 0.33),
        'bottom': (0.33, 1.0)
    }
    URGENCY_NAMES = ['safe', 'caution', 'warning', 'danger']
    
    def __init__(self, verbosity: str = 'normal'):
        self.verbosity = verbosity
    
    def get_direction_from_box(
        self, 
        box: torch.Tensor, 
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[str, str]:
        """Get horizontal and vertical direction from bounding box center."""
        where things are in 3D space relative to them. Without direction, "door detected" is useless
        for navigation - "door slightly left" is actionable.
        
        Arguments:
            box: [cx, cy, w, h] in normalized coordinates [0, 1]
            image_size: (width, height) of image
        
        Returns:
            (horizontal_direction, vertical_direction)
        """cx, cy = box[0].item(), box[1].item()..."""
        Get distance description from zone and optional size.
        
        Arguments:
            distance_zone: 0 (near), 1 (medium), 2 (far) or string ('near', 'medium', 'far')
            box_size: Optional box area for more precise estimation
        
        Returns:
            Distance description string
        """# Handle string distance zones (convert to int)..."""
        Estimate distance in meters (rough approximation).
        
        Arguments:
            distance_zone: 0 (near), 1 (medium), 2 (far)
            box_size: Box area (normalized)
            object_type: Type of object for size reference
        
        Returns:
            Estimated meters string or None if not available
        """if self.verbosity != 'detailed':..."""
        Get relative height description.
        
        Arguments:
            box: [cx, cy, w, h] in normalized coordinates
            image_size: (width, height)
        
        Returns:
            Height description
        """if self.verbosity != 'detailed':..."""
        Generate natural language description for a single object.
        
        CORE FUNCTION - WHY THIS EXISTS:
        This is the heart of MaxSight's "Environmental Structuring" approach. It transforms a technical
        detection (class="door", box=[0.3, 0.5, 0.2, 0.3], distance=0) into actionable information:
        "Door 2 meters ahead, slightly left, at eye level".
        
        This directly implements the MVP feature: "User points phone → app says: 'Door 2 meters ahead,
        handle left' or 'Stop sign.'" Without this transformation, users get raw technical data they
        cannot act upon.
        
        HOW IT SUPPORTS DIFFERENT VISION CONDITIONS:
        - Brief mode: For users who need minimal information (CVI, cognitive overload)
        - Normal mode: Standard actionable descriptions (most users)
        - Detailed mode: For users learning spatial relationships or needing full context
        
        This adaptive verbosity supports "Skill Development Across Senses" - users can start with
        detailed descriptions and gradually reduce to brief as they build spatial awareness.
        
        RELATIONSHIP TO SAFETY:
        Urgency levels are prominently featured because safety is paramount. A "door" is different
        from a "vehicle approaching" - this function ensures hazards are clearly communicated,
        supporting the "Safety-Oriented Visual Awareness" goal.
        
        Implements: "Stairs 3 meters ahead, slightly left"
        
        Arguments:
            class_name: Object class name
            box: [cx, cy, w, h] normalized bounding box
            distance_zone: 0 (near), 1 (medium), 2 (far)
            urgency: Urgency level (0-3)
            priority: Optional priority score (0-100)
            verbosity: Override default verbosity
        
        Returns:
            Natural language description
        """verbosity = verbosity or self.verbosity..."""
        Generate overall scene description from multiple detections.
        
        Implements: "Three people approaching from left, vehicle approaching right"
        
        Arguments:
            detections: List of detection dictionaries with class_name, box, distance, urgency
            urgency_score: Overall scene urgency
            verbosity: Override default verbosity
        
        Returns:
            Scene description string
        """verbosity = verbosity or self.verbosity..."""
        Generate navigation guidance with path suggestions.
        
        WHY NAVIGATION GUIDANCE IS CRITICAL:
        This function directly addresses the core problem: helping users navigate safely when they
        cannot see obstacles. A sighted person can instantly see "obstacle on left, clear path right"
        - this function provides that same information through language.
        
        This is not just about detecting objects - it's about providing actionable navigation advice
        that prevents collisions, falls, and disorientation. This supports the "Safety-Oriented Visual
        Awareness" goal and is essential for independent mobility.
        
        HOW IT CONNECTS TO THE PROBLEM STATEMENT:
        The problem asks: "What are ways that those who cannot see... be able to interact with the
        world like those who can?" Navigation guidance is a direct answer - it provides the spatial
        awareness that sighted people take for granted, enabling safe, independent movement.
        
        RELATIONSHIP TO OTHER FEATURES:
        - Works with urgency scoring to prioritize hazards
        - Integrates with distance estimation to focus on immediate obstacles
        - Feeds into CrossModalScheduler for haptic/audio alerts
        - Supports the "Navigation Assistance" feature from Sprint 3
        
        Implements: "Clear path ahead" or "Obstacle on left, move right"
        
        Arguments:
            detections: List of detections
            target_direction: Optional target direction ('forward', 'left', 'right')
        
        Returns:
            Navigation guidance string
        """if not detections:..."""
        Generate urgent hazard alert.
        
        Implements: "Warning: Vehicle approaching from right"
        
        Arguments:
            class_name: Object class
            box: Bounding box
            distance_zone: Distance zone
            urgency: Urgency level
        
        Returns:
            Alert string
        """h_dir, _ = self.get_direction_from_box(box)..."""
        Generate description from detections (wrapper for generate_scene_description).
        
        This method provides backward compatibility for code that calls generate_description()
        instead of generate_scene_description().
        
        Arguments:
            detections: List of detection dictionaries with class_name, box, distance, urgency
            urgency_score: Overall scene urgency
            verbosity: Override default verbosity
        
        Returns:
            Scene description string
        """return self.generate_scene_description(detections, urgency_score, verbosity)


def create_description_generator(verbosity: str = 'normal') -> DescriptionGenerator:"""Factory function to create description generator."""
    return DescriptionGenerator(verbosity=verbosity)

