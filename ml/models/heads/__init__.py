"""
Therapy Heads Module

Contains all therapy-specific output heads:
- Contrast Map Head
- Motion/Flow Head
- Depth/Focus Head
- ROI Priority Head
- Fatigue/Gaze Head
- Confidence/Uncertainty Head

See docs/therapy_system_implementation_plan.md for implementation details.
"""

from .contrast_head import ContrastMapHead
from .motion_head import MotionHead
from .depth_head import DepthHead
from .roi_priority_head import ROIPriorityHead
from .fatigue_head import FatigueHead
from .uncertainty_head import UncertaintyHead

__all__ = [
    'ContrastMapHead',
    'MotionHead',
    'DepthHead',
    'ROIPriorityHead',
    'FatigueHead',
    'UncertaintyHead'
]

