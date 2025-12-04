"""
Therapy Heads Module - Phase 2 Stubs

This module contains stubs for therapy-specific output heads:
- ContrastMapHead: Contrast sensitivity mapping
- MotionHead: Motion/flow detection
- DepthHead: Depth/focus estimation
- ROIPriorityHead: Region-of-interest prioritization
- FatigueHead: Fatigue/gaze tracking
- UncertaintyHead: Confidence/uncertainty estimation

Status: Phase 2 (Sprint 2) - Not yet integrated into main model
These are placeholder implementations for future development.

Usage:
    from ml.models.heads import ContrastMapHead, MotionHead, DepthHead
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
