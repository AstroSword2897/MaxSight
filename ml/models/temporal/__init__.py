"""
Temporal Encoder Module

Handles temporal processing of video sequences:
- Motion features
- Temporal consistency
- Flicker detection

See docs/therapy_system_implementation_plan.md Phase 1 for implementation details.
"""

from .temporal_encoder import TemporalEncoder, TemporalBuffer

__all__ = ['TemporalEncoder', 'TemporalBuffer']

