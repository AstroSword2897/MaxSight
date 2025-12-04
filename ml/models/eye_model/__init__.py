"""
Eye/Face Micro-Model

Tiny CNN for eye tracking and fatigue detection:
- Blink probability
- Fixation vs saccade patterns
- Pupil-size proxy

See docs/therapy_system_implementation_plan.md Phase 1 for implementation details.
"""

from .eye_model import EyeModel

__all__ = ['EyeModel']

