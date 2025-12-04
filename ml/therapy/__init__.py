"""
Therapy System Module

Contains therapy task generation and session management:
- Task generator
- Session manager
- Task difficulty scaling
- Performance logging

See docs/therapy_system_implementation_plan.md Phase 3 for implementation details.
"""

from .task_generator import TaskGenerator, TaskType
from .session_manager import SessionManager

__all__ = ['TaskGenerator', 'TaskType', 'SessionManager']

