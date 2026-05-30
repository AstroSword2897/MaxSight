"""Adaptive therapy task generation from user performance signals."""

from typing import Dict, List, Optional, Any
from enum import Enum


class TaskType(Enum):
    """Therapy task categories used by the task generator."""
    CONTRAST_MICRO = "contrast_micro"  # Edge finding.
    MOTION_TRACKING = "motion_tracking"
    DEPTH_SHIFT = "depth_shift"  # Focus near→far→near.
    GAZE_STABILIZATION = "gaze_stabilization"
    ROI_FINDABILITY = "roi_findability"
    FATIGUE_REST = "fatigue_rest"


class TaskGenerator:
    """Generate therapy tasks using profile, performance, and failure history."""

    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """Initialize generator state.

        Parameters:
            user_profile: Optional dict with keys like ``duration_bias``.
        """
        self.user_profile = user_profile or {}
        self.recent_failures: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
    
    def generate_task(
        self,
        uncertainty: float,
        fatigue_score: float,
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate the next therapy task specification.

        Parameters:
            uncertainty: Perception uncertainty in ``[0, 1]``.
            fatigue_score: Fatigue estimate; values above 0.7 yield rest tasks.
            recent_performance: Recent attempt records with success/failure flags.

        Returns:
            Task dict with ``task_type``, ``difficulty``, ``duration``, and tuning fields.

        Side effects:
            Appends the generated task to ``task_history``.
        """
        # If fatigued, suggest rest task.
        if fatigue_score > 0.7:
            rest_task = {
                'task_type': TaskType.FATIGUE_REST,
                'difficulty': 0.0,
                'duration': 60,
                'highlight_strength': 0.0,
                'target_speed': 0.0
            }
            self.task_history.append(rest_task)
            return rest_task
        
        # If high uncertainty, reduce difficulty.
        base_difficulty = max(0.1, min(1.0, 1.0 - uncertainty))
        if recent_performance:
            success_rate = sum(1 for item in recent_performance if item.get("success")) / max(1, len(recent_performance))
            base_difficulty = max(0.1, min(1.0, base_difficulty * (0.75 + 0.5 * success_rate)))
        if self.recent_failures:
            base_difficulty = max(0.1, base_difficulty - min(0.2, 0.02 * len(self.recent_failures)))
        
        # Choose task type based on recent failures.
        task_type = self._choose_task_type(recent_performance)
        duration_bias = float(self.user_profile.get("duration_bias", 1.0))
        task = {
            'task_type': task_type,
            'difficulty': base_difficulty,
            'duration': int((30 + (1.0 - base_difficulty) * 30) * max(0.7, min(1.3, duration_bias))),
            'highlight_strength': base_difficulty,
            'target_speed': base_difficulty * 100.0  # Pixels/second.
        }
        self.task_history.append(task)
        return task
    
    def _choose_task_type(self, recent_performance: List[Dict[str, Any]]) -> TaskType:
        """Choose task type based on recent performance."""
        if not recent_performance and not self.task_history:
            return TaskType.CONTRAST_MICRO

        failure_counts: Dict[TaskType, int] = {}
        for item in recent_performance[-20:]:
            task_raw = item.get("task_type")
            if isinstance(task_raw, TaskType):
                task_key = task_raw
            elif isinstance(task_raw, str):
                try:
                    task_key = TaskType(task_raw)
                except ValueError:
                    continue
            else:
                continue
            if item.get("failed", False) or not item.get("success", True):
                failure_counts[task_key] = failure_counts.get(task_key, 0) + 1
        for item in self.recent_failures[-10:]:
            task_raw = item.get("task_type")
            if isinstance(task_raw, TaskType):
                failure_counts[task_raw] = failure_counts.get(task_raw, 0) + 1

        task_order: List[TaskType] = [
            TaskType.CONTRAST_MICRO,
            TaskType.MOTION_TRACKING,
            TaskType.DEPTH_SHIFT,
            TaskType.GAZE_STABILIZATION,
            TaskType.ROI_FINDABILITY
        ]
        if failure_counts:
            return max(task_order, key=lambda task: failure_counts.get(task, 0))
        if not self.task_history:
            return TaskType.CONTRAST_MICRO
        last_task = self.task_history[-1]
        try:
            current_idx = task_order.index(last_task['task_type'])
            next_idx = (current_idx + 1) % len(task_order)
            return task_order[next_idx]
        except (ValueError, KeyError):
            return TaskType.CONTRAST_MICRO
    
    def update_performance(self, task_result: Dict[str, Any]):
        """Update task generator with task result."""
        self.task_history.append(task_result)
        if task_result.get('failed', False):
            self.recent_failures.append(task_result)
            # Keep only recent failures.
            if len(self.recent_failures) > 10:
                self.recent_failures.pop(0)







