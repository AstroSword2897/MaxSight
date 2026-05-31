"""Session lifecycle management for therapy tasks."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionManager:
    """Track therapy sessions, task attempts, metrics, and persistence."""

    def __init__(self, user_id: str | None = None):
        """Initialize session manager for an optional user.

        Parameters:
            user_id: User identifier stored on new sessions.
        """
        self.user_id = user_id
        self.current_session: dict[str, Any] | None = None
        self.session_history: list[dict[str, Any]] = []
        self.task_attempts: list[dict[str, Any]] = []

    def start_session(self, session_config: dict[str, Any] | None = None) -> str:
        """Start a new therapy session.

        Parameters:
            session_config: Optional metadata stored on the session record.

        Returns:
            Generated session ID string.

        Side effects:
            Sets ``current_session`` and resets in-session task list.
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = {
            "session_id": session_id,
            "user_id": self.user_id,
            "start_time": datetime.now().isoformat(),
            "config": session_config or {},
            "tasks": [],
            "metrics": {
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_time": 0.0,
                "reaction_count": 0,
            },
        }
        return session_id

    def log_task_attempt(
        self, task_type: str, task_config: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Append a task attempt and update running session metrics.

        Parameters:
            task_type: Task identifier string.
            task_config: Task parameters for the attempt.
            result: Outcome dict with ``success`` and optional ``reaction_time``.

        Side effects:
            Auto-starts a session when none is active.
        """
        if not self.current_session:
            self.start_session(session_config={"auto_started": True, "user_id": self.user_id})

        if self.current_session is None:
            raise RuntimeError("Failed to initialize session")

        attempt = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "task_config": task_config,
            "result": result,
        }

        self.task_attempts.append(attempt)
        self.current_session["tasks"].append(attempt)

        # Update metrics.
        self.current_session["metrics"]["total_tasks"] += 1
        if result.get("success", False):
            self.current_session["metrics"]["completed_tasks"] += 1
        else:
            self.current_session["metrics"]["failed_tasks"] += 1

        reaction_time = result.get("reaction_time")
        if reaction_time is not None:
            self.current_session["metrics"]["total_time"] += float(reaction_time)
            self.current_session["metrics"]["reaction_count"] += 1

    def end_session(self) -> dict[str, Any]:
        """Finalize the active session and return a report dict.

        Returns:
            Session report including skill curve and summary metrics.

        Side effects:
            Clears ``current_session``, appends to history, and flushes history.

        Failure modes:
            Raises ``RuntimeError`` when no session is active.
        """
        if not self.current_session:
            raise RuntimeError("No active session to end.")

        self.current_session["end_time"] = datetime.now().isoformat()

        # Generate skill curve.
        skill_curve = self._generate_skill_curve()

        report = {
            **self.current_session,
            "skill_curve": skill_curve,
            "summary": self._generate_summary(),
        }

        self.session_history.append(report)
        self.current_session = None
        self.task_attempts = []
        self.flush_history()

        return report

    def _generate_skill_curve(self) -> list[dict[str, Any]]:
        """Generate skill progression curve from session tasks."""
        if self.current_session is None:
            return []

        curve = []
        for i, task in enumerate(self.current_session["tasks"]):
            success = task["result"].get("success", False)
            reaction_time = task["result"].get("reaction_time", 0.0)
            curve.append(
                {
                    "task_index": i,
                    "success": success,
                    "reaction_time": reaction_time,
                    "cumulative_success_rate": sum(
                        1
                        for t in self.current_session["tasks"][: i + 1]
                        if t["result"].get("success", False)
                    )
                    / (i + 1),
                }
            )
        return curve

    def _generate_summary(self) -> dict[str, Any]:
        """Generate session summary."""
        if self.current_session is None:
            return {}

        metrics = self.current_session["metrics"]
        total = metrics["total_tasks"]

        if total == 0:
            return {"success_rate": 0.0, "avg_reaction_time": 0.0}

        success_rate = metrics["completed_tasks"] / total
        reaction_count = max(1, metrics.get("reaction_count", 0))
        avg_reaction_time = metrics["total_time"] / reaction_count

        return {
            "success_rate": success_rate,
            "avg_reaction_time": avg_reaction_time,
            "total_tasks": total,
            "completed_tasks": metrics["completed_tasks"],
            "failed_tasks": metrics["failed_tasks"],
        }

    def save_session(self, filepath: str) -> None:
        """Persist the active session JSON to disk.

        Failure modes:
            Raises ``RuntimeError`` when no active session exists.
            Raises ``OSError`` when file I/O fails.
        """
        if self.current_session is None:
            raise RuntimeError("No active session to save.")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.current_session, f, indent=2)
        except OSError as exc:
            raise OSError(f"Failed to save session to {filepath}: {exc}") from exc

    def flush_history(self, filepath: str = "session_history.jsonl") -> None:
        """Append completed session reports to JSONL and clear in-memory history.

        Parameters:
            filepath: Destination JSONL path (created if missing).

        Failure modes:
            Raises ``OSError`` on write failure.
        """
        if not self.session_history:
            return
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as handle:
                for session in self.session_history:
                    handle.write(json.dumps(session) + "\n")
            self.session_history.clear()
        except OSError as exc:
            raise OSError(f"Failed to flush session history to {path}: {exc}") from exc
