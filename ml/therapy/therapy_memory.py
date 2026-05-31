"""Therapy memory: short-term and long-term state for the closed-loop therapy engine."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShortTermMemory:
    """Rolling window of recent state for response evaluation and cooldowns."""

    last_intervention_type: str | None = None
    last_intervention_time: float = -1e9
    recent_stress_levels: list[float] = field(default_factory=list)
    recent_prompts: list[dict[str, Any]] = field(default_factory=list)
    max_stress_history: int = 30
    max_prompt_history: int = 20

    def push_stress(self, level: float) -> None:
        self.recent_stress_levels.append(level)
        if len(self.recent_stress_levels) > self.max_stress_history:
            self.recent_stress_levels.pop(0)

    def push_prompt(
        self, intervention_type: str, content: str, timestamp: float | None = None
    ) -> None:
        t = timestamp if timestamp is not None else time.time()
        self.last_intervention_type = intervention_type
        self.last_intervention_time = t
        self.recent_prompts.append({"type": intervention_type, "content": content, "time": t})
        if len(self.recent_prompts) > self.max_prompt_history:
            self.recent_prompts.pop(0)

    def stress_trend(self) -> float:
        """Positive = stress increasing, negative = decreasing. Zero if not enough data."""
        if len(self.recent_stress_levels) < 5:
            return 0.0
        half = len(self.recent_stress_levels) // 2
        recent_avg = sum(self.recent_stress_levels[-half:]) / half
        older_avg = sum(self.recent_stress_levels[:-half]) / (len(self.recent_stress_levels) - half)
        return recent_avg - older_avg


@dataclass
class LongTermMemory:
    """Persistent preferences and patterns (e.g. preferred channel, stress triggers)."""

    preferred_channel: str = "audio"  # audio, haptic, visual
    stress_triggers: list[str] = field(default_factory=lambda: ["crowded areas", "noise"])
    successful_interventions: dict[str, float] = field(default_factory=dict)  # type -> success rate
    failed_intervention_types: list[str] = field(default_factory=list)
    user_tolerance_level: float = 0.5  # 0 = low tolerance (fewer prompts), 1 = high

    def record_success(self, intervention_type: str) -> None:
        rate = self.successful_interventions.get(intervention_type, 0.5)
        self.successful_interventions[intervention_type] = min(1.0, rate + 0.05)

    def record_failure(self, intervention_type: str) -> None:
        if intervention_type not in self.failed_intervention_types:
            self.failed_intervention_types.append(intervention_type)
        if len(self.failed_intervention_types) > 50:
            self.failed_intervention_types = self.failed_intervention_types[-50:]

    def get_success_rate(self, intervention_type: str) -> float:
        return self.successful_interventions.get(intervention_type, 0.5)


class TherapyMemorySystem:
    """Short-term + long-term therapy memory for adaptation and safety."""

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def update_after_intervention(
        self, intervention_type: str, content: str, effectiveness: float
    ) -> None:
        self.short_term.push_prompt(intervention_type, content)
        if effectiveness >= 0.5:
            self.long_term.record_success(intervention_type)
        else:
            self.long_term.record_failure(intervention_type)

    def update_stress(self, level: float) -> None:
        self.short_term.push_stress(level)
