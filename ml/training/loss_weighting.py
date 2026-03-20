"""Loss weighting utilities for stable temporal training rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TemporalWeightSchedule:
    """Linear warmup schedule for temporal objectives."""

    start_epoch: int = 0
    warmup_epochs: int = 10
    start_weight: float = 0.1
    target_weight: float = 0.6

    def validate(self) -> None:
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be >= 0")
        if self.warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1")
        if self.start_weight < 0 or self.target_weight < 0:
            raise ValueError("weights must be >= 0")

    def at_epoch(self, epoch: int) -> float:
        """Compute scheduled temporal weight at a given epoch."""
        self.validate()
        if epoch <= self.start_epoch:
            return float(self.start_weight)
        delta = epoch - self.start_epoch
        if delta >= self.warmup_epochs:
            return float(self.target_weight)
        ratio = float(delta) / float(self.warmup_epochs)
        return float(self.start_weight + ratio * (self.target_weight - self.start_weight))


def build_temporal_weight_updates(
    epoch: int,
    schedule: TemporalWeightSchedule,
    temporal_heads: Dict[str, float],
) -> Dict[str, float]:
    """Generate per-head updates for temporal losses in current epoch."""
    weight = schedule.at_epoch(epoch)
    return {head: float(base * weight) for head, base in temporal_heads.items()}

