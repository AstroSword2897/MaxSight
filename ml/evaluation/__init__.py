"""Evaluation metrics module for Phase 9."""

from __future__ import annotations

from typing import Any

# Defer metrics (torch) so safety_gates imports work under the contracts CI profile.
__all__ = ["EvaluationMetrics", "MultiModalMetrics", "AccessibilityMetrics", "RobustnessMetrics"]

_METRICS_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _METRICS_EXPORTS:
        from . import metrics as _metrics

        return getattr(_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
