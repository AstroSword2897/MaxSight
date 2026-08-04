"""Response Evaluation: before_state, intervention, after_state → effectiveness score."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseEvaluationResult:
    """Result of evaluating whether an intervention worked."""

    effectiveness_score: float  # 0-1
    stress_reduction: float
    reason: str


class ResponseEvaluationModel:
    """
    Determines if the intervention worked. Lightweight rule-based implementation;
    can be replaced by a small MLP for learned evaluation later.
    """

    @staticmethod
    def _as_float(value: Any, default: float = 0.5) -> float:
        """Coerce nested/list stress proxies to a scalar float."""
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[-1]
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def evaluate(
        self,
        before_state: dict[str, Any],
        intervention_type: str,
        after_state: dict[str, Any],
    ) -> ResponseEvaluationResult:
        """
        Compare stress/cognitive load before vs after. Positive delta → effectiveness.
        """
        stress_before = self._as_float(before_state.get("environment_stress_level", 0.5))
        stress_after = self._as_float(after_state.get("environment_stress_level", 0.5))
        stress_reduction = stress_before - stress_after

        if stress_reduction > 0.15:
            effectiveness = min(1.0, 0.5 + stress_reduction * 2.0)
            reason = "stress_reduced"
        elif stress_reduction > 0.0:
            effectiveness = 0.5 + stress_reduction
            reason = "slight_improvement"
        elif stress_reduction > -0.1:
            effectiveness = 0.4
            reason = "neutral"
        else:
            effectiveness = max(0.0, 0.3 + stress_reduction)
            reason = "stress_increased"

        return ResponseEvaluationResult(
            effectiveness_score=max(0.0, min(1.0, effectiveness)),
            stress_reduction=stress_reduction,
            reason=reason,
        )
