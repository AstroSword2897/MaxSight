"""Response Evaluation: before_state, intervention, after_state → effectiveness score."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


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

    def evaluate(
        self,
        before_state: Dict[str, Any],
        intervention_type: str,
        after_state: Dict[str, Any],
    ) -> ResponseEvaluationResult:
        """
        Compare stress/cognitive load before vs after. Positive delta → effectiveness.
        """
        stress_before = before_state.get("environment_stress_level", 0.5)
        stress_after = after_state.get("environment_stress_level", 0.5)
        if isinstance(stress_before, (list, tuple)) and stress_before:
            stress_before = float(stress_before[-1]) if stress_before else 0.5
        if isinstance(stress_after, (list, tuple)) and stress_after:
            stress_after = float(stress_after[-1]) if stress_after else 0.5
        stress_reduction = float(stress_before - stress_after)

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
