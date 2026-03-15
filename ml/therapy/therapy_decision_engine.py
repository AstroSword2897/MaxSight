"""Therapy Decision Engine: rule + policy gate. Decides should we intervene, what, how strong."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from ml.therapy.intervention_generator import InterventionType


@dataclass
class TherapyDecision:
    """Output of the decision engine."""
    should_intervene: bool
    intervention_type: str
    strength: float
    reason: str


class TherapyDecisionEngine:
    """
    Core brain of the therapy subsystem. Rule layer (safety constraints) plus simple policy:
    high stress → calming or grounding; navigation complexity → reassurance; etc.
    ML policy can be plugged in later for learned optimal interventions.
    """

    def __init__(
        self,
        stress_trigger_threshold: float = 0.6,
        high_stress_threshold: float = 0.75,
    ):
        self.stress_trigger_threshold = stress_trigger_threshold
        self.high_stress_threshold = high_stress_threshold

    def decide(
        self,
        context: Dict[str, Any],
        adaptation_engine: Optional[Any] = None,
        current_time: float = 0.0,
    ) -> TherapyDecision:
        """
        Decide whether to intervene and with what. Context is from SituationUnderstandingLayer.
        adaptation_engine can provide preferred intervention types from past effectiveness.
        """
        stress = context.get("environment_stress_level", 0.0)
        uncertainty = context.get("uncertainty", 0.0)
        nav_complexity = context.get("navigation_complexity", 0.0)
        cognitive_load = context.get("cognitive_load_estimate", 0.0)

        if stress >= self.high_stress_threshold:
            best_type = (adaptation_engine.get_best_intervention_type_for_context(context)
                         if adaptation_engine is not None else None)
            if best_type:
                return TherapyDecision(
                    should_intervene=True,
                    intervention_type=best_type,
                    strength=0.8,
                    reason="high_stress_preferred_intervention",
                )
            return TherapyDecision(
                should_intervene=True,
                intervention_type=InterventionType.CALMING_PROMPT.value,
                strength=0.8,
                reason="high_stress_calming",
            )
        if stress >= self.stress_trigger_threshold:
            if nav_complexity > 0.5:
                return TherapyDecision(
                    should_intervene=True,
                    intervention_type=InterventionType.NAVIGATION_REASSURANCE.value,
                    strength=0.5 + stress * 0.3,
                    reason="stress_and_navigation",
                )
            return TherapyDecision(
                should_intervene=True,
                intervention_type=InterventionType.GROUNDING_PROMPT.value,
                strength=0.4 + stress * 0.3,
                reason="moderate_stress_grounding",
            )
        if cognitive_load > 0.7:
            return TherapyDecision(
                should_intervene=True,
                intervention_type=InterventionType.ATTENTION_REDIRECTION.value,
                strength=0.5,
                reason="high_cognitive_load",
            )
        if stress > 0.4 and nav_complexity > 0.6:
            return TherapyDecision(
                should_intervene=True,
                intervention_type=InterventionType.NAVIGATION_REASSURANCE.value,
                strength=0.5,
                reason="navigation_reassurance",
            )

        return TherapyDecision(
            should_intervene=False,
            intervention_type="",
            strength=0.0,
            reason="below_threshold",
        )
