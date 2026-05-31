"""Therapy Decision Engine: rule + policy gate. Decides should we intervene, what, how strong."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ml.therapy.adaptation_engine import AdaptationEngine
from ml.therapy.intervention_generator import InterventionType
from ml.therapy.situation_understanding import SituationContext

if TYPE_CHECKING:
    from ml.therapy.scoring import TherapyScoringModel


@dataclass(frozen=True)
class InterventionStrength:
    """Named intervention strength constants for auditable tuning."""

    HIGH_STRESS: float = 0.8
    MODERATE_STRESS_BASE: float = 0.4
    NAV_STRESS_BASE: float = 0.5
    STRESS_SCALE_FACTOR: float = 0.3
    COGNITIVE_LOAD: float = 0.5
    NAV_REASSURANCE: float = 0.5
    NAV_COMPLEXITY_THRESHOLD: float = 0.5
    COGNITIVE_LOAD_THRESHOLD: float = 0.7
    LOW_STRESS_NAV_THRESHOLD: float = 0.4
    HIGH_NAV_COMPLEXITY_THRESHOLD: float = 0.6


_DEFAULT_INTERVENTION_STRENGTH = InterventionStrength()


@dataclass
class TherapyDecision:
    """Output of the decision engine."""

    should_intervene: bool
    intervention_type: str
    strength: float
    reason: str


class TherapyDecisionEngine:
    """Rule layer plus scoring-model policy for therapy intervention decisions.

    Rule layer handles must-fire conditions (safety, threshold gates). When
    those rules are satisfied and ``scoring_model`` is provided, the model
    selects the optimal intervention type using learned + constraint weights.
    """

    def __init__(
        self,
        stress_trigger_threshold: float = 0.6,
        high_stress_threshold: float = 0.75,
        scoring_model: Optional["TherapyScoringModel"] = None,
        strength: InterventionStrength | None = None,
    ):
        """Initialise with thresholds; attach optional learned scoring model."""
        self.stress_trigger_threshold = stress_trigger_threshold
        self.high_stress_threshold = high_stress_threshold
        self.scoring_model = scoring_model
        self.strength = strength or _DEFAULT_INTERVENTION_STRENGTH

    def decide(
        self,
        context: SituationContext,
        adaptation_engine: AdaptationEngine | None = None,
        current_time: float = 0.0,
    ) -> TherapyDecision:
        """Decide whether and how to intervene given the current situation context.

        Parameters:
            context: Typed situation context from ``SituationUnderstandingLayer``.
            adaptation_engine: Optional engine for preference-based routing.
            current_time: Unix timestamp for rate-limiting downstream.

        Returns:
            ``TherapyDecision`` specifying whether to intervene, which type, and strength.
        """
        stress = context.environment_stress_level
        nav_complexity = context.navigation_complexity
        cognitive_load = context.cognitive_load_estimate
        disability_id = context.disability_id

        if stress >= self.high_stress_threshold:
            best_type = self._select_type(
                context,
                disability_id,
                adaptation_engine,
                fallback=InterventionType.CALMING_PROMPT.value,
            )
            return TherapyDecision(
                should_intervene=True,
                intervention_type=best_type,
                strength=self.strength.HIGH_STRESS,
                reason="high_stress_preferred_intervention"
                if best_type != InterventionType.CALMING_PROMPT.value
                else "high_stress_calming",
            )
        if stress >= self.stress_trigger_threshold:
            if nav_complexity > self.strength.NAV_COMPLEXITY_THRESHOLD:
                best_type = self._select_type(
                    context,
                    disability_id,
                    adaptation_engine,
                    fallback=InterventionType.NAVIGATION_REASSURANCE.value,
                )
                return TherapyDecision(
                    should_intervene=True,
                    intervention_type=best_type,
                    strength=self.strength.NAV_STRESS_BASE
                    + stress * self.strength.STRESS_SCALE_FACTOR,
                    reason="stress_and_navigation",
                )
            best_type = self._select_type(
                context,
                disability_id,
                adaptation_engine,
                fallback=InterventionType.GROUNDING_PROMPT.value,
            )
            return TherapyDecision(
                should_intervene=True,
                intervention_type=best_type,
                strength=self.strength.MODERATE_STRESS_BASE
                + stress * self.strength.STRESS_SCALE_FACTOR,
                reason="moderate_stress_grounding",
            )
        if cognitive_load > self.strength.COGNITIVE_LOAD_THRESHOLD:
            best_type = self._select_type(
                context,
                disability_id,
                adaptation_engine,
                fallback=InterventionType.ATTENTION_REDIRECTION.value,
            )
            return TherapyDecision(
                should_intervene=True,
                intervention_type=best_type,
                strength=self.strength.COGNITIVE_LOAD,
                reason="high_cognitive_load",
            )
        if (
            stress > self.strength.LOW_STRESS_NAV_THRESHOLD
            and nav_complexity > self.strength.HIGH_NAV_COMPLEXITY_THRESHOLD
        ):
            return TherapyDecision(
                should_intervene=True,
                intervention_type=InterventionType.NAVIGATION_REASSURANCE.value,
                strength=self.strength.NAV_REASSURANCE,
                reason="navigation_reassurance",
            )

        return TherapyDecision(
            should_intervene=False,
            intervention_type="",
            strength=0.0,
            reason="below_threshold",
        )

    def _select_type(
        self,
        context: SituationContext,
        disability_id: str,
        adaptation_engine: AdaptationEngine | None,
        fallback: str,
    ) -> str:
        """Return the best intervention type from scoring → adaptation → fallback chain."""
        if self.scoring_model is not None and disability_id:
            return self.scoring_model.recommend_intervention_type(disability_id, context)
        if adaptation_engine is not None:
            best = adaptation_engine.get_best_intervention_type_for_context(context)
            if best:
                return best
        return fallback
