"""Personalize therapy delivery from intervention effectiveness history."""

from typing import cast

from ml.therapy.situation_understanding import SituationContext
from ml.therapy.therapy_memory import TherapyMemorySystem


class AdaptationEngine:
    """Update therapy memory and channel preferences from response outcomes."""

    def __init__(self, memory: TherapyMemorySystem | None = None):
        """Attach or create therapy memory for adaptation updates."""
        self.memory = memory or TherapyMemorySystem()

    def update(
        self,
        intervention_type: str,
        content: str,
        effectiveness_score: float,
        context_after: SituationContext,
    ) -> None:
        """Record intervention outcome in long-term memory.

        Parameters:
            intervention_type: Intervention identifier string.
            content: Delivered prompt content.
            effectiveness_score: Score in ``[0, 1]`` from response evaluation.
            context_after: Post-intervention situation context.
        """
        self.memory.update_after_intervention(intervention_type, content, effectiveness_score)

    def get_preferred_channel(self) -> str:
        """Return the user's preferred delivery channel from long-term memory."""
        return str(self.memory.long_term.preferred_channel)

    def set_preferred_channel(self, channel: str) -> None:
        """Set preferred channel when it is audio, haptic, or visual."""
        if channel in ("audio", "haptic", "visual"):
            self.memory.long_term.preferred_channel = channel

    def get_best_intervention_type_for_context(self, context: SituationContext) -> str | None:
        """Return the best intervention type for the current stress context.

        Parameters:
            context: Typed situation context with stress and cognitive load estimates.

        Returns:
            Intervention type string, or ``None`` when memory is empty.
        """
        rates = self.memory.long_term.successful_interventions
        if not rates:
            return None
        stress = context.environment_stress_level
        cognitive_load = context.cognitive_load_estimate
        if stress >= 0.7 or cognitive_load >= 0.7:
            prioritized = ["grounding", "breathing", "calming", "attention_redirection"]
        else:
            prioritized = [
                "navigation_reassurance",
                "attention_redirection",
                "grounding",
                "rest_suggestion",
            ]
        candidates = [(kind, rates.get(kind, 0.0)) for kind in prioritized if kind in rates]
        if candidates:
            return cast(str, max(candidates, key=lambda item: item[1])[0])
        return cast(str, max(rates.items(), key=lambda item: item[1])[0])
