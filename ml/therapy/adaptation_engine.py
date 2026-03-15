"""Adaptation Engine: personalize therapy from response evaluation (which prompts work, tolerance)."""

from typing import Dict, Any, Optional
from ml.therapy.therapy_memory import TherapyMemorySystem, LongTermMemory


class AdaptationEngine:
    """
    Updates long-term preferences from effectiveness. Learns which prompts work and
    user tolerance so the decision engine can prefer successful interventions.
    """

    def __init__(self, memory: Optional[TherapyMemorySystem] = None):
        self.memory = memory or TherapyMemorySystem()

    def update(
        self,
        intervention_type: str,
        content: str,
        effectiveness_score: float,
        context_after: Dict[str, Any],
    ) -> None:
        """Record outcome and update memory (success/failure, optional channel preference)."""
        self.memory.update_after_intervention(intervention_type, content, effectiveness_score)
        if effectiveness_score >= 0.6:
            self.memory.long_term.record_success(intervention_type)
        else:
            self.memory.long_term.record_failure(intervention_type)

    def get_preferred_channel(self) -> str:
        return self.memory.long_term.preferred_channel

    def set_preferred_channel(self, channel: str) -> None:
        if channel in ("audio", "haptic", "visual"):
            self.memory.long_term.preferred_channel = channel

    def get_best_intervention_type_for_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Return intervention type with highest success rate in memory (for policy use)."""
        rates = self.memory.long_term.successful_interventions
        if not rates:
            return None
        best = max(rates.items(), key=lambda x: x[1])
        return best[0]
