"""Hybrid therapy scoring: deterministic base + learned correction layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from ml.therapy.constraints_loader import TherapyConstraints
from ml.therapy.situation_understanding import SituationContext

ContextInput = SituationContext | dict[str, float]


@dataclass
class ScoreTrace:
    """Explainable score decomposition for SCRUM-19 response traces."""

    base_score: float
    stress_component: float
    effectiveness_component: float
    safety_penalty: float
    learned_adjustment: float
    final_score: float
    factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "base_score": self.base_score,
            "stress_component": self.stress_component,
            "effectiveness_component": self.effectiveness_component,
            "safety_penalty": self.safety_penalty,
            "learned_adjustment": self.learned_adjustment,
            "final_score": self.final_score,
            **self.factors,
        }


class TherapyScoringModel:
    """Deterministic + lightweight learned hybrid scorer."""

    def __init__(self, constraints: TherapyConstraints | None = None) -> None:
        self.constraints = constraints or TherapyConstraints.load()
        self._effectiveness_memory: dict[str, float] = {}

    def update_effectiveness(self, intervention_type: str, score: float) -> None:
        """Record observed effectiveness for learned adjustment."""
        prev = self._effectiveness_memory.get(intervention_type, 0.5)
        self._effectiveness_memory[intervention_type] = 0.7 * prev + 0.3 * float(score)

    def effectiveness_snapshot(self) -> dict[str, float]:
        """Return per-intervention running effectiveness means for observability."""
        return dict(self._effectiveness_memory)

    @staticmethod
    def _read_context(context: ContextInput) -> tuple[float, float, float]:
        if isinstance(context, SituationContext):
            return (
                context.environment_stress_level,
                context.cognitive_load_estimate,
                context.uncertainty,
            )
        return (
            float(context.get("environment_stress_level", 0.0)),
            float(context.get("cognitive_load_estimate", 0.0)),
            float(context.get("uncertainty", 0.0)),
        )

    def score_intervention(
        self,
        context: ContextInput,
        intervention_type: str,
        *,
        prior_effectiveness: float | None = None,
    ) -> ScoreTrace:
        """Compute hybrid score with full trace."""
        weights = self.constraints.scoring_weights
        stress, cognitive, uncertainty = self._read_context(context)

        stress_component = (1.0 - stress) * weights.get("stress_reduction", 0.4)
        eff = prior_effectiveness
        if eff is None:
            eff = self._effectiveness_memory.get(intervention_type, 0.5)
        effectiveness_component = float(eff) * weights.get("user_engagement", 0.2)

        safety_penalty = 0.0
        if uncertainty > self.constraints.suppress_threshold:
            safety_penalty = 0.5

        learned_adjustment = self._effectiveness_memory.get(intervention_type, 0.5) * 0.1
        base_score = max(0.0, 1.0 - cognitive * 0.5)
        final = max(
            0.0,
            min(
                1.0,
                base_score
                + stress_component
                + effectiveness_component
                + learned_adjustment
                - safety_penalty,
            ),
        )
        return ScoreTrace(
            base_score=base_score,
            stress_component=stress_component,
            effectiveness_component=effectiveness_component,
            safety_penalty=safety_penalty,
            learned_adjustment=learned_adjustment,
            final_score=final,
            factors={"stress": stress, "cognitive_load": cognitive, "uncertainty": uncertainty},
        )

    def recommend_intervention_type(
        self,
        disability_id: str,
        context: ContextInput,
    ) -> str:
        """Route intervention using ontology + score."""
        routing = self.constraints.disability_routing.get(disability_id, ["grounding"])
        best = routing[0]
        best_score = -1.0
        for candidate in routing:
            trace = self.score_intervention(context, candidate)
            if trace.final_score > best_score:
                best_score = trace.final_score
                best = candidate
        return best
