"""
Therapy Engine: closed-loop behavioral feedback system layered on top of perception.

Pipeline:
  Perception → SituationUnderstanding → DecisionEngine → InterventionGenerator
  → Output Scheduler (audio/haptics) → User Response → ResponseEvaluation
  → AdaptationEngine → TherapyMemory
"""

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

from ml.therapy.adaptation_engine import AdaptationEngine
from ml.therapy.intervention_generator import InterventionGenerator, TherapeuticAction
from ml.therapy.response_evaluation import ResponseEvaluationModel
from ml.therapy.scoring import TherapyScoringModel
from ml.therapy.situation_understanding import SituationContext, SituationUnderstandingLayer
from ml.therapy.therapy_decision_engine import TherapyDecision, TherapyDecisionEngine
from ml.therapy.therapy_memory import TherapyMemorySystem
from ml.therapy.therapy_safety import TherapySafetyLayer

logger = logging.getLogger(__name__)


@dataclass
class TherapyEngineConfig:
    """Therapy engine thresholds and delivery preferences.

    Attributes:
        stress_trigger_threshold: Minimum stress before moderate interventions.
        high_stress_threshold: Stress level triggering stronger interventions.
        preferred_channel: Default output channel (audio, haptic, visual).
    """

    stress_trigger_threshold: float = 0.6
    high_stress_threshold: float = 0.75
    preferred_channel: str = "audio"


class TherapyEngine:
    """Closed-loop therapy controller from perception to adaptive interventions.

    Call ``update()`` each perception tick to obtain deliverable actions.
    Call ``on_user_response()`` on a subsequent tick so adaptation/memory learn
    from intervention effectiveness.
    """

    def __init__(self, config: TherapyEngineConfig | None = None):
        """Build therapy subsystems and wire preferred delivery channel.

        Parameters:
            config: Optional engine configuration; defaults are safe for production.

        Side effects:
            Initializes memory, safety, adaptation state, and scoring model.
        """
        self.config = config or TherapyEngineConfig()
        self.situation_layer = SituationUnderstandingLayer()
        self.scoring_model = TherapyScoringModel()
        self.decision_engine = TherapyDecisionEngine(
            stress_trigger_threshold=self.config.stress_trigger_threshold,
            high_stress_threshold=self.config.high_stress_threshold,
            scoring_model=self.scoring_model,
        )
        self.intervention_generator = InterventionGenerator(
            preferred_channel=self.config.preferred_channel,
        )
        self.safety = TherapySafetyLayer()
        self.memory = TherapyMemorySystem()
        self.adaptation = AdaptationEngine(memory=self.memory)
        # Initialize adaptation memory from config so the controller's channel
        # selection respects user preference from the first update.
        self.adaptation.set_preferred_channel(self.config.preferred_channel)
        self.response_evaluation = ResponseEvaluationModel()
        self._last_context: SituationContext | None = None
        self._last_decision: TherapyDecision | None = None
        self._last_action: TherapeuticAction | None = None
        self.suppression_counts: dict[str, int] = {}
        self.drop_counts: dict[str, int] = {}

    def update(
        self, perception: dict[str, Any], current_time: float | None = None
    ) -> list[TherapeuticAction]:
        """Run one closed-loop step and return deliverable therapeutic actions.

        Parameters:
            perception: Perception stack outputs (detections, uncertainty, etc.).
            current_time: Optional unix timestamp for rate limiting.

        Returns:
            List of ``TherapeuticAction`` objects (empty when suppressed or no intervention).

        Side effects:
            Updates stress memory, suppression/drop counters, and last action context.

        Failure modes:
            Never raises; suppression and drops are logged and counted.
        """
        try:
            return self._update_impl(perception, current_time)
        except Exception:
            logger.exception("therapy_update_failed module=therapy_engine function=update")
            return []

    def _update_impl(
        self,
        perception: dict[str, Any],
        current_time: float | None = None,
    ) -> list[TherapeuticAction]:
        t = current_time if current_time is not None else time.time()
        context = self.situation_layer.compute(perception)
        self.memory.update_stress(context.environment_stress_level)
        self._last_context = context

        decision = self.decision_engine.decide(
            context,
            adaptation_engine=self.adaptation,
            current_time=t,
        )
        self._last_decision = decision

        if not decision.should_intervene:
            return []

        suppress, reason = self.safety.should_suppress(context, t)
        if suppress:
            self.suppression_counts[reason] = self.suppression_counts.get(reason, 0) + 1
            from ml.training.observability import emit_event

            emit_event(
                "therapy.suppressed",
                module="therapy_engine",
                function="update",
                reason=reason,
                count=self.suppression_counts[reason],
            )
            return []

        action = self.intervention_generator.generate(
            decision.intervention_type,
            decision.strength,
            context,
            channel_override=self.adaptation.get_preferred_channel(),
        )
        if action is None:
            self.drop_counts["generator_none"] = self.drop_counts.get("generator_none", 0) + 1
            logger.warning(
                "therapy_drop module=therapy_engine function=update reason=generator_none intervention=%s count=%d",
                decision.intervention_type,
                self.drop_counts["generator_none"],
            )
            return []
        sanitized = self.safety.sanitize_content(action.content)
        if not sanitized:
            self.drop_counts["sanitized_empty"] = self.drop_counts.get("sanitized_empty", 0) + 1
            logger.warning(
                "therapy_drop module=therapy_engine function=update reason=sanitized_empty intervention=%s count=%d",
                decision.intervention_type,
                self.drop_counts["sanitized_empty"],
            )
            return []
        action = dataclasses.replace(action, content=sanitized)
        self._last_action = action
        self.safety.record_prompt_delivered(t)
        return [action]

    def on_user_response(
        self,
        perception_after: dict[str, Any],
        current_time: float | None = None,
    ) -> None:
        """Evaluate prior intervention effectiveness and update adaptation memory.

        Parameters:
            perception_after: Perception state after the user had time to respond.
            current_time: Optional timestamp (reserved for future temporal logic).

        Side effects:
            Updates adaptation engine and long-term intervention success rates.

        Failure modes:
            Logs and returns when no prior action/context exists.
        """
        if self._last_context is None or self._last_action is None:
            logger.warning(
                "therapy_response_skipped module=therapy_engine function=on_user_response reason=missing_last_state"
            )
            return
        after_context = self.situation_layer.compute(perception_after)
        result = self.response_evaluation.evaluate(
            {
                "environment_stress_level": self._last_context.environment_stress_level,
                "cognitive_load_estimate": self._last_context.cognitive_load_estimate,
            },
            self._last_action.intervention_type,
            {
                "environment_stress_level": after_context.environment_stress_level,
                "cognitive_load_estimate": after_context.cognitive_load_estimate,
            },
        )
        self.adaptation.update(
            self._last_action.intervention_type,
            self._last_action.content,
            result.effectiveness_score,
            after_context,
        )
        self.scoring_model.update_effectiveness(
            self._last_action.intervention_type,
            result.effectiveness_score,
        )

    def get_adaptation_telemetry(self) -> dict[str, Any]:
        """Return closed-loop adaptation state for manifests and debugging."""
        return {
            "effectiveness_by_intervention": self.scoring_model.effectiveness_snapshot(),
            "suppression_counts": dict(self.suppression_counts),
            "drop_counts": dict(self.drop_counts),
        }

    def get_last_context(self) -> SituationContext | None:
        """Return the situation context from the most recent ``update()`` call."""
        return self._last_context

    def get_memory(self) -> TherapyMemorySystem:
        """Return the therapy memory system for inspection or persistence."""
        return self.memory
