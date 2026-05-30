"""
Therapy Engine: closed-loop behavioral feedback system layered on top of perception.

Pipeline:
  Perception → SituationUnderstanding → DecisionEngine → InterventionGenerator
  → Output Scheduler (audio/haptics) → User Response → ResponseEvaluation
  → AdaptationEngine → TherapyMemory
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ml.therapy.situation_understanding import SituationUnderstandingLayer, SituationContext
from ml.therapy.therapy_decision_engine import TherapyDecisionEngine, TherapyDecision
from ml.therapy.intervention_generator import InterventionGenerator, TherapeuticAction
from ml.therapy.therapy_safety import TherapySafetyLayer
from ml.therapy.therapy_memory import TherapyMemorySystem
from ml.therapy.response_evaluation import ResponseEvaluationModel
from ml.therapy.adaptation_engine import AdaptationEngine

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

    def __init__(self, config: Optional[TherapyEngineConfig] = None):
        """Build therapy subsystems and wire preferred delivery channel.

        Parameters:
            config: Optional engine configuration; defaults are safe for production.

        Side effects:
            Initializes memory, safety, and adaptation state.
        """
        self.config = config or TherapyEngineConfig()
        self.situation_layer = SituationUnderstandingLayer()
        self.decision_engine = TherapyDecisionEngine(
            stress_trigger_threshold=self.config.stress_trigger_threshold,
            high_stress_threshold=self.config.high_stress_threshold,
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
        self._last_context: Optional[SituationContext] = None
        self._last_decision: Optional[TherapyDecision] = None
        self._last_action: Optional[TherapeuticAction] = None
        self.suppression_counts: Dict[str, int] = {}
        self.drop_counts: Dict[str, int] = {}

    def update(self, perception: Dict[str, Any], current_time: Optional[float] = None) -> List[TherapeuticAction]:
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
        t = current_time if current_time is not None else time.time()
        context = self.situation_layer.compute(perception)
        context_dict = context.to_dict()
        self.memory.update_stress(context.environment_stress_level)
        self._last_context = context

        decision = self.decision_engine.decide(
            context_dict,
            adaptation_engine=self.adaptation,
            current_time=t,
        )
        self._last_decision = decision

        if not decision.should_intervene:
            return []

        suppress, reason = self.safety.should_suppress(context_dict, t)
        if suppress:
            self.suppression_counts[reason] = self.suppression_counts.get(reason, 0) + 1
            logger.info(
                "therapy_suppressed module=therapy_engine function=update reason=%s count=%d",
                reason,
                self.suppression_counts[reason],
            )
            return []

        action = self.intervention_generator.generate(
            decision.intervention_type,
            decision.strength,
            context_dict,
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
        action = TherapeuticAction(
            intervention_type=action.intervention_type,
            channel=action.channel,
            content=sanitized,
            intensity=action.intensity,
            duration_s=action.duration_s,
            priority=action.priority,
            metadata=action.metadata,
        )
        self._last_action = action
        self.safety.record_prompt_delivered(t)
        return [action]

    def on_user_response(
        self,
        perception_after: Dict[str, Any],
        current_time: Optional[float] = None,
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
        before_dict = {
            "environment_stress_level": self._last_context.environment_stress_level,
            "cognitive_load_estimate": self._last_context.cognitive_load_estimate,
        }
        after_dict = {
            "environment_stress_level": after_context.environment_stress_level,
            "cognitive_load_estimate": after_context.cognitive_load_estimate,
        }
        result = self.response_evaluation.evaluate(
            before_dict,
            self._last_action.intervention_type,
            after_dict,
        )
        self.adaptation.update(
            self._last_action.intervention_type,
            self._last_action.content,
            result.effectiveness_score,
            after_dict,
        )

    def get_last_context(self) -> Optional[SituationContext]:
        """Return the situation context from the most recent ``update()`` call."""
        return self._last_context

    def get_memory(self) -> TherapyMemorySystem:
        """Return the therapy memory system for inspection or persistence."""
        return self.memory
