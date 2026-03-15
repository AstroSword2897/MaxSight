"""
Therapy Engine: closed-loop behavioral feedback system layered on top of perception.

Pipeline:
  Perception → SituationUnderstanding → DecisionEngine → InterventionGenerator
  → Output Scheduler (audio/haptics) → User Response → ResponseEvaluation
  → AdaptationEngine → TherapyMemory
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ml.therapy.situation_understanding import SituationUnderstandingLayer, SituationContext
from ml.therapy.therapy_decision_engine import TherapyDecisionEngine, TherapyDecision
from ml.therapy.intervention_generator import InterventionGenerator, TherapeuticAction
from ml.therapy.therapy_safety import TherapySafetyLayer
from ml.therapy.therapy_memory import TherapyMemorySystem
from ml.therapy.response_evaluation import ResponseEvaluationModel
from ml.therapy.adaptation_engine import AdaptationEngine


@dataclass
class TherapyEngineConfig:
    """Configuration for the therapy engine."""
    stress_trigger_threshold: float = 0.6
    high_stress_threshold: float = 0.75
    preferred_channel: str = "audio"


class TherapyEngine:
    """
    Single entry point for the therapy subsystem. Consumes perception outputs,
    runs the closed loop, and produces therapeutic actions for the output manager.
    """

    def __init__(self, config: Optional[TherapyEngineConfig] = None):
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
        self.response_evaluation = ResponseEvaluationModel()
        self._last_context: Optional[SituationContext] = None
        self._last_decision: Optional[TherapyDecision] = None
        self._last_action: Optional[TherapeuticAction] = None

    def update(self, perception: Dict[str, Any], current_time: Optional[float] = None) -> List[TherapeuticAction]:
        """
        One step of the closed loop: perception in → situation context → decision
        → safety check → intervention generation. Returns list of actions to deliver
        (caller sends them to output scheduler). Does not deliver prompts itself.
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
            return []

        action = self.intervention_generator.generate(
            decision.intervention_type,
            decision.strength,
            context_dict,
            channel_override=self.adaptation.get_preferred_channel(),
        )
        if action is None:
            return []
        sanitized = self.safety.sanitize_content(action.content)
        if not sanitized:
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
        """
        Call after user has had time to respond (e.g. next frame or after delay).
        Evaluates effectiveness and updates adaptation/memory.
        """
        if self._last_context is None or self._last_action is None:
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
        return self._last_context

    def get_memory(self) -> TherapyMemorySystem:
        return self.memory
