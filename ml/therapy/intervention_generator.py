"""Intervention Generator: therapy decisions → concrete therapeutic actions (audio/haptic/visual)."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional


class InterventionType(Enum):
    """Types of therapeutic intervention. Maps to grounding, reassurance, breathing, etc."""
    GROUNDING_PROMPT = "grounding"
    NAVIGATION_REASSURANCE = "navigation_reassurance"
    BREATHING_GUIDANCE = "breathing"
    COGNITIVE_REFRAMING = "cognitive_reframing"
    ATTENTION_REDIRECTION = "attention_redirection"
    CALMING_PROMPT = "calming"
    REST_SUGGESTION = "rest_suggestion"


@dataclass
class TherapeuticAction:
    """Single intervention to be delivered via output scheduler."""
    intervention_type: str
    channel: str  # audio, haptic, visual
    content: str
    intensity: float  # 0-1
    duration_s: float
    priority: int  # 0-100
    metadata: Dict[str, Any]


class InterventionGenerator:
    """
    Converts decision engine output (should_intervene, intervention_type, strength)
    into specific prompts and patterns for the output manager. No ML here; content is rule-based.
    """

    def __init__(self, preferred_channel: str = "audio"):
        self.preferred_channel = preferred_channel

    def generate(
        self,
        intervention_type: str,
        strength: float,
        context: Dict[str, Any],
        channel_override: Optional[str] = None,
    ) -> Optional[TherapeuticAction]:
        """
        Produce one therapeutic action. Returns None if type is unknown or content would be empty.
        """
        channel = channel_override or self.preferred_channel
        strength = max(0.0, min(1.0, strength))

        if intervention_type == InterventionType.GROUNDING_PROMPT.value or intervention_type == "grounding":
            content = "Name three things you can hear or feel right now."
            return TherapeuticAction(
                intervention_type=InterventionType.GROUNDING_PROMPT.value,
                channel=channel,
                content=content,
                intensity=0.5 + strength * 0.3,
                duration_s=8.0,
                priority=60,
                metadata={"category": "grounding"},
            )
        if intervention_type == InterventionType.NAVIGATION_REASSURANCE.value or intervention_type == "navigation_reassurance":
            content = "Stay to the right side of the sidewalk. You have time."
            return TherapeuticAction(
                intervention_type=InterventionType.NAVIGATION_REASSURANCE.value,
                channel=channel,
                content=content,
                intensity=0.4 + strength * 0.4,
                duration_s=5.0,
                priority=70,
                metadata={"category": "reassurance"},
            )
        if intervention_type == InterventionType.BREATHING_GUIDANCE.value or intervention_type == "breathing":
            content = "Take a slow breath in, then out."
            return TherapeuticAction(
                intervention_type=InterventionType.BREATHING_GUIDANCE.value,
                channel=channel,
                content=content,
                intensity=0.3 + strength * 0.3,
                duration_s=6.0,
                priority=65,
                metadata={"category": "breathing"},
            )
        if intervention_type == InterventionType.CALMING_PROMPT.value or intervention_type == "calming":
            content = "Pause for a moment. You are doing fine."
            return TherapeuticAction(
                intervention_type=InterventionType.CALMING_PROMPT.value,
                channel=channel,
                content=content,
                intensity=0.4 + strength * 0.3,
                duration_s=4.0,
                priority=68,
                metadata={"category": "calming"},
            )
        if intervention_type == InterventionType.ATTENTION_REDIRECTION.value or intervention_type == "attention_redirection":
            content = "Focus on the path ahead. One step at a time."
            return TherapeuticAction(
                intervention_type=InterventionType.ATTENTION_REDIRECTION.value,
                channel=channel,
                content=content,
                intensity=0.5 + strength * 0.3,
                duration_s=5.0,
                priority=62,
                metadata={"category": "attention"},
            )
        if intervention_type == InterventionType.REST_SUGGESTION.value or intervention_type == "rest_suggestion":
            content = "Consider a short pause when you can. No rush."
            return TherapeuticAction(
                intervention_type=InterventionType.REST_SUGGESTION.value,
                channel=channel,
                content=content,
                intensity=0.3,
                duration_s=6.0,
                priority=55,
                metadata={"category": "rest"},
            )
        return None
