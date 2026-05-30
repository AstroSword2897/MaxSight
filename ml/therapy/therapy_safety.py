"""Safety layer for the therapy subsystem: guardrails so therapy is non-intrusive and never harmful."""

from typing import Dict, Any
from ml.runtime_constants import (
    THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD,
    THERAPY_MAX_PROMPTS_PER_MINUTE,
    THERAPY_MIN_GAP_BETWEEN_PROMPTS_S,
)


class TherapySafetyLayer:
    """Enforce therapy rate limits, uncertainty gating, and content sanitization."""

    def __init__(
        self,
        uncertainty_suppress_threshold: float = THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD,
        max_prompts_per_minute: float = THERAPY_MAX_PROMPTS_PER_MINUTE,
        min_gap_s: float = THERAPY_MIN_GAP_BETWEEN_PROMPTS_S,
    ):
        """Configure safety thresholds and initialize prompt timing state.

        Parameters:
            uncertainty_suppress_threshold: Suppress therapy above this uncertainty.
            max_prompts_per_minute: Rolling one-minute prompt cap.
            min_gap_s: Minimum seconds between consecutive prompts.
        """
        self.uncertainty_suppress_threshold = uncertainty_suppress_threshold
        self.max_prompts_per_minute = max_prompts_per_minute
        self.min_gap_s = min_gap_s
        self._last_prompt_time: float = -1e9
        self._prompts_this_minute: list = []

    def should_suppress(
        self,
        context: Dict[str, Any],
        current_time: float,
    ) -> tuple[bool, str]:
        """Decide whether to suppress therapy delivery for the current context.

        Parameters:
            context: Situation dict; reads ``uncertainty`` when present.
            current_time: Monotonic or wall-clock seconds for rate limiting.

        Returns:
            Tuple of ``(suppress, reason)``. ``reason`` is empty when not suppressed.
        """
        uncertainty = context.get("uncertainty", 0.0)
        if uncertainty > self.uncertainty_suppress_threshold:
            return True, "uncertainty_above_threshold"

        if self._last_prompt_time >= 0 and (current_time - self._last_prompt_time) < self.min_gap_s:
            return True, "min_gap_not_elapsed"

        self._prune_old_prompts(current_time)
        if len(self._prompts_this_minute) >= self.max_prompts_per_minute:
            return True, "max_prompts_per_minute"

        return False, ""

    def record_prompt_delivered(self, current_time: float) -> None:
        """Record a delivered prompt for rate-limit accounting."""
        self._last_prompt_time = current_time
        self._prompts_this_minute.append(current_time)
        self._prune_old_prompts(current_time)

    def _prune_old_prompts(self, current_time: float) -> None:
        cutoff = current_time - 60.0
        self._prompts_this_minute = [t for t in self._prompts_this_minute if t > cutoff]

    def sanitize_content(self, content: str) -> str:
        """Strip disallowed medical or diagnostic phrasing from prompt text.

        Returns:
            Sanitized content, or empty string when disallowed phrases are found.
        """
        content = content.strip()
        disallowed = (
            "diagnos", "prescribe", "cure", "treatment", "medical advice",
            "you have", "you are sick", "take medication", "see a doctor for",
        )
        lower = content.lower()
        for phrase in disallowed:
            if phrase in lower:
                return ""
        return content
