"""Safety layer for the therapy subsystem: guardrails so therapy is non-intrusive and never harmful."""

from __future__ import annotations

import threading

from ml.runtime_constants import (
    THERAPY_MAX_PROMPTS_PER_MINUTE,
    THERAPY_MIN_GAP_BETWEEN_PROMPTS_S,
    THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD,
)
from ml.therapy.situation_understanding import SituationContext


class TherapySafetyLayer:
    """Enforce therapy rate limits, uncertainty gating, and content sanitization."""

    DEFAULT_DISALLOWED_PHRASES: frozenset[str] = frozenset(
        {
            "diagnos",
            "prescribe",
            "cure",
            "treatment",
            "medical advice",
            "you have",
            "you are sick",
            "take medication",
            "see a doctor for",
        }
    )

    def __init__(
        self,
        uncertainty_suppress_threshold: float = THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD,
        max_prompts_per_minute: float = THERAPY_MAX_PROMPTS_PER_MINUTE,
        min_gap_s: float = THERAPY_MIN_GAP_BETWEEN_PROMPTS_S,
        disallowed_phrases: frozenset[str] | None = None,
    ):
        """Configure safety thresholds and initialize prompt timing state.

        Parameters:
            uncertainty_suppress_threshold: Suppress therapy above this uncertainty.
            max_prompts_per_minute: Rolling one-minute prompt cap.
            min_gap_s: Minimum seconds between consecutive prompts.
            disallowed_phrases: Override default blocked medical/diagnostic phrases.
        """
        self.uncertainty_suppress_threshold = uncertainty_suppress_threshold
        self.max_prompts_per_minute = max_prompts_per_minute
        self.min_gap_s = min_gap_s
        self.disallowed_phrases = disallowed_phrases or self.DEFAULT_DISALLOWED_PHRASES
        self._last_prompt_time: float = -1e9
        self._prompts_this_minute: list[float] = []
        self._lock = threading.Lock()

    def should_suppress(
        self,
        context: SituationContext,
        current_time: float,
    ) -> tuple[bool, str]:
        """Decide whether to suppress therapy delivery for the current context.

        Parameters:
            context: Typed situation context from ``SituationUnderstandingLayer``.
            current_time: Monotonic or wall-clock seconds for rate limiting.

        Returns:
            Tuple of ``(suppress, reason)``. ``reason`` is empty when not suppressed.
        """
        with self._lock:
            uncertainty = float(context.uncertainty)
            if uncertainty > self.uncertainty_suppress_threshold:
                return True, "uncertainty_above_threshold"

            if (
                self._last_prompt_time >= 0
                and (current_time - self._last_prompt_time) < self.min_gap_s
            ):
                return True, "min_gap_not_elapsed"

            self._prune_old_prompts(current_time)
            if len(self._prompts_this_minute) >= self.max_prompts_per_minute:
                return True, "max_prompts_per_minute"

        return False, ""

    def record_prompt_delivered(self, current_time: float) -> None:
        """Record a delivered prompt for rate-limit accounting."""
        with self._lock:
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
        lower = content.lower()
        for phrase in self.disallowed_phrases:
            if phrase in lower:
                return ""
        return content
