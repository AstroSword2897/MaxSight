"""Map HazardResult urgency to haptic-first patterns (audio optional)."""

from __future__ import annotations

from app.ui.haptic_feedback import HapticFeedback, HapticPattern
from ml.runtime.stage_a.types import HazardResult


def urgency_to_pattern(urgency: int) -> HapticPattern:
    if urgency >= 3:
        return HapticPattern.FAILURE_SEQUENCE
    if urgency == 2:
        return HapticPattern.LONG_PULSE
    if urgency == 1:
        return HapticPattern.MICRO_PULSE
    return HapticPattern.SUCCESS_SEQUENCE


def deliver_hazard_haptic(
    result: HazardResult,
    haptics: HapticFeedback,
    *,
    voice_enabled: bool = False,
) -> HapticPattern:
    """Deliver urgency alert via haptics even when voice/audio is disabled."""
    _ = voice_enabled  # Explicitly unused: haptic-first path does not require audio.
    pattern = urgency_to_pattern(result.urgency)
    intensity = 0.3 + 0.2 * min(3, max(0, result.urgency))
    haptics.trigger(pattern, intensity)
    return pattern
