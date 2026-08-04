"""Haptic-first urgency delivery tests (MAXS-1001)."""

from __future__ import annotations

from app.ui.hazard_haptics import deliver_hazard_haptic, urgency_to_pattern
from app.ui.haptic_backends import LogHapticBackend
from app.ui.haptic_feedback import HapticFeedback, HapticPattern
from ml.runtime.stage_a.types import HazardResult


def _result(urgency: int) -> HazardResult:
    return HazardResult(
        event_type="hazard",
        urgency=urgency,
        direction="center",
        distance_zone="near",
        confidence=0.9,
        uncertainty=0.1,
        latency_ms=10.0,
        model_version="t",
        model_hash="h",
        condition_mode="none",
        timestamp_source=0.0,
        timestamp_emit=0.0,
    )


def test_urgency_three_uses_strong_pattern() -> None:
    assert urgency_to_pattern(3) is HapticPattern.FAILURE_SEQUENCE


def test_haptic_fires_with_voice_disabled() -> None:
    backend = LogHapticBackend()
    haptics = HapticFeedback(enabled=True, backend=backend)
    pattern = deliver_hazard_haptic(_result(3), haptics, voice_enabled=False)
    assert pattern is HapticPattern.FAILURE_SEQUENCE
