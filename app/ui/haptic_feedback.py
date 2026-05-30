"""Haptic feedback facade with platform-specific device adapters."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Union

from app.ui.haptic_backends import HapticBackend, LogHapticBackend, resolve_haptic_backend

logger = logging.getLogger(__name__)


class HapticPattern(Enum):
    """Named haptic feedback patterns used by therapy and simulator flows."""

    MICRO_PULSE = "micro_pulse"
    LONG_PULSE = "long_pulse"
    SUCCESS_SEQUENCE = "success_sequence"
    FAILURE_SEQUENCE = "failure_sequence"


class HapticFeedback:
    """Deliver therapy haptics through a resolved platform backend."""

    def __init__(
        self,
        enabled: bool = True,
        backend: Optional[Union[str, HapticBackend]] = None,
        allow_log_fallback: bool = False,
        *,
        allow_stub: Optional[bool] = None,
    ):
        """Initialize haptic delivery.

        Parameters:
            enabled: When False, all trigger/stop calls no-op.
            backend: Explicit backend name (``auto``, ``darwin``, ``linux``,
                ``log``, ``none``) or a ``HapticBackend`` instance.
            allow_log_fallback: Use log backend when hardware is unavailable.
            allow_stub: Deprecated alias for ``allow_log_fallback``.

        Side effects:
            Resolves and stores a concrete backend adapter.

        Failure modes:
            Raises ``RuntimeError`` when no backend can be resolved and log
            fallback is disabled.
        """
        if allow_stub is not None:
            allow_log_fallback = allow_log_fallback or allow_stub
        self.enabled = enabled
        if isinstance(backend, HapticBackend):
            self._backend: HapticBackend = backend
        else:
            self._backend = resolve_haptic_backend(
                backend,
                allow_log_fallback=allow_log_fallback,
            )

    def trigger(self, pattern: HapticPattern, intensity: float = 0.5) -> None:
        """Play a haptic pattern.

        Parameters:
            pattern: Pattern enum value.
            intensity: Normalized intensity in ``[0, 1]``.

        Side effects:
            Invokes the active backend when enabled.

        Failure modes:
            Propagates backend ``RuntimeError`` when hardware delivery fails.
        """
        if not self.enabled:
            return
        intensity = max(0.0, min(1.0, float(intensity)))
        self._backend.trigger(pattern, intensity)

    def micro_pulse(self, intensity: float = 0.3) -> None:
        """Short pulse indicating a target was found."""
        self.trigger(HapticPattern.MICRO_PULSE, intensity)

    def long_pulse(self, intensity: float = 0.6) -> None:
        """Longer pulse indicating an incorrect region."""
        self.trigger(HapticPattern.LONG_PULSE, intensity)

    def success_sequence(self) -> None:
        """Two-step success feedback sequence."""
        self.trigger(HapticPattern.MICRO_PULSE, 0.5)
        self.trigger(HapticPattern.SUCCESS_SEQUENCE, 0.8)

    def failure_sequence(self) -> None:
        """Two-step failure feedback sequence."""
        self.trigger(HapticPattern.LONG_PULSE, 0.6)
        self.trigger(HapticPattern.FAILURE_SEQUENCE, 0.4)

    def stop(self) -> None:
        """Stop in-flight haptic playback when supported by the backend."""
        if not self.enabled:
            return
        self._backend.stop()

    @property
    def backend_name(self) -> str:
        """Return the active backend class name for diagnostics."""
        return self._backend.name
