"""Platform haptic backends selected at runtime for therapy feedback."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.haptic_feedback import HapticPattern

logger = logging.getLogger(__name__)


class HapticBackend(ABC):
    """Abstract device adapter for delivering haptic patterns."""

    @abstractmethod
    def trigger(self, pattern: "HapticPattern", intensity: float) -> None:
        """Play a haptic pattern at the given intensity in [0, 1]."""

    @abstractmethod
    def stop(self) -> None:
        """Stop in-flight haptic playback when the platform supports it."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class NoopHapticBackend(HapticBackend):
    """Backend that intentionally performs no output."""

    def trigger(self, pattern: "HapticPattern", intensity: float) -> None:
        return

    def stop(self) -> None:
        return


class LogHapticBackend(HapticBackend):
    """Development backend that records haptic events without hardware I/O."""

    def trigger(self, pattern: "HapticPattern", intensity: float) -> None:
        logger.info("Haptic %s intensity=%.3f backend=log", pattern.value, intensity)

    def stop(self) -> None:
        logger.info("Haptic stop backend=log")


class DarwinHapticBackend(HapticBackend):
    """macOS backend using NSHapticFeedbackManager (PyObjC or Swift fallback)."""

    _PATTERN_MAP = {
        "micro_pulse": "generic",
        "long_pulse": "alignment",
        "success_sequence": "levelChange",
        "failure_sequence": "alignment",
    }

    def __init__(self) -> None:
        self._use_pyobjc = False
        self._performer = None
        self._swift_available = shutil.which("swift") is not None
        try:
            import AppKit  # type: ignore[import-untyped]

            self._performer = AppKit.NSHapticFeedbackManager.defaultPerformer()
            self._use_pyobjc = True
        except Exception as exc:
            logger.debug("PyObjC AppKit unavailable for haptics: %s", exc)

    def trigger(self, pattern: "HapticPattern", intensity: float) -> None:
        style = self._PATTERN_MAP.get(pattern.value, "generic")
        if self._use_pyobjc and self._performer is not None:
            self._trigger_pyobjc(style)
            return
        if self._swift_available:
            self._trigger_swift(style)
            return
        raise RuntimeError("Darwin haptic backend unavailable: install PyObjC AppKit or Swift toolchain.")

    def stop(self) -> None:
        # NSHapticFeedbackManager has no explicit stop API.
        return

    def _trigger_pyobjc(self, style: str) -> None:
        import AppKit  # type: ignore[import-untyped]

        pattern_enum = {
            "generic": AppKit.NSHapticFeedbackPatternGeneric,
            "alignment": AppKit.NSHapticFeedbackPatternAlignment,
            "levelChange": AppKit.NSHapticFeedbackPatternLevelChange,
        }.get(style, AppKit.NSHapticFeedbackPatternGeneric)
        self._performer.performFeedbackPattern_performanceTime_(  # type: ignore[attr-defined]
            pattern_enum,
            AppKit.NSHapticFeedbackPerformanceTimeDefault,
        )

    def _trigger_swift(self, style: str) -> None:
        script = f"""
import AppKit
let performer = NSHapticFeedbackManager.defaultPerformer()
let pattern: NSHapticFeedbackManager.FeedbackPattern = .{style}
performer.perform(pattern, performanceTime: .default)
"""
        subprocess.run(
            ["swift", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )


class LinuxEvdevHapticBackend(HapticBackend):
    """Linux backend using force-feedback evdev devices when present."""

    def __init__(self) -> None:
        self._device = None
        self._effect_id: int | None = None
        self._ff = None
        self._ecodes = None
        try:
            import evdev  # type: ignore[import-untyped]
            from evdev import ecodes, ff  # type: ignore[import-untyped]

            self._ff = ff
            self._ecodes = ecodes
            for path in evdev.list_devices():
                device = evdev.InputDevice(path)
                caps = device.capabilities()
                if ecodes.EV_FF in caps and ecodes.FF_RUMBLE in caps.get(ecodes.EV_FF, []):
                    self._device = device
                    break
        except Exception as exc:
            logger.debug("Linux evdev haptic unavailable: %s", exc)

    def trigger(self, pattern: "HapticPattern", intensity: float) -> None:
        if self._device is None or self._ff is None or self._ecodes is None:
            raise RuntimeError("No Linux force-feedback evdev device found.")
        duration_ms = int(80 + 220 * float(intensity))
        if pattern.value in {"long_pulse", "failure_sequence"}:
            duration_ms = int(160 + 340 * float(intensity))
        magnitude = max(0, min(0xFFFF, int(0xFFFF * float(intensity))))
        rumble = self._ff.Rumble(strong=magnitude, weak=max(0, magnitude // 2))
        effect = self._ff.Effect(
            self._ecodes.FF_RUMBLE,
            -1,
            0,
            self._ff.Trigger(0, 0),
            self._ff.Replay(duration_ms, 0),
            self._ff.EffectType(ff_rumble=rumble),
        )
        effect_id = self._device.upload_effect(effect)
        self._effect_id = effect_id
        self._device.write(self._ecodes.EV_FF, effect_id, 1)

    def stop(self) -> None:
        if self._device is None or self._ecodes is None or self._effect_id is None:
            return
        try:
            self._device.write(self._ecodes.EV_FF, self._effect_id, 0)
        except Exception as exc:
            logger.warning("Failed to stop Linux haptic device: %s", exc)


def resolve_haptic_backend(
    backend: Optional[str] = None,
    *,
    allow_log_fallback: bool = False,
) -> HapticBackend:
    """Select a haptic backend from explicit config, env, or platform defaults.

    Parameters:
        backend: One of ``auto``, ``darwin``, ``linux``, ``log``, ``none``. When
            omitted, reads ``MAXSIGHT_HAPTIC_BACKEND``.
        allow_log_fallback: When True, fall back to ``LogHapticBackend`` instead
            of raising if no hardware backend is available.

    Returns:
        A concrete ``HapticBackend`` instance.

    Raises:
        RuntimeError: When no backend can be resolved and log fallback is disabled.
    """
    selected = (backend or os.environ.get("MAXSIGHT_HAPTIC_BACKEND", "auto")).strip().lower()
    if selected == "none":
        return NoopHapticBackend()
    if selected == "log":
        return LogHapticBackend()

    candidates: list[HapticBackend] = []
    system = platform.system().lower()
    if selected in {"auto", "darwin"} and system == "darwin":
        candidates.append(DarwinHapticBackend())
    if selected in {"auto", "linux"} and system == "linux":
        candidates.append(LinuxEvdevHapticBackend())

    for candidate in candidates:
        try:
            if isinstance(candidate, LinuxEvdevHapticBackend) and candidate._device is None:
                continue
            if isinstance(candidate, DarwinHapticBackend):
                if not candidate._use_pyobjc and not candidate._swift_available:
                    continue
            return candidate
        except Exception as exc:
            logger.debug("Skipping haptic backend %s: %s", candidate.name, exc)

    if allow_log_fallback or selected == "auto":
        logger.info("Using log haptic backend fallback.")
        return LogHapticBackend()
    raise RuntimeError(f"No haptic backend available for selection '{selected}'.")
