"""Configuration for MaxSight Web Simulator. Centralizes settings and production overrides."""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.runtime.mode import is_production_runtime
from ml.utils.output_scheduler import OutputMode

# Single default for docs, start_simulator.sh, and SimulatorConfig.port (before MAXSIGHT_PORT / bind resolution).
DEFAULT_SIMULATOR_PORT = 8002


def _default_demo_assumptions() -> dict[str, Any]:
    # Documented constraints for risk review; production mode flips development_mode via _apply_runtime_overrides.
    return {
        "single_camera": True,
        "single_user_per_session": True,
        "stable_lighting": False,
        "no_adversarial_input": True,
        "local_network_only": True,
        "development_mode": True,
    }


@dataclass
class SimulatorConfig:
    """Centralized configuration for the simulator and local product demo server."""

    host: str = "0.0.0.0"
    port: int = 8005
    debug: bool = True

    multi_user_enabled: bool = True
    session_timeout_seconds: int = 30 * 60

    confidence_threshold: float = 0.3
    max_ocr_texts_in_description: int = 3
    therapy_difficulty: float = 0.5
    urgency_warning_threshold: int = 2
    max_alerts_per_frame: int = 5
    alert_cooldown_frames: int = 5
    therapy_temporal_reliability_floor: float = 0.45
    therapy_temporal_history: int = 8

    haptic_intensity_high: float = 0.7
    haptic_intensity_low: float = 0.3

    baseline_save_frame: int = 1

    temporal_enabled: bool = True
    temporal_window_frames: int = 8
    temporal_stride: int = 1
    temporal_max_window_frames: int = 16

    rate_limit_per_session: int = 60
    rate_limit_global: int = 1000

    max_image_size_mb: int = 10
    allowed_image_formats: tuple[str, ...] = ("JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF")
    max_frames_data_count: int = 16
    max_frames_payload_mb: int = 40

    log_level: str = "INFO"
    enable_structured_logging: bool = True

    enable_metrics: bool = True
    metrics_port: int | None = None

    default_output_mode: OutputMode = OutputMode.PATIENT

    enable_dev_sprint_tests: bool = True

    model_checkpoint_path: str | None = None

    min_confidence_for_patient_output: float = 0.5
    min_confidence_for_critical_alert: float = 0.7

    max_spatial_memory_entries: int = 1000
    max_history_depth: int = 100
    max_memory_mb_per_session: int = 500

    voice_queue_maxsize: int = 10
    haptic_queue_maxsize: int = 10

    demo_assumptions: dict[str, Any] = field(default_factory=_default_demo_assumptions)

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be in 1..65535, got {self.port}")
        for name in (
            "confidence_threshold",
            "therapy_difficulty",
            "therapy_temporal_reliability_floor",
            "haptic_intensity_high",
            "haptic_intensity_low",
            "min_confidence_for_patient_output",
            "min_confidence_for_critical_alert",
        ):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.session_timeout_seconds <= 0:
            raise ValueError("session_timeout_seconds must be positive")
        if self.max_frames_data_count < 1:
            raise ValueError("max_frames_data_count must be >= 1")
        if self.max_image_size_mb < 1 or self.max_frames_payload_mb < 1:
            raise ValueError("payload size limits must be at least 1 MiB")


def _apply_runtime_and_flask_env(cfg: SimulatorConfig) -> None:
    # Production profile: align with hardened HTTP stack and no dev-only routes.
    if is_production_runtime():
        cfg.debug = False
        cfg.enable_dev_sprint_tests = False
        cfg.demo_assumptions["development_mode"] = False
    if os.getenv("FLASK_ENV", "").strip().lower() == "production":
        cfg.debug = False


def _apply_port_env(cfg: SimulatorConfig) -> None:
    raw = os.getenv("MAXSIGHT_PORT", "").strip()
    if not raw:
        return
    try:
        p = int(raw)
        if not (1 <= p <= 65535):
            raise ValueError("port out of range")
        cfg.port = p
    except ValueError:
        warnings.warn(f"MAXSIGHT_PORT invalid ({raw!r}); keeping default port {cfg.port}.")


config = SimulatorConfig()

if os.getenv("MAXSIGHT_CHECKPOINT_PATH"):
    checkpoint_path = Path(os.getenv("MAXSIGHT_CHECKPOINT_PATH", "")).expanduser().resolve()
    if checkpoint_path.exists():
        config.model_checkpoint_path = str(checkpoint_path)
    else:
        warnings.warn(f"Checkpoint path from environment does not exist: {checkpoint_path}")

_apply_port_env(config)
_apply_runtime_and_flask_env(config)
