"""Frozen Stage A contract types. Network and connectivity parameters are intentionally absent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """Single camera frame for on-device Stage A inference."""

    image: np.ndarray
    frame_id: str
    timestamp: float


@dataclass(frozen=True)
class HazardResult:
    """Safety-critical Stage A output aligned with CriticalEvent semantics."""

    event_type: str
    urgency: int
    direction: str
    distance_zone: str
    confidence: float
    uncertainty: float
    latency_ms: float
    model_version: str
    model_hash: str
    condition_mode: str
    timestamp_source: float
    timestamp_emit: float
    distance_meters: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class StageARunner(Protocol):
    """Hard contract: infer accepts only a frame. No network client or connectivity flag."""

    def infer(self, frame: CameraFrame) -> HazardResult:
        """Run Stage A hazard inference on a local frame."""
        ...
