"""Stage A hard contract layer: on-device hazard inference with zero network dependency."""

from __future__ import annotations

from typing import Any

from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE
from ml.runtime.stage_a.model_handle import ModelHandle, ResolutionState, resolve_model_handle
from ml.runtime.stage_a.types import CameraFrame, HazardResult, StageARunner

# TorchStageARunner is lazy so pure-type imports work under the contracts profile.
__all__ = [
    "CameraFrame",
    "HAZARD_UNAVAILABLE_MESSAGE",
    "HazardResult",
    "ModelHandle",
    "ResolutionState",
    "StageARunner",
    "TorchStageARunner",
    "resolve_model_handle",
]


def __getattr__(name: str) -> Any:
    if name == "TorchStageARunner":
        from ml.runtime.stage_a.torch_runner import TorchStageARunner

        return TorchStageARunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
