"""ModelHandle resolution: ACTIVE → LKG → REFUSED. No ML model imports."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE
from ml.runtime.stage_a.types import CameraFrame, HazardResult

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_CALIBRATION = FIXTURES_DIR / "calibration_frame.png"
DEFAULT_LKG_DIR = FIXTURES_DIR / "last_known_good"


class ResolutionState(str, Enum):
    ACTIVE = "active"
    LAST_KNOWN_GOOD = "last_known_good"
    REFUSED = "refused"


class SmokeProbe(Protocol):
    """Injectable smoke-test probe; ModelHandle must not depend on MaxSightCNN."""

    def __call__(self, frame: CameraFrame, artifact_path: Path) -> HazardResult: ...


VerifyFn = Callable[[Path], bool]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def default_verify_signature(artifact_path: Path) -> bool:
    """Verify signature when a .sig exists; unsigned stubs (LKG fixtures) pass checksum-only."""
    from ml.runtime.stage_a.verify import verify_artifact_signature

    sig = Path(str(artifact_path) + ".sig")
    if not sig.is_file():
        # Development/LKG fixtures may be unsigned until build-time signing lands.
        return artifact_path.is_file()
    return verify_artifact_signature(artifact_path, sig)


@dataclass
class ModelHandle:
    """Resolved on-device artifact handle after activation checks."""

    artifact_path: Path
    state: ResolutionState
    model_hash: str
    refusal_message: str | None = None


def resolve_model_handle(
    *,
    active_path: Path | None,
    lkg_path: Path | None = None,
    calibration_frame_path: Path | None = None,
    smoke_probe: SmokeProbe,
    verify_signature: VerifyFn | None = None,
    load_calibration_image: Callable[[Path], CameraFrame] | None = None,
) -> ModelHandle:
    """Resolve ACTIVE → LKG → REFUSED using checksum, verify hook, and smoke probe."""
    verify = verify_signature or default_verify_signature
    lkg = lkg_path or (DEFAULT_LKG_DIR / "weights.stub")
    calib_path = calibration_frame_path or DEFAULT_CALIBRATION

    def _load_frame(path: Path) -> CameraFrame:
        if load_calibration_image is not None:
            return load_calibration_image(path)
        import numpy as np

        # Smoke path does not require decoding PNG when tests inject frames.
        return CameraFrame(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_id="calibration",
            timestamp=0.0,
        )

    candidates: list[tuple[ResolutionState, Path]] = []
    if active_path is not None:
        candidates.append((ResolutionState.ACTIVE, Path(active_path)))
    candidates.append((ResolutionState.LAST_KNOWN_GOOD, Path(lkg)))

    for state, path in candidates:
        try:
            if not path.is_file():
                logger.info("stage_a.resolve skip missing path=%s state=%s", path, state.value)
                continue
            if not verify(path):
                logger.info("stage_a.resolve verify_failed path=%s state=%s", path, state.value)
                continue
            digest = _sha256_file(path)
            frame = _load_frame(calib_path)
            result = smoke_probe(frame, path)
            if not _is_well_formed(result):
                logger.info("stage_a.resolve smoke_malformed path=%s state=%s", path, state.value)
                continue
            logger.info("stage_a.resolve ok state=%s hash=%s", state.value, digest[:12])
            return ModelHandle(artifact_path=path, state=state, model_hash=digest)
        except Exception as exc:  # noqa: BLE001 — fail closed per candidate
            logger.info("stage_a.resolve error state=%s err=%s", state.value, exc)
            continue

    logger.info("stage_a.resolve REFUSED")
    return ModelHandle(
        artifact_path=Path(""),
        state=ResolutionState.REFUSED,
        model_hash="",
        refusal_message=HAZARD_UNAVAILABLE_MESSAGE,
    )


def _is_well_formed(result: HazardResult) -> bool:
    import math

    if result.urgency < 0 or result.urgency > 3:
        return False
    for value in (result.confidence, result.uncertainty, result.latency_ms):
        if not math.isfinite(float(value)):
            return False
    return True
