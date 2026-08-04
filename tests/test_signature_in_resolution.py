"""Signature verification inserted into resolution (MAXS-203b)."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE
from ml.runtime.stage_a.model_handle import ResolutionState, resolve_model_handle
from ml.runtime.stage_a.types import CameraFrame, HazardResult
from ml.runtime.stage_a.verify import verify_artifact_signature

KEYS = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "keys"
FIXTURES = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "fixtures"
LKG = FIXTURES / "last_known_good" / "weights.stub"


def _ok_probe(frame: CameraFrame, artifact_path: Path) -> HazardResult:
    return HazardResult(
        event_type="none",
        urgency=0,
        direction="center",
        distance_zone="far",
        confidence=0.5,
        uncertainty=0.5,
        latency_ms=1.0,
        model_version="probe",
        model_hash=artifact_path.name,
        condition_mode="none",
        timestamp_source=frame.timestamp,
        timestamp_emit=frame.timestamp,
    )


def _sign(path: Path, key: bytes) -> None:
    digest = hashlib.sha256(path.read_bytes()).digest()
    (Path(str(path) + ".sig")).write_bytes(hmac.new(key, digest, hashlib.sha256).digest())


def test_tampered_active_falls_to_lkg(tmp_path: Path) -> None:
    key = (KEYS / "current.hmac").read_bytes().strip()
    active = tmp_path / "active.bin"
    active.write_bytes(b"good-active")
    _sign(active, key)
    # Flip one byte after signing.
    active.write_bytes(b"hood-active")
    assert verify_artifact_signature(active, keys_dir=KEYS) is False

    handle = resolve_model_handle(
        active_path=active,
        lkg_path=LKG,
        smoke_probe=_ok_probe,
        verify_signature=lambda p: (
            verify_artifact_signature(p, keys_dir=KEYS)
            if Path(str(p) + ".sig").is_file()
            else p.is_file()
        ),
    )
    assert handle.state is ResolutionState.LAST_KNOWN_GOOD


def test_tampered_lkg_refuses(tmp_path: Path) -> None:
    key = (KEYS / "current.hmac").read_bytes().strip()
    lkg = tmp_path / "lkg.bin"
    lkg.write_bytes(b"lkg-bytes")
    _sign(lkg, key)
    lkg.write_bytes(b"lkg-BYTEZ")
    handle = resolve_model_handle(
        active_path=None,
        lkg_path=lkg,
        smoke_probe=_ok_probe,
        verify_signature=lambda p: verify_artifact_signature(p, keys_dir=KEYS),
    )
    assert handle.state is ResolutionState.REFUSED
    assert handle.refusal_message == HAZARD_UNAVAILABLE_MESSAGE
