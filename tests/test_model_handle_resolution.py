"""ModelHandle resolution tests with injectable smoke probe (MAXS-102b)."""

from __future__ import annotations

from pathlib import Path

from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE
from ml.runtime.stage_a.model_handle import ResolutionState, resolve_model_handle
from ml.runtime.stage_a.types import CameraFrame, HazardResult

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


def _boom_probe(frame: CameraFrame, artifact_path: Path) -> HazardResult:
    raise RuntimeError("smoke failed")


def test_active_selected_when_healthy(tmp_path: Path) -> None:
    active = tmp_path / "active.pt"
    active.write_bytes(b"active-artifact")
    handle = resolve_model_handle(active_path=active, lkg_path=LKG, smoke_probe=_ok_probe)
    assert handle.state is ResolutionState.ACTIVE
    assert handle.artifact_path == active


def test_falls_through_to_lkg_on_active_failure(tmp_path: Path) -> None:
    active = tmp_path / "active.pt"
    active.write_bytes(b"bad-active")

    def probe(frame: CameraFrame, artifact_path: Path) -> HazardResult:
        if artifact_path == active:
            raise RuntimeError("active smoke fail")
        return _ok_probe(frame, artifact_path)

    handle = resolve_model_handle(active_path=active, lkg_path=LKG, smoke_probe=probe)
    assert handle.state is ResolutionState.LAST_KNOWN_GOOD
    assert handle.artifact_path == LKG


def test_refused_when_both_fail(tmp_path: Path) -> None:
    active = tmp_path / "active.pt"
    active.write_bytes(b"x")
    handle = resolve_model_handle(active_path=active, lkg_path=LKG, smoke_probe=_boom_probe)
    assert handle.state is ResolutionState.REFUSED
    assert handle.refusal_message == HAZARD_UNAVAILABLE_MESSAGE


def test_module_does_not_import_maxsight_cnn() -> None:
    import ml.runtime.stage_a.model_handle as mh

    assert "ml.models" not in mh.__dict__.get("__file__", "")
    src = Path(mh.__file__).read_text(encoding="utf-8")
    assert "maxsight_cnn" not in src
    assert "ml.models" not in src
