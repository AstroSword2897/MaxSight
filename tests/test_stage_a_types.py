"""Unit tests for frozen Stage A contract types (MAXS-101a)."""

from __future__ import annotations

import inspect

import numpy as np

from ml.runtime.stage_a.types import CameraFrame, HazardResult, StageARunner


def test_hazard_result_required_fields() -> None:
    result = HazardResult(
        event_type="hazard",
        urgency=3,
        direction="center",
        distance_zone="near",
        confidence=0.9,
        uncertainty=0.1,
        latency_ms=12.0,
        model_version="lkg-1",
        model_hash="abc",
        condition_mode="glaucoma",
        timestamp_source=1.0,
        timestamp_emit=1.01,
    )
    data = result.to_dict()
    assert data["urgency"] == 3
    assert "network" not in data


def test_camera_frame_holds_image() -> None:
    frame = CameraFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id="f0",
        timestamp=0.0,
    )
    assert frame.frame_id == "f0"
    assert frame.image.shape == (8, 8, 3)


def test_stage_a_runner_infer_signature_has_no_network_params() -> None:
    sig = inspect.signature(StageARunner.infer)
    params = list(sig.parameters)
    assert params == ["self", "frame"]
    for name in params:
        assert "network" not in name.lower()
        assert "connect" not in name.lower()
        assert "http" not in name.lower()
        assert "client" not in name.lower()


def test_concrete_runner_satisfies_protocol() -> None:
    class _Stub:
        def infer(self, frame: CameraFrame) -> HazardResult:
            return HazardResult(
                event_type="none",
                urgency=0,
                direction="center",
                distance_zone="far",
                confidence=0.0,
                uncertainty=1.0,
                latency_ms=0.0,
                model_version="stub",
                model_hash="0",
                condition_mode="none",
                timestamp_source=frame.timestamp,
                timestamp_emit=frame.timestamp,
            )

    runner: StageARunner = _Stub()
    assert isinstance(runner, StageARunner)
