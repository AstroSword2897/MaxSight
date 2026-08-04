"""TorchStageARunner.infer maps model outputs into frozen HazardResult (MAXS-102c)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.runtime.stage_a.torch_runner import TorchStageARunner
from ml.runtime.stage_a.types import CameraFrame, HazardResult

TYPES_PATH = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "types.py"
FIXTURES = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "fixtures"


def test_infer_returns_frozen_hazard_result(tmp_path: Path) -> None:
    artifact = tmp_path / "random.pt"
    artifact.write_bytes(b"not-a-real-checkpoint")
    runner = TorchStageARunner(artifact, condition_mode="glaucoma")
    frame = CameraFrame(
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        frame_id="t0",
        timestamp=1.0,
    )
    result = runner.infer(frame)
    assert isinstance(result, HazardResult)
    assert set(result.to_dict().keys()) == {
        "event_type",
        "urgency",
        "direction",
        "distance_zone",
        "confidence",
        "uncertainty",
        "latency_ms",
        "model_version",
        "model_hash",
        "condition_mode",
        "timestamp_source",
        "timestamp_emit",
        "distance_meters",
    }
    assert 0 <= result.urgency <= 3
    assert result.condition_mode == "glaucoma"


def test_types_py_unchanged_by_infer_module_contract() -> None:
    # Guardrail: PR-07 must adapt ML to types.py, not reshape it.
    text = TYPES_PATH.read_text(encoding="utf-8")
    assert "class HazardResult" in text
    assert "def infer(self, frame: CameraFrame) -> HazardResult" in text
