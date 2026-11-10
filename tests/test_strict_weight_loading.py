"""Inference-side weight loading must fail closed.

Every path that turns a checkpoint into a model used for predictions (Stage A on-device runner,
SageMaker model_fn, export CLI) loads strictly: a missing, corrupt, or mismatched checkpoint raises
instead of leaving some or all layers at random initialisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.training.checkpoint_io import (  # noqa: E402
    CheckpointError,
    load_model_weights,
    write_atomic_torch,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.head = torch.nn.Linear(4, 2)


def _trained_state() -> dict:
    torch.manual_seed(0)
    return TinyModel().state_dict()


# ── load_model_weights ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("wrapped", [True, False], ids=["model_state_dict", "bare_state_dict"])
def test_matching_checkpoint_loads_exact_weights(tmp_path: Path, wrapped: bool):
    state = _trained_state()
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"model_state_dict": state} if wrapped else state)

    model = TinyModel()
    load_model_weights(model, path)

    for key, value in state.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_missing_layer_raises_and_names_it(tmp_path: Path):
    state = _trained_state()
    del state["head.weight"]
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"model_state_dict": state})

    with pytest.raises(CheckpointError, match="head.weight"):
        load_model_weights(TinyModel(), path)


def test_unexpected_layer_raises_and_names_it(tmp_path: Path):
    state = _trained_state()
    state["extra.weight"] = torch.zeros(1)
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"model_state_dict": state})

    with pytest.raises(CheckpointError, match="extra.weight"):
        load_model_weights(TinyModel(), path)


def test_shape_mismatch_raises(tmp_path: Path):
    state = _trained_state()
    state["head.weight"] = torch.zeros(3, 4)
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"model_state_dict": state})

    with pytest.raises(CheckpointError, match="head.weight"):
        load_model_weights(TinyModel(), path)


def test_failed_load_leaves_no_partial_weights(tmp_path: Path):
    state = _trained_state()
    del state["head.bias"]
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"model_state_dict": state})
    model = TinyModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    with pytest.raises(CheckpointError):
        load_model_weights(model, path)

    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model_weights(TinyModel(), tmp_path / "absent.pt")


def test_garbage_file_raises(tmp_path: Path):
    path = tmp_path / "w.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(CheckpointError):
        load_model_weights(TinyModel(), path)


def test_payload_without_tensors_raises(tmp_path: Path):
    path = tmp_path / "w.pt"
    write_atomic_torch(path, {"epoch": 3, "note": "no weights"})
    with pytest.raises(CheckpointError):
        load_model_weights(TinyModel(), path)


# ── Stage A on-device runner ────────────────────────────────────────────────────


def _frame():
    from ml.runtime.stage_a.types import CameraFrame

    return CameraFrame(image=np.zeros((64, 64, 3), dtype=np.uint8), frame_id="t0", timestamp=1.0)


@pytest.mark.parametrize(
    "name,content",
    [("random.pt", b"not-a-real-checkpoint"), ("weights.stub", b"stub"), ("model.onnx", b"x")],
    ids=["corrupt_pt", "stub", "unsupported_suffix"],
)
def test_stage_a_runner_refuses_unloadable_artifact(tmp_path: Path, name: str, content: bytes):
    from ml.runtime.stage_a.torch_runner import TorchStageARunner

    artifact = tmp_path / name
    artifact.write_bytes(content)
    runner = TorchStageARunner(artifact, condition_mode="glaucoma")

    with pytest.raises(CheckpointError):
        runner.infer(_frame())


def test_stage_a_runner_refuses_missing_artifact(tmp_path: Path):
    from ml.runtime.stage_a.torch_runner import TorchStageARunner

    runner = TorchStageARunner(tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError):
        runner.infer(_frame())


# ── SageMaker model_fn ──────────────────────────────────────────────────────────


@pytest.fixture
def tiny_create_model(monkeypatch):
    import ml.models.maxsight_cnn as cnn

    monkeypatch.setattr(cnn, "create_model", lambda **_: TinyModel())


def test_model_fn_refuses_missing_checkpoint(tmp_path: Path, tiny_create_model):
    from ml.infra.inference_handler import model_fn
    from ml.middleware.error_sanitizer import ModelLoadError

    with pytest.raises(ModelLoadError, match="best.pt"):
        model_fn(str(tmp_path))


def test_model_fn_refuses_mismatched_checkpoint(tmp_path: Path, tiny_create_model):
    from ml.infra.inference_handler import model_fn
    from ml.middleware.error_sanitizer import ModelLoadError

    state = _trained_state()
    del state["conv.bias"]
    write_atomic_torch(tmp_path / "best.pt", {"model_state_dict": state})

    with pytest.raises(ModelLoadError, match="conv.bias"):
        model_fn(str(tmp_path))


def test_model_fn_loads_matching_checkpoint(tmp_path: Path, tiny_create_model):
    from ml.infra.inference_handler import model_fn

    state = _trained_state()
    write_atomic_torch(tmp_path / "best.pt", {"model_state_dict": state})

    loaded = model_fn(str(tmp_path))["model"]

    assert torch.equal(loaded.state_dict()["head.weight"].cpu(), state["head.weight"])


# ── Export CLI ──────────────────────────────────────────────────────────────────


def test_export_refuses_missing_checkpoint(tmp_path: Path):
    from ml.training.export import load_export_model

    with pytest.raises(FileNotFoundError):
        load_export_model(str(tmp_path / "absent.pt"), model_factory=TinyModel)


def test_export_refuses_mismatched_checkpoint(tmp_path: Path):
    from ml.training.export import load_export_model

    state = _trained_state()
    del state["head.weight"]
    path = tmp_path / "w.pt"
    write_atomic_torch(path, state)

    with pytest.raises(CheckpointError):
        load_export_model(str(path), model_factory=TinyModel)


def test_export_without_checkpoint_is_an_explicit_untrained_model():
    from ml.training.export import load_export_model

    assert isinstance(load_export_model(None, model_factory=TinyModel), TinyModel)
