"""Ticket 402: a checkpoint saved and loaded again has identical weights, and the loader raises on a
missing, corrupt or malformed file instead of continuing silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.training.checkpoint_io import CheckpointError, read_checkpoint, write_atomic_torch  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.head = torch.nn.Linear(4, 2)


def make_checkpoint() -> dict:
    torch.manual_seed(0)
    model = TinyModel()
    return {"epoch": 3, "model_state_dict": model.state_dict()}


def test_saved_weights_load_back_identically(tmp_path: Path):
    # Arrange
    saved = make_checkpoint()
    path = tmp_path / "best_model.pt"

    # Act
    write_atomic_torch(path, saved)
    loaded = read_checkpoint(path)

    # Assert: same keys, same shapes, bit-identical values.
    assert loaded["epoch"] == 3
    assert loaded["model_state_dict"].keys() == saved["model_state_dict"].keys()
    for name, tensor in saved["model_state_dict"].items():
        assert torch.equal(loaded["model_state_dict"][name], tensor), name


def test_loaded_weights_restore_a_fresh_model(tmp_path: Path):
    saved = make_checkpoint()
    path = tmp_path / "best_model.pt"
    write_atomic_torch(path, saved)

    fresh = TinyModel()
    result = fresh.load_state_dict(read_checkpoint(path)["model_state_dict"], strict=True)

    assert not result.missing_keys and not result.unexpected_keys
    for name, tensor in fresh.state_dict().items():
        assert torch.equal(tensor, saved["model_state_dict"][name]), name


def test_a_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_checkpoint(tmp_path / "does_not_exist.pt")


@pytest.mark.parametrize("content", [b"", b"not a torch checkpoint at all"])
def test_a_corrupt_file_raises(tmp_path: Path, content: bytes):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(content)
    with pytest.raises(CheckpointError, match="corrupt"):
        read_checkpoint(path)


def test_a_truncated_file_raises(tmp_path: Path):
    path = tmp_path / "truncated.pt"
    write_atomic_torch(path, make_checkpoint())
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(CheckpointError, match="corrupt"):
        read_checkpoint(path)


def test_a_file_without_model_weights_raises(tmp_path: Path):
    path = tmp_path / "no_weights.pt"
    write_atomic_torch(path, {"epoch": 1})
    with pytest.raises(CheckpointError, match="model_state_dict"):
        read_checkpoint(path)


def test_writes_never_leave_temporary_files(tmp_path: Path):
    write_atomic_torch(tmp_path / "best_model.pt", make_checkpoint())
    assert not list(tmp_path.glob("*.tmp"))


def test_resuming_training_from_a_missing_checkpoint_raises(tmp_path: Path):
    """Before ticket 402 a missing resume checkpoint logged a warning and training started from scratch."""
    import logging
    from types import SimpleNamespace

    from ml.training.train_loop import ProductionTrainLoop

    trainer = SimpleNamespace(device="cpu", logger=logging.getLogger("test"), checkpoint_resume_status="not_attempted")

    with pytest.raises(FileNotFoundError):
        ProductionTrainLoop._load_checkpoint(trainer, str(tmp_path / "missing.pt"))
    assert trainer.checkpoint_resume_status == "missing"


def test_resuming_training_from_a_corrupt_checkpoint_raises(tmp_path: Path):
    import logging
    from types import SimpleNamespace

    from ml.training.train_loop import ProductionTrainLoop

    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"garbage")
    trainer = SimpleNamespace(device="cpu", logger=logging.getLogger("test"), checkpoint_resume_status="not_attempted")

    with pytest.raises(CheckpointError):
        ProductionTrainLoop._load_checkpoint(trainer, str(path))
    assert trainer.checkpoint_resume_status == "corrupt"
