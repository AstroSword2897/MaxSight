"""Tests for training hardening: checkpoints, reproducibility, distributed, observability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from ml.training.distributed import distributed_sampler, is_main_process, should_checkpoint
from ml.training.observability import StructuredEvent, emit_event
from ml.training.reproducibility import checkpoint_content_hash
from ml.training.train_loop import write_atomic_json, write_atomic_torch


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def test_checkpoint_content_hash_stable():
    model = TinyModel()
    state = model.state_dict()
    assert checkpoint_content_hash(state) == checkpoint_content_hash(state)


def test_write_atomic_torch_never_leaves_partial_file(tmp_path: Path):
    payload = {"epoch": 1, "model_state_dict": TinyModel().state_dict()}
    target = tmp_path / "checkpoint.pt"
    write_atomic_torch(target, payload)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 1
    assert not any(tmp_path.glob("*.tmp"))


def test_write_atomic_json_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.json"
    write_atomic_json(path, {"seed": 42})
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 42


def test_structured_event_validation():
    event = StructuredEvent(
        name="therapy.suppressed",
        fields={
            "module": "therapy_engine",
            "function": "update",
            "reason": "rate_limit",
            "count": 1,
        },
    )
    line = event.to_log_line()
    assert line.startswith("event=")
    payload = json.loads(line.split("=", 1)[1])
    assert payload["event"] == "therapy.suppressed"


def test_emit_event_unknown_schema_raises():
    with pytest.raises(ValueError, match="unknown structured event schema"):
        emit_event("unknown.event", foo="bar")


def test_distributed_helpers_default_single_process():
    assert is_main_process() is True
    assert should_checkpoint() is True


def test_distributed_sampler_builds():
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(4, 2))
    sampler = distributed_sampler(dataset, rank=0, world_size=1)
    assert sampler.rank == 0
    assert sampler.num_replicas == 1
