"""Temporal scalar supervision wired through MultiHeadLoss (matches train_maxsight --temporal-supervision)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.losses import MultiHeadLoss, ScalarMSELoss  # noqa: E402


def test_scalar_mse_loss_exact_value_and_gradient() -> None:
    loss_fn = ScalarMSELoss()
    pred = torch.tensor([[1.0], [0.0]], requires_grad=True)
    tgt = torch.tensor([1.0, 2.0], dtype=torch.float32)
    L = loss_fn(pred, tgt)
    assert L.ndim == 0
    expected = ((1.0 - 1.0) ** 2 + (0.0 - 2.0) ** 2) / 2
    assert L.item() == pytest.approx(expected)
    L.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_scalar_mse_batched_mean_over_last_dim() -> None:
    loss_fn = ScalarMSELoss()
    pred = torch.tensor([[0.5, 0.5]], requires_grad=True)
    tgt = torch.tensor([1.0])
    L = loss_fn(pred, tgt)
    assert L.item() == pytest.approx(0.25)


def test_multihead_temporal_skips_missing_pair() -> None:
    m = MultiHeadLoss(
        {
            "temporal_consistency": ScalarMSELoss(),
            "flicker": ScalarMSELoss(),
        },
        loss_weights={"temporal_consistency": 1.0, "flicker": 1.0},
    )
    preds = {"temporal_consistency": torch.tensor([[0.5]], dtype=torch.float32)}
    targs = {"temporal_consistency": torch.tensor([0.5])}
    out = m(preds, targs)
    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert "flicker" not in out or out.get("flicker") is None


def test_multihead_temporal_numeric_total() -> None:
    m = MultiHeadLoss(
        {
            "temporal_consistency": ScalarMSELoss(),
            "flicker": ScalarMSELoss(),
        },
        loss_weights={"temporal_consistency": 1.0, "flicker": 1.0},
    )
    preds = {
        "temporal_consistency": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "flicker": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }
    targs = {
        "temporal_consistency": torch.tensor([0.0, 1.0]),
        "flicker": torch.tensor([0.0, 1.0]),
    }
    out = m(preds, targs)
    assert out["temporal_consistency"].item() == pytest.approx(0.0, abs=1e-5)
    assert out["flicker"].item() == pytest.approx(0.0, abs=1e-5)
    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_create_loss_fn_train_maxsight_module() -> None:
    p = PROJECT_ROOT / "scripts" / "ops" / "train_maxsight.py"
    spec = importlib.util.spec_from_file_location("train_maxsight_sprint", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.create_loss_fn(10, False, temporal_supervision=True)
    assert hasattr(fn, "loss_functions")
    assert "temporal_consistency" in fn.loss_functions
    assert "flicker" in fn.loss_functions
    fn_off = mod.create_loss_fn(10, False, temporal_supervision=False)
    assert "temporal_consistency" not in fn_off.loss_functions
    assert "flicker" not in fn_off.loss_functions
