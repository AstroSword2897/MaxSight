"""Temporal scalar supervision wired through MultiHeadLoss + run_training builder.

The train_maxsight CLI now resolves config into ResolvedTrainingConfig and
delegates loss construction to ml.training.runner._build_loss, so this
file checks that builder honours `loss.temporal_supervision` instead of
the removed train_maxsight.create_loss_fn helper.
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.losses import MultiHeadLoss, ScalarMSELoss  # noqa: E402
from ml.training.run_config import ResolvedTrainingConfig  # noqa: E402
from ml.training.runner import _build_loss  # noqa: E402


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


def _resolve(stem: str) -> ResolvedTrainingConfig:
    cfg_path = PROJECT_ROOT / "ml" / "training" / "configs" / f"{stem}.yaml"
    return ResolvedTrainingConfig.from_sources(
        cfg_path,
        cli_overrides={"run_id": "test", "experiment": "ci"},
    )


def _heads(fn) -> set:
    """MultiHeadLoss / GradNormMultiHeadLoss expose head_losses (ModuleDict)."""
    container = getattr(fn, "head_losses", None) or getattr(fn, "loss_functions", None)
    assert container is not None, "loss object must expose head_losses or loss_functions"
    return set(container.keys())


def test_runner_build_loss_with_temporal_supervision() -> None:
    cfg = _resolve("t5_temporal")
    heads = _heads(_build_loss(cfg))
    assert "temporal_consistency" in heads
    assert "flicker" in heads


def test_runner_build_loss_without_temporal_supervision() -> None:
    cfg = _resolve("t2_hybrid_vit")
    heads = _heads(_build_loss(cfg))
    assert "temporal_consistency" not in heads
    assert "flicker" not in heads
