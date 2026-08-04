"""Condition tensor forward / attribution tests (MAXS-302b)."""

from __future__ import annotations

import torch

from ml.evaluation.condition_attribution import condition_tensor_sensitivity
from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model
from ml.runtime_constants import CONDITION_TENSOR_WIDTH, CONDITION_MODE_IDS


def test_condition_tensor_is_distinct_named_input() -> None:
    model = create_model(
        use_audio=False,
        tier_config=TierConfig.for_tier(CapabilityTier.T0_BASELINE_CNN),
    )
    assert hasattr(model, "condition_gate")
    images = torch.randn(1, 3, 224, 224)
    a = torch.zeros(1, CONDITION_TENSOR_WIDTH)
    a[:, CONDITION_MODE_IDS["glaucoma"]] = 1.0
    b = torch.zeros(1, CONDITION_TENSOR_WIDTH)
    b[:, CONDITION_MODE_IDS["amd"]] = 1.0
    with torch.no_grad():
        out_a = model(images, condition_tensor=a)
        out_b = model(images, condition_tensor=b)
    # Different condition tensors must be able to change logits (gating path live).
    assert not torch.allclose(out_a["urgency_scores"], out_b["urgency_scores"])


def test_condition_attribution_reports_grad() -> None:
    model = create_model(
        use_audio=False,
        tier_config=TierConfig.for_tier(CapabilityTier.T0_BASELINE_CNN),
    )
    images = torch.randn(1, 3, 224, 224)
    cond = torch.zeros(1, CONDITION_TENSOR_WIDTH)
    cond[:, CONDITION_MODE_IDS["glaucoma"]] = 1.0
    report = condition_tensor_sensitivity(model, images, cond)
    assert report["grad_l1"] >= 0.0
    assert len(report["grad_per_mode"]) == CONDITION_TENSOR_WIDTH
