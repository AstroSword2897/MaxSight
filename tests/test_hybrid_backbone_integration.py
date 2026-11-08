"""The hybrid CNN-ViT branch of MaxSightCNN either runs or fails loudly.

It used to raise a channel-mismatch error on every forward pass; the model swallowed it, appended a
message to a list that grew without bound, and silently ran without the branch. A channel adapter was
also created lazily inside forward(), after the optimizer had been built, so it never trained.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model  # noqa: E402


@pytest.fixture(scope="module")
def hybrid_model():
    torch.manual_seed(0)
    return create_model(tier_config=TierConfig.for_tier(CapabilityTier.T2_HYBRID_VIT), use_audio=False).eval()


def _forward(model):
    with torch.no_grad():
        return model(torch.randn(1, 3, 224, 224))


def test_hybrid_branch_runs_and_changes_stage_b_features(hybrid_model, monkeypatch):
    stage_a = torch.randn(1, hybrid_model.hybrid_backbone.cnn_out_channels, 28, 28)
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        with_hybrid, _ = hybrid_model._forward_stage_b_backbone(images, stage_a)
        monkeypatch.setattr(hybrid_model, "use_hybrid", False)
        without_hybrid, _ = hybrid_model._forward_stage_b_backbone(images, stage_a)
    assert not torch.equal(with_hybrid, without_hybrid)


def test_a_hybrid_failure_propagates_instead_of_being_swallowed(hybrid_model, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("hybrid exploded")

    monkeypatch.setattr(hybrid_model.hybrid_backbone, "forward", boom)
    with pytest.raises(RuntimeError, match="hybrid exploded"):
        _forward(hybrid_model)


def test_forward_creates_no_parameters_and_keeps_no_growing_state(hybrid_model):
    params_before = {name for name, _ in hybrid_model.named_parameters()}
    attrs_before = set(vars(hybrid_model))
    for _ in range(3):
        _forward(hybrid_model)
    assert {name for name, _ in hybrid_model.named_parameters()} == params_before
    assert not hasattr(hybrid_model, "_hybrid_backbone_warnings")
    assert set(vars(hybrid_model)) - attrs_before <= {"_grid_norm_cache"}
