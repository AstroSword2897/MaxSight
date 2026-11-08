"""use_dynamic_conv does what it says.

The flag was True for T2-T5 but MaxSightCNN never built a DynamicConv2d, so it did nothing. It now
adds a residual DynamicConv2d to Stage B whose gate starts at zero (a fresh model behaves as before
until trained). It is off in every tier preset and training YAML, so existing checkpoints keep loading
under strict loading; turning it on means retraining with the preset and YAML changed together.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "ml" / "training" / "configs"


@pytest.mark.parametrize("tier", list(CapabilityTier), ids=lambda t: t.name)
def test_off_in_every_tier_preset(tier):
    assert TierConfig.for_tier(tier).use_dynamic_conv is False


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("t*.yaml")), ids=lambda p: p.name)
def test_off_in_every_training_yaml_so_training_and_inference_build_the_same_model(path):
    model = yaml.safe_load(path.read_text()).get("model", {})
    assert model.get("use_dynamic_conv", False) is False


def _model(flag: bool):
    torch.manual_seed(0)
    cfg = dataclasses.replace(TierConfig.for_tier(CapabilityTier.T2_DETECTOR), use_dynamic_conv=flag)
    return create_model(tier_config=cfg, use_audio=False).eval()


def test_flag_off_builds_no_dynamic_conv():
    model = _model(False)
    assert model.dynamic_conv is None
    assert not any(name.startswith("dynamic_conv") for name, _ in model.named_parameters())


def test_flag_on_registers_the_layer_at_construction():
    model = _model(True)
    names = {name for name, _ in model.named_parameters()}
    assert "dynamic_conv_gate" in names and any(n.startswith("dynamic_conv.") for n in names)


def test_zero_gate_leaves_stage_b_unchanged_and_an_open_gate_changes_it():
    model = _model(True)
    channels = model.dynamic_conv.in_channels
    stage_a = torch.randn(1, channels, 28, 28)
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        closed, _ = model._forward_stage_b_backbone(images, stage_a)
        model.dynamic_conv_gate.fill_(1.0)
        opened, _ = model._forward_stage_b_backbone(images, stage_a)
    assert torch.equal(closed, stage_a)
    assert not torch.equal(opened, stage_a)


def test_the_layer_and_its_gate_receive_gradients():
    model = _model(True).train()
    stage_a = torch.randn(2, model.dynamic_conv.in_channels, 16, 16, requires_grad=True)
    out, _ = model._forward_stage_b_backbone(torch.randn(2, 3, 224, 224), stage_a)
    out.square().mean().backward()
    assert model.dynamic_conv_gate.grad is not None and float(model.dynamic_conv_gate.grad.abs()) > 0


def test_full_forward_runs_with_the_flag_on():
    with torch.no_grad():
        outputs = _model(True)(torch.randn(1, 3, 224, 224))
    assert isinstance(outputs, dict) and outputs
