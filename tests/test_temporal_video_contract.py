import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_temporal_forward_accepts_5d_and_keeps_tensor_surface() -> None:
    """
    Video/sequence contract:
    - forward must accept `[B, T, 3, H, W]`
    - outputs must be returned per-frame in the flattened batch dimension (`B*T`)
    - forward must remain trace/export compatible (no scene_description strings)
    """
    from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    tier_config.use_retrieval = False  # Keep this test focused on temporal tensor contracts.
    tier_config.use_cross_task_attention = True  # So stage flags are present.

    model = create_model(use_audio=False, tier_config=tier_config)
    model.eval()

    b, t, c, h, w = 1, 8, 3, 224, 224
    dummy_video = torch.randn(b, t, c, h, w)

    with torch.no_grad():
        outputs = model(dummy_video)

    assert outputs["classifications"].shape[0] == b * t
    assert outputs["boxes"].shape[0] == b * t
    assert outputs["objectness"].shape[0] == b * t

    # Stage flags are Python booleans in the forward dict.
    assert "stage_a_completed" in outputs
    assert outputs["stage_a_completed"] is True

    # Forward keeps non-tensor text outputs disabled.
    assert "scene_description" not in outputs

