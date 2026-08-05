"""MiDaS / DPT depth model loader via torch.hub.

There is no pip package providing ``midas.model_loader``; this module is the
in-repo replacement used by DepthExtractor. First call may download weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Map historical Intel-isl hub names used by callers.
_HUB_MODEL_NAMES = {
    "DPT_Large": "DPT_Large",
    "DPT_Hybrid": "DPT_Hybrid",
    "MiDaS_small": "MiDaS_small",
    "MiDaS": "MiDaS",
}


class _HubDepthModel(nn.Module):
    """Wrap a torch.hub MiDaS model so forward returns [B, 1, H, W] depth."""

    def __init__(self, hub_model: nn.Module):
        super().__init__()
        self.hub_model = hub_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Hub models expect [B, 3, H, W] in roughly ImageNet-normalized space.
        depth = self.hub_model(images)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth.mean(dim=1, keepdim=True)
        # Match caller spatial size when hub resizes internally.
        if depth.shape[-2:] != images.shape[-2:]:
            depth = F.interpolate(depth, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return depth


def load_model(name: str = "DPT_Large") -> nn.Module:
    """Load a MiDaS/DPT depth network from the intel-isl hub.

    Args:
        name: Model key (DPT_Large, DPT_Hybrid, MiDaS_small, MiDaS).

    Returns:
        nn.Module whose forward maps [B, 3, H, W] -> [B, 1, H, W].

    Raises:
        ValueError: Unknown model name.
        RuntimeError: Hub load failed (offline / network / missing deps).
    """
    hub_name = _HUB_MODEL_NAMES.get(name, name)
    if name not in _HUB_MODEL_NAMES and hub_name not in _HUB_MODEL_NAMES.values():
        raise ValueError(
            f"Unknown MiDaS model {name!r}. Supported: {sorted(_HUB_MODEL_NAMES)}"
        )
    try:
        hub_model = torch.hub.load(
            "intel-isl/MiDaS",
            hub_name,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MiDaS model {hub_name!r} from torch.hub "
            f"(intel-isl/MiDaS). First run needs network access to download weights. "
            f"Underlying error: {exc}"
        ) from exc
    if not isinstance(hub_model, nn.Module):
        raise RuntimeError(f"torch.hub returned non-Module for {hub_name!r}: {type(hub_model)!r}")
    hub_model.eval()
    return _HubDepthModel(hub_model)
