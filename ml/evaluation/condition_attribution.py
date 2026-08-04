"""Minimal attribution helper for the explicit condition tensor input."""

from __future__ import annotations

from typing import Any

import torch


def condition_tensor_sensitivity(
    model: torch.nn.Module,
    images: torch.Tensor,
    condition_tensor: torch.Tensor,
    output_key: str = "urgency_scores",
    eps: float = 1e-3,
) -> dict[str, Any]:
    """Finite-difference sensitivity of an output key to each condition one-hot dim."""
    model.eval()
    base = condition_tensor.detach().float().clone()
    with torch.no_grad():
        out0 = model(images, condition_tensor=base)[output_key].float().reshape(-1).mean()
        grads: list[float] = []
        for i in range(base.shape[-1]):
            pert = base.clone()
            pert[0, i] = pert[0, i] + eps
            out1 = model(images, condition_tensor=pert)[output_key].float().reshape(-1).mean()
            grads.append(float(abs((out1 - out0) / eps)))
    return {
        "output_key": output_key,
        "grad_l1": float(sum(grads)),
        "grad_per_mode": grads,
    }
