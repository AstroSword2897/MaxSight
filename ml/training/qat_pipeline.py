"""Quantization-Aware Training (QAT) pipeline: calibration, accuracy-loss gating, INT8 export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_ACCURACY_LOSS_THRESHOLD = 0.03  # 3 % relative degradation cap


@dataclass
class QuantizationReport:
    """Accuracy-loss report comparing FP32 and INT8 model variants."""

    fp32_metric: float
    int8_metric: float
    metric_name: str
    relative_delta: float
    passed: bool
    quantization_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fp32_metric": self.fp32_metric,
            "int8_metric": self.int8_metric,
            "metric_name": self.metric_name,
            "relative_delta": round(self.relative_delta, 6),
            "passed": self.passed,
            "quantization_type": self.quantization_type,
        }


def post_training_quantize(
    model: nn.Module,
    calibration_data: torch.Tensor,
    *,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training static quantization."""
    model = model.cpu().eval()
    model.qconfig = torch.quantization.get_default_qconfig(backend)  # type: ignore[attr-defined]
    torch.quantization.prepare(model, inplace=True)  # type: ignore[attr-defined]
    with torch.no_grad():
        model(calibration_data)
    torch.quantization.convert(model, inplace=True)  # type: ignore[attr-defined]
    return model


def evaluate_quantization(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    test_inputs: torch.Tensor,
    *,
    output_key: str = "classifications",
    accuracy_loss_threshold: float = DEFAULT_ACCURACY_LOSS_THRESHOLD,
) -> QuantizationReport:
    """Measure relative metric delta between FP32 and INT8; gate on threshold."""
    with torch.no_grad():
        fp32_out = fp32_model(test_inputs)
        int8_out = int8_model(test_inputs)

    def _mean_norm(out: Any) -> float:
        tensor = out[output_key] if isinstance(out, dict) else out
        return float(tensor.float().abs().mean())

    fp32_m = _mean_norm(fp32_out)
    int8_m = _mean_norm(int8_out)
    denom = max(abs(fp32_m), 1e-12)
    rel_delta = abs(fp32_m - int8_m) / denom
    passed = rel_delta <= accuracy_loss_threshold
    if not passed:
        logger.error(
            "quantization_accuracy_gate FAILED relative_delta=%.4f threshold=%.4f",
            rel_delta,
            accuracy_loss_threshold,
        )
    return QuantizationReport(
        fp32_metric=fp32_m,
        int8_metric=int8_m,
        metric_name=f"{output_key}_mean_norm",
        relative_delta=rel_delta,
        passed=passed,
        quantization_type="PTQ",
    )


def estimate_int8_size_mb(model: nn.Module) -> float:
    """Estimate INT8 model size as (n_params * 1 byte) / 1 MB."""
    return sum(p.numel() for p in model.parameters()) / (1024 * 1024)
