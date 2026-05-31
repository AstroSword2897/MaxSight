"""CoreML / ONNX export parity harness: verify output tolerance vs PyTorch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 1e-3


@dataclass
class ParityReport:
    """Comparison result between PyTorch and exported model outputs."""

    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: float
    format: str
    failures: list

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_abs_diff": round(self.max_abs_diff, 8),
            "mean_abs_diff": round(self.mean_abs_diff, 8),
            "tolerance": self.tolerance,
            "format": self.format,
            "failures": self.failures,
        }


def check_onnx_parity(
    pytorch_model: nn.Module,
    onnx_path: Path,
    test_input: torch.Tensor,
    *,
    output_key: str = "classifications",
    tolerance: float = DEFAULT_TOLERANCE,
) -> ParityReport:
    """Compare ONNX runtime output against PyTorch output."""
    failures = []
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        return ParityReport(False, 0.0, 0.0, tolerance, "onnx", ["onnxruntime not installed"])

    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(test_input)
    pt_arr = (pt_out[output_key] if isinstance(pt_out, dict) else pt_out).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    onnx_arr = sess.run(None, {inp_name: test_input.numpy()})[0]

    abs_diff = np.abs(pt_arr - onnx_arr)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    if max_diff > tolerance:
        failures.append(f"max_abs_diff={max_diff:.6f} exceeds tolerance={tolerance}")

    return ParityReport(
        passed=len(failures) == 0,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        tolerance=tolerance,
        format="onnx",
        failures=failures,
    )


def check_coreml_parity(
    pytorch_model: nn.Module,
    coreml_path: Path,
    test_input: torch.Tensor,
    *,
    output_key: str = "classifications",
    tolerance: float = DEFAULT_TOLERANCE,
) -> ParityReport:
    """Compare CoreML output against PyTorch output (macOS only)."""
    failures = []
    try:
        import coremltools as ct  # type: ignore[import-untyped]
    except ImportError:
        return ParityReport(False, 0.0, 0.0, tolerance, "coreml", ["coremltools not installed"])

    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(test_input)
    pt_arr = (pt_out[output_key] if isinstance(pt_out, dict) else pt_out).numpy()

    mlmodel = ct.models.MLModel(str(coreml_path))
    np_input = test_input.numpy()
    cml_out = mlmodel.predict({"input": np_input})
    cml_arr = np.array(list(cml_out.values())[0])

    abs_diff = np.abs(pt_arr - cml_arr)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    if max_diff > tolerance:
        failures.append(f"max_abs_diff={max_diff:.6f} exceeds tolerance={tolerance}")

    return ParityReport(
        passed=len(failures) == 0,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        tolerance=tolerance,
        format="coreml",
        failures=failures,
    )
