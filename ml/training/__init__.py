# MaxSight Training Module - Core training components.
# Defer heavy imports so contracts CI can import observability without torch.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ProductionTrainLoop",
    "train_model",
    "EMA",
    "DetectionMetrics",
    "match_batch",
    "match_predictions_to_gt",
    "SceneMetrics",
    "generate_evaluation_report",
    "benchmark_inference",
    "export_to_jit",
    "export_to_executorch",
    "export_to_coreml",
    "export_to_onnx",
    "quantize_model_int8",
]

# Satisfy static checkers for __all__ without importing torch at runtime.
if TYPE_CHECKING:
    from .benchmark import benchmark_inference
    from .evaluation import generate_evaluation_report
    from .export import export_to_coreml, export_to_executorch, export_to_jit, export_to_onnx
    from .matching import match_batch, match_predictions_to_gt
    from .metrics import DetectionMetrics
    from .quantization import quantize_model_int8
    from .scene_metrics import SceneMetrics
    from .train_loop import EMA, ProductionTrainLoop, train_model

_EXPORT_MODULES: dict[str, tuple[str, str]] = {
    "ProductionTrainLoop": (".train_loop", "ProductionTrainLoop"),
    "train_model": (".train_loop", "train_model"),
    "EMA": (".train_loop", "EMA"),
    "DetectionMetrics": (".metrics", "DetectionMetrics"),
    "match_batch": (".matching", "match_batch"),
    "match_predictions_to_gt": (".matching", "match_predictions_to_gt"),
    "SceneMetrics": (".scene_metrics", "SceneMetrics"),
    "generate_evaluation_report": (".evaluation", "generate_evaluation_report"),
    "benchmark_inference": (".benchmark", "benchmark_inference"),
    "export_to_jit": (".export", "export_to_jit"),
    "export_to_executorch": (".export", "export_to_executorch"),
    "export_to_coreml": (".export", "export_to_coreml"),
    "export_to_onnx": (".export", "export_to_onnx"),
    "quantize_model_int8": (".quantization", "quantize_model_int8"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        import importlib

        mod_name, attr = _EXPORT_MODULES[name]
        module = importlib.import_module(mod_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
