# MaxSight Training Module - Core training components.
from .benchmark import benchmark_inference
from .evaluation import generate_evaluation_report
from .export import export_to_coreml, export_to_executorch, export_to_jit, export_to_onnx
from .matching import match_batch, match_predictions_to_gt
from .metrics import DetectionMetrics
from .quantization import quantize_model_int8
from .scene_metrics import SceneMetrics
from .train_loop import EMA, ProductionTrainLoop, train_model

# Loss functions are in ml/training/losses.py but MaxSightLoss doesn't exist.
# Individual loss functions are available: ObjectDetectionLoss, OCRLoss, etc.

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
