"""Runtime and safety gate constants for production. Align with docs/productization/02 and 04."""

# Critical path: urgency level at and above this is always surfaced (SG-07).
CRITICAL_URGENCY_THRESHOLD = 3  # danger; 0=safe, 1=caution, 2=warning, 3=danger

# Latency budgets (ms) for critical path (SG-03, SG-04). Target: 80 ms.
LATENCY_MEDIAN_MS = 80
LATENCY_P95_MS = 80

# Overload guardrail: max alerts per minute in dense scenes unless emergency (SG-08).
ALERTS_PER_MINUTE_CAP = 12

# Minimum interval (s) between non-emergency outputs; emergency can bypass.
MIN_CHANNEL_INTERVAL_S = 0.3

# Therapy subsystem: do not prompt when perception uncertainty is above this (fail-safe).
THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD = 0.7
# Max therapy prompts per minute to avoid overload (separate from hazard alerts).
THERAPY_MAX_PROMPTS_PER_MINUTE = 2
# Min gap (s) between therapy prompts.
THERAPY_MIN_GAP_BETWEEN_PROMPTS_S = 10.0

# Safety gate thresholds for release validation (SG-01, SG-02, SG-05, SG-06).
HAZARD_RECALL_MIN = 0.95
FALSE_SAFE_RATE_MAX = 0.01
DIRECTION_CORRECTNESS_MIN = 0.90
DISTANCE_ZONE_ACCURACY_MIN = 0.85

# Default MaxSightCNN parameter envelope for CI size assertions (full tier wiring).
DEFAULT_MODEL_MIN_PARAMS = 90_000_000
DEFAULT_MODEL_MAX_PARAMS = 400_000_000
DEFAULT_MODEL_INT8_MAX_MB = 400.0
# Model-output keys that the T5-only MVP runtime is allowed to depend on.
MVP_MODEL_OUTPUT_KEYS = (
    "classifications",  # logits for detection head.
    "boxes",  # bounding boxes for hazards/objects.
    "objectness",  # objectness scores for detections.
    "text_regions",  # OCR/text-region logits.
    "urgency_scores",  # per-image or per-detection urgency.
    "distance_zones",  # coarse distance buckets.
    "precise_distances",  # per-detection distance estimates.
    "distance_uncertainties",
    "uncertainty",  # global uncertainty scalar.
    "temporal_consistency",  # temporal stability features.
)


def check_safety_gate_report(metrics: dict) -> tuple[bool, list[str]]:
    """Return (all_passed, list of failed gate ids). metrics keys: hazard_recall, false_safe_rate, direction_correctness, distance_zone_accuracy."""
    failed = []
    if metrics.get("hazard_recall", 0) < HAZARD_RECALL_MIN:
        failed.append("SG-01")
    if metrics.get("false_safe_rate", 1.0) > FALSE_SAFE_RATE_MAX:
        failed.append("SG-02")
    if metrics.get("direction_correctness", 0) < DIRECTION_CORRECTNESS_MIN:
        failed.append("SG-05")
    if metrics.get("distance_zone_accuracy", 0) < DISTANCE_ZONE_ACCURACY_MIN:
        failed.append("SG-06")
    return (len(failed) == 0, failed)


def filter_mvp_model_outputs(outputs: dict, *, training: bool = False) -> dict:
    """Filter raw model outputs down to the T5 MVP runtime surface.

    Training keeps the full dictionary so losses and diagnostics are unaffected;
    runtime paths (eval/inference) can call this to enforce a smaller contract.
    """
    if training:
        return outputs
    return {k: v for k, v in outputs.items() if k in MVP_MODEL_OUTPUT_KEYS}
