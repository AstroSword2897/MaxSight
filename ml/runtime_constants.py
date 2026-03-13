"""Runtime and safety gate constants for production. Align with docs/productization/02 and 04."""

# Critical path: urgency level at and above this is always surfaced (SG-07).
CRITICAL_URGENCY_THRESHOLD = 3  # danger; 0=safe, 1=caution, 2=warning, 3=danger

# Latency budgets (ms) for critical path (SG-03, SG-04).
LATENCY_MEDIAN_MS = 350
LATENCY_P95_MS = 600

# Overload guardrail: max alerts per minute in dense scenes unless emergency (SG-08).
ALERTS_PER_MINUTE_CAP = 12

# Minimum interval (s) between non-emergency outputs; emergency can bypass.
MIN_CHANNEL_INTERVAL_S = 0.3

# Safety gate thresholds for release validation (SG-01, SG-02, SG-05, SG-06).
HAZARD_RECALL_MIN = 0.95
FALSE_SAFE_RATE_MAX = 0.01
DIRECTION_CORRECTNESS_MIN = 0.90
DISTANCE_ZONE_ACCURACY_MIN = 0.85


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
