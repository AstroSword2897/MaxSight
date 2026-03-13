"""Unit tests for runtime and safety gate constants and critical-path behavior."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRuntimeConstants:
    """Constants must match docs/productization/02 and 04."""

    def test_critical_urgency_threshold(self):
        from ml.runtime_constants import CRITICAL_URGENCY_THRESHOLD
        assert CRITICAL_URGENCY_THRESHOLD == 3

    def test_latency_budgets(self):
        from ml.runtime_constants import LATENCY_MEDIAN_MS, LATENCY_P95_MS
        assert LATENCY_MEDIAN_MS <= 350
        assert LATENCY_P95_MS <= 600

    def test_alerts_per_minute_cap(self):
        from ml.runtime_constants import ALERTS_PER_MINUTE_CAP
        assert ALERTS_PER_MINUTE_CAP == 12

    def test_safety_gate_thresholds(self):
        from ml.runtime_constants import (
            HAZARD_RECALL_MIN,
            FALSE_SAFE_RATE_MAX,
            DIRECTION_CORRECTNESS_MIN,
            DISTANCE_ZONE_ACCURACY_MIN,
        )
        assert HAZARD_RECALL_MIN >= 0.95
        assert FALSE_SAFE_RATE_MAX <= 0.01
        assert DIRECTION_CORRECTNESS_MIN >= 0.90
        assert DISTANCE_ZONE_ACCURACY_MIN >= 0.85

    def test_check_safety_gate_report_pass(self):
        from ml.runtime_constants import check_safety_gate_report
        metrics = {
            "hazard_recall": 0.96,
            "false_safe_rate": 0.005,
            "direction_correctness": 0.92,
            "distance_zone_accuracy": 0.87,
        }
        passed, failed = check_safety_gate_report(metrics)
        assert passed is True
        assert len(failed) == 0

    def test_check_safety_gate_report_fail(self):
        from ml.runtime_constants import check_safety_gate_report
        metrics = {
            "hazard_recall": 0.90,
            "false_safe_rate": 0.02,
        }
        passed, failed = check_safety_gate_report(metrics)
        assert passed is False
        assert "SG-01" in failed
        assert "SG-02" in failed


class TestSchedulerCriticalPath:
    """Critical detections must always be surfaced (SG-07)."""

    def test_critical_detections_always_included_under_uncertainty(self):
        from ml.utils.output_scheduler import CrossModalScheduler, OutputConfig, AlertFrequency
        from ml.runtime_constants import CRITICAL_URGENCY_THRESHOLD

        config = OutputConfig(alert_frequency=AlertFrequency.LOW, uncertainty_threshold=0.3)
        scheduler = CrossModalScheduler(config)

        detections = [
            {"urgency": 3, "priority": 95, "class_name": "vehicle", "box": [0.5, 0.5, 0.1, 0.1], "findability": 0.8},
            {"urgency": 0, "priority": 10, "class_name": "chair", "box": [0.2, 0.2, 0.1, 0.1], "findability": 0.5},
        ]
        model_outputs = {"uncertainty": torch.tensor(0.9)}
        scheduled = scheduler.schedule_outputs(detections, model_outputs, timestamp=1000.0)

        assert len([d for d in detections if d.get("urgency", 0) >= CRITICAL_URGENCY_THRESHOLD]) == 1
        assert len(scheduled) >= 1, "critical detection must produce at least one output (SG-07)"

    def test_scheduler_uses_runtime_constants(self):
        from ml.utils.output_scheduler import CrossModalScheduler, OutputConfig, AlertFrequency
        from ml.runtime_constants import MIN_CHANNEL_INTERVAL_S

        config = OutputConfig(alert_frequency=AlertFrequency.MEDIUM)
        scheduler = CrossModalScheduler(config)
        assert scheduler.min_channel_interval == MIN_CHANNEL_INTERVAL_S
