"""Production hardening tests: pipeline latency, priority filter, temporal smoother, safety bias, thermal throttling, alert cooldown. Run with: pytest tests/test_production_hardening.py -v."""

import sys
import time
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock Flask before importing web_simulator (needed for PipelineLatencyTracker)
sys.modules["flask"] = unittest.mock.MagicMock()
sys.modules["flask_cors"] = unittest.mock.MagicMock()

from ml.utils.alert_cooldown import AlertCooldownFilter
from ml.utils.priority_filter import PriorityBudgetFilter
from ml.utils.stage_a_smoother import StageATemporalSmoother
from tools.simulation.simulator.inference_engine import ThermalThrottleDetector
from tools.simulation.web_simulator import PipelineLatencyTracker


class TestPriorityBudgetFilter:
    def test_empty_list(self):
        f = PriorityBudgetFilter(max_alerts_per_frame=5)
        assert f.filter_alerts([]) == []

    def test_fewer_than_max(self):
        f = PriorityBudgetFilter(max_alerts_per_frame=5)
        dets = [
            {"urgency": 0, "confidence": 0.5, "distance": "near"},
            {"urgency": 1, "confidence": 0.8, "distance": "medium"},
        ]
        out = f.filter_alerts(dets)
        assert len(out) == 2

    def test_more_than_max_returns_top_n(self):
        f = PriorityBudgetFilter(max_alerts_per_frame=3)
        dets = [
            {"urgency": 0, "confidence": 0.3, "distance": "far"},
            {"urgency": 2, "confidence": 0.9, "distance": "near"},
            {"urgency": 1, "confidence": 0.7, "distance": "medium"},
            {"urgency": 3, "confidence": 0.8, "distance": "near"},
            {"urgency": 0, "confidence": 0.5, "distance": "medium"},
        ]
        out = f.filter_alerts(dets)
        assert len(out) == 3
        # Top 3 by priority (urgency * conf * 1/(dist_ord+1)): high urgency+near first.
        urgencies = [d["urgency"] for d in out]
        assert 3 in urgencies and 2 in urgencies

    def test_priority_score_ordering(self):
        f = PriorityBudgetFilter(max_alerts_per_frame=1)
        dets = [
            {"urgency": 0, "confidence": 0.5, "distance": "far"},
            {"urgency": 2, "confidence": 0.6, "distance": "near"},
        ]
        out = f.filter_alerts(dets)
        assert len(out) == 1
        assert out[0]["urgency"] == 2


class TestStageATemporalSmoother:
    def test_smooth_detections_passthrough_first_frame(self):
        s = StageATemporalSmoother(alpha=0.7, max_age=5)
        dets = [
            {"class_name": "car", "box": [0.5, 0.5, 0.1, 0.1], "confidence": 0.8},
        ]
        out = s.smooth_detections(dets)
        assert len(out) == 1
        assert out[0]["confidence"] == 0.8

    def test_smooth_detections_ema(self):
        s = StageATemporalSmoother(alpha=0.7, max_age=5)
        dets1 = [{"class_name": "car", "box": [0.5, 0.5, 0.1, 0.1], "confidence": 0.8}]
        s.smooth_detections(dets1)
        dets2 = [{"class_name": "car", "box": [0.52, 0.52, 0.1, 0.1], "confidence": 0.9}]
        out = s.smooth_detections(dets2)
        assert len(out) == 1
        # EMA: 0.7 * 0.8 + 0.3 * 0.9 = 0.83.
        assert 0.82 <= out[0]["confidence"] <= 0.84

    def test_get_object_id_stable(self):
        s = StageATemporalSmoother(alpha=0.7, max_age=5)
        dets = [{"class_name": "person", "box": [0.3, 0.4, 0.05, 0.1], "confidence": 0.7}]
        out1 = s.smooth_detections(dets)
        out2 = s.smooth_detections(dets)
        assert len(out1) == 1 and len(out2) == 1


class TestAlertCooldownFilter:
    def test_cooldown_blocks_repeat(self):
        c = AlertCooldownFilter(cooldown_frames=3)
        det = {"class_name": "car", "box": [0.5, 0.5, 0.1, 0.1]}
        out1 = c.filter_alerts([det], frame_id=0)
        assert len(out1) == 1
        out2 = c.filter_alerts([det], frame_id=1)
        assert len(out2) == 0
        out3 = c.filter_alerts([det], frame_id=2)
        assert len(out3) == 0
        out4 = c.filter_alerts([det], frame_id=4)
        assert len(out4) == 1

    def test_different_objects_not_blocked(self):
        c = AlertCooldownFilter(cooldown_frames=2)
        d1 = {"class_name": "car", "box": [0.1, 0.1, 0.1, 0.1]}
        d2 = {"class_name": "person", "box": [0.8, 0.8, 0.1, 0.1]}
        out = c.filter_alerts([d1, d2], frame_id=0)
        assert len(out) == 2


class TestThermalThrottleDetector:
    def test_requires_enough_samples(self):
        d = ThermalThrottleDetector(window_size_seconds=30.0)
        assert d.check_thermal_throttle(50.0) is False
        for _ in range(8):
            d.check_thermal_throttle(50.0)
        assert d.check_thermal_throttle(50.0) is False

    def test_detects_sustained_degradation(self):
        d = ThermalThrottleDetector(window_size_seconds=30.0)
        base = 20.0
        for _ in range(5):
            d.check_thermal_throttle(base)
        for _ in range(12):
            d.check_thermal_throttle(60.0)
        assert d.check_thermal_throttle(60.0) is True


class TestPipelineLatencyTracker:
    def test_stages_recorded(self):
        t = PipelineLatencyTracker()
        t.start_stage("preprocess")
        time.sleep(0.01)
        t.end_stage()
        t.start_stage("model")
        time.sleep(0.02)
        t.end_stage()
        b = t.get_breakdown()
        assert "preprocess" in b and "model" in b and "total_ms" in b
        assert b["preprocess"] >= 10 and b["model"] >= 20


class TestSafetyBiasUrgency:
    def test_get_urgency_with_safety_bias(self):
        from ml.models.maxsight_cnn import MaxSightCNN

        model = MaxSightCNN(num_classes=80)
        # Hazard class + confidence + large box -> at least warning.
        u = model._get_urgency("car", box_size=0.25, confidence=0.5)
        assert u >= 2
        # Non-hazard, small box: low urgency (0 or 1 depending on _urgency_map)
        u_safe = model._get_urgency("chair", box_size=0.01, confidence=0.9)
        assert u_safe <= 1
        # Large/close object gets +1 urgency.
        u_large = model._get_urgency("chair", box_size=0.3, confidence=0.5)
        assert u_large >= 1
