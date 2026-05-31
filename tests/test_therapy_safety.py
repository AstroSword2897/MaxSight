"""Safety-adjacent tests for TherapySafetyLayer and TherapyDecisionEngine."""

from __future__ import annotations

import threading

from ml.therapy.situation_understanding import SituationContext
from ml.therapy.therapy_decision_engine import InterventionStrength, TherapyDecisionEngine
from ml.therapy.therapy_safety import TherapySafetyLayer


def _context(**kwargs) -> SituationContext:
    return SituationContext(**kwargs)


class TestTherapySafetyLayerSuppression:
    def test_suppresses_above_uncertainty_threshold(self):
        layer = TherapySafetyLayer(uncertainty_suppress_threshold=0.5)
        suppress, reason = layer.should_suppress(_context(uncertainty=0.6), current_time=0.0)
        assert suppress is True
        assert reason == "uncertainty_above_threshold"

    def test_suppresses_when_min_gap_not_elapsed(self):
        layer = TherapySafetyLayer(min_gap_s=5.0)
        layer.record_prompt_delivered(current_time=100.0)
        suppress, reason = layer.should_suppress(_context(), current_time=103.0)
        assert suppress is True
        assert reason == "min_gap_not_elapsed"

    def test_suppresses_when_max_prompts_per_minute_reached(self):
        layer = TherapySafetyLayer(max_prompts_per_minute=2, min_gap_s=0.0)
        layer.record_prompt_delivered(current_time=10.0)
        layer.record_prompt_delivered(current_time=11.0)
        suppress, reason = layer.should_suppress(_context(), current_time=12.0)
        assert suppress is True
        assert reason == "max_prompts_per_minute"

    def test_sanitize_strips_diagnostic_phrases(self):
        layer = TherapySafetyLayer()
        assert layer.sanitize_content("You should see a doctor for this") == ""
        assert layer.sanitize_content("Take a deep breath") == "Take a deep breath"

    def test_disallowed_phrases_are_injected_and_tested_in_isolation(self):
        layer = TherapySafetyLayer(disallowed_phrases=frozenset({"forbidden"}))
        assert layer.sanitize_content("this is forbidden") == ""
        assert layer.sanitize_content("this is fine") == "this is fine"

    def test_thread_safe_prompt_accounting(self):
        layer = TherapySafetyLayer(max_prompts_per_minute=100, min_gap_s=0.0)
        errors: list[Exception] = []

        def worker(start: float) -> None:
            try:
                for i in range(20):
                    t = start + i * 0.01
                    layer.record_prompt_delivered(t)
                    layer.should_suppress(_context(), t)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(float(idx),)) for idx in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        assert not errors
        assert len(layer._prompts_this_minute) <= 100


class TestTherapyScoringObservability:
    def test_effectiveness_snapshot_tracks_updates(self):
        from ml.therapy.scoring import TherapyScoringModel

        model = TherapyScoringModel()
        model.update_effectiveness("grounding", 0.9)
        model.update_effectiveness("grounding", 0.3)
        snapshot = model.effectiveness_snapshot()
        assert "grounding" in snapshot
        assert 0.0 < snapshot["grounding"] < 1.0


class TestTherapyDecisionEngine:
    def test_high_stress_branch(self):
        engine = TherapyDecisionEngine(stress_trigger_threshold=0.6, high_stress_threshold=0.75)
        decision = engine.decide(_context(environment_stress_level=0.9))
        assert decision.should_intervene is True
        assert decision.strength == InterventionStrength.HIGH_STRESS

    def test_navigation_reassurance_low_stress_branch(self):
        engine = TherapyDecisionEngine(stress_trigger_threshold=0.6, high_stress_threshold=0.75)
        decision = engine.decide(
            _context(
                environment_stress_level=0.45,
                navigation_complexity=0.7,
            )
        )
        assert decision.should_intervene is True
        assert decision.intervention_type == "navigation_reassurance"
        assert decision.reason == "navigation_reassurance"

    def test_below_threshold_returns_no_intervention(self):
        engine = TherapyDecisionEngine()
        decision = engine.decide(_context(environment_stress_level=0.1))
        assert decision.should_intervene is False
        assert decision.reason == "below_threshold"
