"""Tests for SCRUM-8,9,25,27,30: disability ontology, therapy constraints, PCA, scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── SCRUM-8: Disability ontology ──────────────────────────────────────────────

from ml.data.ontology.loader import DisabilityOntology


class TestDisabilityOntology:
    def test_loads_exactly_seven_disabilities(self):
        o = DisabilityOntology.load()
        o.validate()
        assert len(o.to_dict()["ids"]) == 7

    def test_all_expected_keys_present(self):
        o = DisabilityOntology.load()
        expected_keys = {
            "amblyopia",
            "cvi",
            "color_vision_deficiency",
            "low_vision_amd",
            "low_vision_diabetic_retinopathy",
            "low_vision_glaucoma",
            "retinitis_pigmentosa",
        }
        assert expected_keys == set(o.to_dict()["ids"])

    def test_each_entry_has_therapy_focus(self):
        o = DisabilityOntology.load()
        for did in o.to_dict()["ids"]:
            focus = o.therapy_focus_for(did)
            assert len(focus) > 0, f"{did} has no therapy_focus"

    def test_model_condition_keys_unique(self):
        o = DisabilityOntology.load()
        keys = [o.get(did).model_condition_key for did in o.to_dict()["ids"]]
        assert len(keys) == len(set(keys)), "duplicate model_condition_key"

    def test_get_unknown_id_raises(self):
        o = DisabilityOntology.load()
        with pytest.raises(KeyError):
            o.get("not_a_real_disability")

    def test_validate_fails_on_wrong_count(self):
        o = DisabilityOntology.load()
        del o._by_id[next(iter(o._by_id))]
        with pytest.raises(ValueError, match="exactly 7"):
            o.validate()

    def test_ontology_file_is_valid_json(self):
        import json

        path = PROJECT_ROOT / "ml/data/ontology/disability_ontology.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "disabilities" in data
        assert len(data["disabilities"]) == 7


# ── SCRUM-9: Therapy constraints ──────────────────────────────────────────────

from ml.therapy.constraints_loader import TherapyConstraints


class TestTherapyConstraints:
    def test_loads_without_error(self):
        c = TherapyConstraints.load()
        assert c.version == "1.0.0"

    def test_rate_limits_sane(self):
        c = TherapyConstraints.load()
        assert 1 <= c.max_prompts_per_minute <= 20
        assert c.min_gap_between_prompts_s >= 1.0

    def test_suppress_threshold_in_range(self):
        c = TherapyConstraints.load()
        assert 0.0 < c.suppress_threshold < 1.0

    def test_disallowed_phrases_non_empty(self):
        c = TherapyConstraints.load()
        assert len(c.disallowed_phrases) >= 5

    def test_disallowed_content_detection(self):
        c = TherapyConstraints.load()
        assert c.is_disallowed_content("you are sick and need treatment")
        assert c.is_disallowed_content("prescribe this medication")
        assert not c.is_disallowed_content("Take a slow breath and focus ahead")

    def test_scoring_weights_sum_to_one(self):
        c = TherapyConstraints.load()
        total = sum(c.scoring_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_disability_routing_covers_known_ids(self):
        c = TherapyConstraints.load()
        o = DisabilityOntology.load()
        for key in c.disability_routing:
            assert key in o.to_dict()["ids"] or key.startswith("low_vision")


# ── SCRUM-25: PCA feature transform ──────────────────────────────────────────

from ml.data.feature_transform import (
    FeatureTransformArtifact,
    fit_feature_transform,
    transform_features,
    transform_summary,
)


class TestFeatureTransform:
    def test_fit_produces_artifact(self):
        X = np.random.randn(100, 64)
        art = fit_feature_transform(X, variance_threshold=0.90)
        assert isinstance(art, FeatureTransformArtifact)
        assert art.n_components < 64

    def test_explained_variance_meets_threshold(self):
        X = np.random.randn(200, 32)
        threshold = 0.85
        art = fit_feature_transform(X, variance_threshold=threshold)
        assert float(art.explained_variance_ratio.sum()) >= threshold - 0.05

    def test_transform_output_shape(self):
        X = np.random.randn(50, 16)
        art = fit_feature_transform(X, n_components=5)
        out = transform_features(art, X[:10])
        assert out.shape == (10, 5)

    def test_save_and_load_roundtrip(self, tmp_path):
        X = np.random.randn(40, 12)
        art = fit_feature_transform(X, n_components=4)
        p = tmp_path / "pca.json"
        art.save(p)
        loaded = FeatureTransformArtifact.load(p)
        out_orig = transform_features(art, X[:5])
        out_loaded = transform_features(loaded, X[:5])
        np.testing.assert_allclose(out_orig, out_loaded, rtol=1e-5)

    def test_single_sample_transform(self):
        X = np.random.randn(30, 8)
        art = fit_feature_transform(X, n_components=3)
        single = X[0]
        out = transform_features(art, single)
        assert out.shape == (1, 3)

    def test_fit_requires_at_least_two_samples(self):
        with pytest.raises(ValueError):
            fit_feature_transform(np.random.randn(1, 4))

    def test_summary_keys(self):
        X = np.random.randn(40, 10)
        art = fit_feature_transform(X)
        summary = transform_summary(art)
        assert "n_components" in summary
        assert "explained_variance_sum" in summary


# ── SCRUM-27 / SCRUM-30: Therapy scoring ─────────────────────────────────────

from ml.therapy.scoring import TherapyScoringModel


class TestTherapyScoringModel:
    def _ctx(self, stress=0.5, cognitive=0.4, uncertainty=0.1):
        return {
            "environment_stress_level": stress,
            "cognitive_load_estimate": cognitive,
            "uncertainty": uncertainty,
        }

    def test_score_trace_fields_complete(self):
        m = TherapyScoringModel()
        trace = m.score_intervention(self._ctx(), "grounding")
        d = trace.to_dict()
        for key in ("base_score", "stress_component", "final_score", "safety_penalty"):
            assert key in d

    def test_final_score_in_range(self):
        m = TherapyScoringModel()
        for _ in range(20):
            ctx = self._ctx(
                stress=np.random.uniform(0, 1),
                cognitive=np.random.uniform(0, 1),
                uncertainty=np.random.uniform(0, 1),
            )
            trace = m.score_intervention(ctx, "grounding")
            assert 0.0 <= trace.final_score <= 1.0

    def test_high_uncertainty_applies_penalty(self):
        m = TherapyScoringModel()
        high_unc = m.score_intervention(self._ctx(uncertainty=0.9), "grounding")
        low_unc = m.score_intervention(self._ctx(uncertainty=0.1), "grounding")
        assert high_unc.safety_penalty > 0.0
        assert high_unc.final_score < low_unc.final_score

    def test_update_effectiveness_shifts_learned_adjustment(self):
        m = TherapyScoringModel()
        ctx = self._ctx()
        m.update_effectiveness("grounding", 0.9)
        trace_after = m.score_intervention(ctx, "grounding")
        assert trace_after.learned_adjustment > 0.0

    def test_recommend_returns_valid_intervention_type(self):
        m = TherapyScoringModel()
        result = m.recommend_intervention_type("low_vision_amd", self._ctx())
        c = TherapyConstraints.load()
        routing = c.disability_routing.get("low_vision_amd", [])
        assert result in routing

    def test_score_deterministic_for_fixed_input(self):
        m = TherapyScoringModel()
        ctx = self._ctx(stress=0.6, cognitive=0.3, uncertainty=0.15)
        t1 = m.score_intervention(ctx, "breathing")
        t2 = m.score_intervention(ctx, "breathing")
        assert t1.final_score == t2.final_score
