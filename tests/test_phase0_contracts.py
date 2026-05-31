"""Tests for SCRUM-34: runtime contracts, schemas, and tier routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from ml.runtime.contracts import (
    REQUIRED_CRITICAL_EVENT_FIELDS,
    REQUIRED_RUNTIME_RESPONSE_FIELDS,
    ComputeTier,
    CriticalEvent,
    DegradedMode,
    ModelOutputContract,
    RagContext,
    RuntimeRequest,
    RuntimeResponse,
    TherapyRecommendation,
    validate_critical_event,
    validate_model_outputs,
    validate_runtime_response,
)
from ml.runtime.tier_router import TierRouter


def _make_critical_event(**overrides) -> dict:
    base = {
        "event_type": "obstacle",
        "urgency": 2,
        "direction": "center",
        "distance_zone": "near",
        "confidence": 0.92,
        "uncertainty": 0.08,
        "timestamp_source": 1000.0,
        "timestamp_emit": 1000.05,
    }
    base.update(overrides)
    return base


def _make_response(**overrides) -> dict:
    base = {
        "frame_id": "f1",
        "tier": "silver",
        "degraded_mode": "D0",
        "classifications": [],
        "critical_events": [_make_critical_event()],
        "secondary_events": [],
        "therapy": [],
        "latency_ms": 45.0,
    }
    base.update(overrides)
    return base


# ── CriticalEvent contract ────────────────────────────────────────────────────


class TestCriticalEventContract:
    def test_valid_event_passes(self):
        validate_critical_event(_make_critical_event())

    def test_missing_urgency_raises(self):
        with pytest.raises(ValueError, match="urgency"):
            validate_critical_event(
                {k: v for k, v in _make_critical_event().items() if k != "urgency"}
            )

    def test_missing_direction_raises(self):
        with pytest.raises(ValueError):
            validate_critical_event(
                {k: v for k, v in _make_critical_event().items() if k != "direction"}
            )

    def test_all_required_fields_checked(self):
        for field in REQUIRED_CRITICAL_EVENT_FIELDS:
            with pytest.raises(ValueError):
                validate_critical_event(
                    {k: v for k, v in _make_critical_event().items() if k != field}
                )

    def test_dataclass_roundtrip(self):
        ev = CriticalEvent(
            event_type="vehicle",
            urgency=3,
            direction="left",
            distance_zone="near",
            confidence=0.95,
            uncertainty=0.05,
            timestamp_source=1.0,
            timestamp_emit=1.04,
        )
        d = ev.to_dict()
        validate_critical_event(d)
        assert d["urgency"] == 3


# ── RuntimeResponse contract ──────────────────────────────────────────────────


class TestRuntimeResponseContract:
    def test_valid_response_passes(self):
        validate_runtime_response(_make_response())

    def test_missing_frame_id_raises(self):
        with pytest.raises(ValueError, match="frame_id"):
            validate_runtime_response(
                {k: v for k, v in _make_response().items() if k != "frame_id"}
            )

    def test_all_required_fields_enforced(self):
        for field in REQUIRED_RUNTIME_RESPONSE_FIELDS:
            with pytest.raises(ValueError):
                validate_runtime_response({k: v for k, v in _make_response().items() if k != field})

    def test_nested_critical_event_validated(self):
        resp = _make_response(critical_events=[{"event_type": "obstacle"}])
        with pytest.raises(ValueError):
            validate_runtime_response(resp)

    def test_full_dataclass_to_dict(self):
        resp = RuntimeResponse(
            frame_id="f42",
            tier=ComputeTier.GOLD,
            degraded_mode=DegradedMode.D0_NORMAL,
            critical_events=[
                CriticalEvent(
                    event_type="curb",
                    urgency=2,
                    direction="right",
                    distance_zone="near",
                    confidence=0.88,
                    uncertainty=0.12,
                    timestamp_source=2.0,
                    timestamp_emit=2.05,
                )
            ],
            therapy=[
                TherapyRecommendation(
                    intervention_type="grounding",
                    channel="audio",
                    content="Take a breath.",
                    intensity=0.5,
                    score=0.82,
                )
            ],
            rag=RagContext(
                guidance="therapy_prompt_high_confidence",
                advisory_score=0.8,
                retrieved_count=3,
                grounded=True,
            ),
            latency_ms=38.2,
        )
        d = resp.to_dict()
        validate_runtime_response(d)
        assert d["tier"] == "gold"
        assert d["rag"]["grounded"] is True


# ── JSON Schema artifact ──────────────────────────────────────────────────────


class TestJsonSchema:
    def test_schema_file_exists(self):
        path = PROJECT_ROOT / "docs/contracts/schemas/runtime_response.json"
        assert path.exists()

    def test_schema_required_fields_complete(self):
        path = PROJECT_ROOT / "docs/contracts/schemas/runtime_response.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_required = set(data["required"])
        assert REQUIRED_RUNTIME_RESPONSE_FIELDS.issubset(schema_required)

    def test_openapi_exists(self):
        path = PROJECT_ROOT / "docs/contracts/openapi.yaml"
        assert path.exists()
        text = path.read_text()
        assert "/v1/runtime/process" in text
        assert "RuntimeResponse" in text


# ── TierRouter ────────────────────────────────────────────────────────────────


class TestTierRouter:
    def test_resolves_all_three_tiers(self):
        router = TierRouter()
        for tier in ComputeTier:
            profile = router.resolve(tier)
            assert profile.tier == tier

    def test_gold_downgrades_on_battery_low(self):
        router = TierRouter()
        profile = router.resolve(ComputeTier.GOLD, battery_low=True)
        assert profile.tier == ComputeTier.SILVER

    def test_gold_downgrades_on_tight_budget(self):
        router = TierRouter()
        bronze_latency = router.resolve(ComputeTier.BRONZE).max_latency_ms
        profile = router.resolve(ComputeTier.GOLD, latency_budget_ms=bronze_latency - 1)
        assert profile.tier in (ComputeTier.SILVER, ComputeTier.BRONZE)

    def test_trace_keys(self):
        router = TierRouter()
        profile = router.resolve(ComputeTier.SILVER)
        trace = router.to_trace(profile)
        for key in ("tier", "model_tier", "max_latency_ms", "enable_rag"):
            assert key in trace

    def test_missing_tier_raises(self):
        router = TierRouter(config_dir=Path("/tmp/__no_tiers__"))
        with pytest.raises(KeyError):
            router.resolve(ComputeTier.GOLD)


# ── Ownership map ─────────────────────────────────────────────────────────────


def test_module_ownership_doc_exists():
    path = PROJECT_ROOT / "docs/architecture/module_ownership.md"
    assert path.exists()
    content = path.read_text()
    assert "ml/runtime/" in content
    assert "ml/therapy/" in content


# ── Contract validator script ─────────────────────────────────────────────────


def test_validate_runtime_contracts_script():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/infra/validate_runtime_contracts.py")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr


# ── RuntimeOrchestrator ────────────────────────────────────────────────────────


class TestRuntimeOrchestrator:
    """SCRUM-18/19: RuntimeOrchestrator builds valid RuntimeResponse end-to-end."""

    def _make_request(self, **kwargs):
        return RuntimeRequest(
            frame_id="test-frame-001",
            perception={"environment_stress_level": 0.3, "uncertainty": 0.2},
            **kwargs,
        )

    def test_basic_response_shape(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        resp = orch.process(self._make_request())
        assert resp.frame_id == "test-frame-001"
        assert isinstance(resp.tier, ComputeTier)
        assert isinstance(resp.degraded_mode, DegradedMode)
        assert resp.latency_ms >= 0

    def test_response_validates_contract(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        resp = orch.process(self._make_request())
        d = resp.to_dict()
        validate_runtime_response(d)

    def test_therapy_disabled_yields_empty_list(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        resp = orch.process(self._make_request(enable_therapy=False))
        assert resp.therapy == []

    def test_high_stress_produces_therapy(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        req = RuntimeRequest(
            frame_id="stress-test",
            perception={
                "environment_stress_level": 0.9,
                "cognitive_load_estimate": 0.2,
                "uncertainty": 0.1,
            },
            enable_therapy=True,
            tier=ComputeTier.SILVER,
        )
        resp = orch.process(req)
        assert len(resp.therapy) > 0
        for rec in resp.therapy:
            assert rec.intervention_type
            assert 0.0 <= rec.score <= 1.0
            assert "final_score" in rec.score_trace

    def test_score_trace_populated(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        req = RuntimeRequest(
            frame_id="trace-test",
            perception={
                "environment_stress_level": 0.85,
                "cognitive_load_estimate": 0.3,
                "uncertainty": 0.1,
            },
            enable_therapy=True,
            tier=ComputeTier.SILVER,
        )
        resp = orch.process(req)
        if resp.therapy:
            trace = resp.therapy[0].score_trace
            assert "final_score" in trace
            assert "stress_component" in trace

    def test_rag_disabled_on_bronze(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        req = RuntimeRequest(
            frame_id="bronze-test",
            perception={"environment_stress_level": 0.3},
            enable_rag=True,
            tier=ComputeTier.BRONZE,
        )
        resp = orch.process(req)
        # Bronze tier has enable_rag=False in tier config.
        assert resp.rag is None

    def test_tier_trace_present(self):
        from ml.runtime.mode import RuntimeOrchestrator

        orch = RuntimeOrchestrator()
        resp = orch.process(self._make_request())
        assert "tier_profile" in resp.trace
        assert "tier" in resp.trace["tier_profile"]


# ── Pre-SageMaker gate ─────────────────────────────────────────────────────────


def test_pre_sagemaker_gate_passes():
    """SCRUM-29: Pre-SageMaker gate must exit 0 with all checks passing."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/infra/pre_sagemaker_gate.py")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"Gate failed:\n{result.stdout}\n{result.stderr}"
    assert "GATE PASSED" in result.stdout


def test_required_fields_track_dataclass_definitions():
    """Contract required fields must stay aligned with dataclass definitions."""
    from dataclasses import fields

    from ml.runtime.contracts import CriticalEvent, RuntimeResponse

    assert (
        frozenset(f.name for f in fields(CriticalEvent) if f.name != "distance_meters")
        == REQUIRED_CRITICAL_EVENT_FIELDS
    )
    assert (
        frozenset(f.name for f in fields(RuntimeResponse) if f.name not in {"rag", "trace"})
        == REQUIRED_RUNTIME_RESPONSE_FIELDS
    )


class TestModelOutputContract:
    def test_validate_model_outputs_strips_non_mvp_keys(self):
        outputs = {
            "classifications": [1, 2],
            "scene_embedding": [0.1],
            "user_id": "u1",
        }
        filtered = validate_model_outputs(outputs)
        assert "classifications" in filtered
        assert "user_id" in filtered
        assert "scene_embedding" not in filtered

    def test_model_output_contract_matches_mvp_keys(self):
        from ml.runtime_constants import MVP_MODEL_OUTPUT_KEYS

        contract = ModelOutputContract()
        assert contract.allowed_keys == frozenset(MVP_MODEL_OUTPUT_KEYS)

    def test_model_output_schema_file_exists(self):
        schema_path = PROJECT_ROOT / "docs/contracts/schemas/model_output.json"
        assert schema_path.exists()
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        assert "properties" in data
        assert set(data["properties"]) == set(ModelOutputContract().allowed_keys)
