"""Tests for RAG SLO contracts, failure taxonomy, and RAGReliabilityWrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.retrieval.rag_hardening import HardenedRagResult, RetrievalResult
from ml.retrieval.rag_reliability import (
    RAGAlertLevel,
    RAGFailureType,
    RAGFallbackMode,
    RAGMetrics,
    RAGReliabilityWrapper,
    RAGSLO,
    RAGSLOViolation,
    RAGWindowMetrics,
    classify_rag_failure,
    compute_window,
    decide_fallback,
    evaluate_slo,
    map_violation_to_alert,
    metrics_from_hardened_result,
    wrap_rag_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLO_PATH = PROJECT_ROOT / "ml/config/rag_slo.yaml"


def _metrics(**overrides) -> RAGMetrics:
    base = {
        "latency_ms": 50.0,
        "retrieved_docs": 3,
        "fallback_used": False,
        "query_embedding_norm": 0.0,
        "retrieval_score_max": 0.8,
        "error_type": None,
    }
    base.update(overrides)
    return RAGMetrics(**base)


def _hardened_result(
    *,
    retrieved: list[RetrievalResult] | None = None,
    latency_ms: float = 40.0,
    guard_reason: str = "",
) -> HardenedRagResult:
    if retrieved is None:
        retrieved = [RetrievalResult(payload={}, score=0.9)]
    return HardenedRagResult(
        guidance="therapy_prompt_high_confidence",
        advisory_score=0.8,
        retrieved=retrieved,
        grounded=True,
        guard_reason=guard_reason,
        latency_ms=latency_ms,
        cache_hit=False,
    )


class TestClassifyRagFailure:
    def test_timeout(self) -> None:
        assert classify_rag_failure(_metrics(error_type="timeout")) == RAGFailureType.TIMEOUT

    def test_empty_retrieval(self) -> None:
        assert (
            classify_rag_failure(_metrics(retrieved_docs=0, retrieval_score_max=0.0))
            == RAGFailureType.EMPTY_RETRIEVAL
        )

    def test_fallback_triggered(self) -> None:
        assert (
            classify_rag_failure(
                _metrics(retrieved_docs=2, fallback_used=True, retrieval_score_max=0.5)
            )
            == RAGFailureType.FALLBACK_TRIGGERED
        )

    def test_low_relevance(self) -> None:
        assert (
            classify_rag_failure(_metrics(retrieval_score_max=0.1))
            == RAGFailureType.LOW_RELEVANCE
        )

    def test_none_when_healthy(self) -> None:
        assert classify_rag_failure(_metrics()) is None


class TestDecideFallback:
    def test_heavy(self) -> None:
        assert (
            decide_fallback(_metrics(retrieved_docs=0, fallback_used=True))
            == RAGFallbackMode.HEAVY
        )

    def test_offline(self) -> None:
        assert decide_fallback(_metrics(retrieval_score_max=0.05)) == RAGFallbackMode.OFFLINE

    def test_light(self) -> None:
        assert decide_fallback(_metrics(fallback_used=True)) == RAGFallbackMode.LIGHT

    def test_none(self) -> None:
        assert decide_fallback(_metrics()) == RAGFallbackMode.NONE


class TestComputeWindow:
    def test_rates_and_p95(self) -> None:
        samples = [
            _metrics(latency_ms=10.0, fallback_used=False, retrieved_docs=2),
            _metrics(latency_ms=20.0, fallback_used=True, retrieved_docs=0),
            _metrics(latency_ms=100.0, fallback_used=False, retrieved_docs=1),
            _metrics(latency_ms=200.0, fallback_used=True, retrieved_docs=0),
        ]
        window = compute_window(samples)
        assert window.total_requests == 4
        assert window.fallback_rate == 0.5
        assert window.empty_rate == 0.5
        assert window.p95_latency_ms == 200.0


class TestEvaluateSlo:
    def test_ok(self) -> None:
        window = RAGWindowMetrics(
            total_requests=10,
            fallback_rate=0.05,
            empty_rate=0.02,
            avg_latency_ms=50.0,
            p95_latency_ms=100.0,
        )
        v = evaluate_slo(window, RAGSLO())
        assert not v.violated
        assert v.severity == "ok"

    def test_high_single_violation(self) -> None:
        window = RAGWindowMetrics(
            total_requests=10,
            fallback_rate=0.5,
            empty_rate=0.02,
            avg_latency_ms=50.0,
            p95_latency_ms=100.0,
        )
        v = evaluate_slo(window, RAGSLO())
        assert v.violated
        assert v.severity == "high"
        assert "fallback_rate" in v.violated_fields

    def test_critical_multiple_violations(self) -> None:
        window = RAGWindowMetrics(
            total_requests=10,
            fallback_rate=0.5,
            empty_rate=0.5,
            avg_latency_ms=200.0,
            p95_latency_ms=500.0,
        )
        v = evaluate_slo(window, RAGSLO())
        assert v.severity == "critical"
        assert len(v.violated_fields) >= 2


class TestMapViolationToAlert:
    def test_critical(self) -> None:
        v = RAGSLOViolation(True, ["a", "b"], "critical", "a;b")
        assert map_violation_to_alert(v) == RAGAlertLevel.CRITICAL

    def test_warning(self) -> None:
        v = RAGSLOViolation(True, ["latency_p95"], "high", "latency_p95")
        assert map_violation_to_alert(v) == RAGAlertLevel.WARNING


class TestRAGSLOLoad:
    def test_yaml_overrides_defaults(self) -> None:
        slo = RAGSLO.load(SLO_PATH)
        assert slo.retrieval_latency_ms_p95 == 120.0
        assert slo.fallback_rate_max == 0.10
        assert slo.window_max_requests == 100


@dataclass
class _StubPipeline:
    results: list[HardenedRagResult]

    def query(self, query: dict, temporal_reliability: float = 1.0) -> HardenedRagResult:
        result = self.results.pop(0)
        metrics = metrics_from_hardened_result(result, fallback_used=not result.retrieved)
        result.reliability = {"metrics": metrics.to_dict()}
        return result


class TestRAGReliabilityWrapper:
    def test_emits_alert_on_slo_violation(self) -> None:
        empty = _hardened_result(retrieved=[])
        stub = _StubPipeline([empty] * 20)
        wrapper = RAGReliabilityWrapper(
            stub,
            slo=RAGSLO(
                fallback_rate_max=0.01,
                empty_retrieval_rate_max=0.01,
                window_max_requests=50,
            ),
        )
        with patch("ml.retrieval.rag_reliability.emit_event") as mock_emit:
            for _ in range(20):
                wrapper.query({"k": "v"})
            alert_calls = [c for c in mock_emit.call_args_list if c.args and c.args[0] == "rag.alert"]
            assert alert_calls
            assert alert_calls[-1].kwargs["level"] in ("warning", "critical")

    def test_wrap_rag_pipeline_factory(self) -> None:
        stub = _StubPipeline([_hardened_result()])
        wrapped = wrap_rag_pipeline(stub, slo=RAGSLO(window_max_requests=10))
        assert isinstance(wrapped, RAGReliabilityWrapper)
        out = wrapped.query({})
        assert out.reliability is not None
        assert "fallback_mode" in out.reliability


class TestMetricsFromHardenedResult:
    def test_scores_from_retrieved(self) -> None:
        result = _hardened_result(
            retrieved=[
                RetrievalResult(payload={}, score=0.3),
                RetrievalResult(payload={}, score=0.9),
            ]
        )
        m = metrics_from_hardened_result(result)
        assert m.retrieval_score_max == 0.9
        assert m.retrieved_docs == 2
