"""RAG reliability: SLO contracts, failure taxonomy, window evaluation, and alerting."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import yaml

from ml.training.observability import emit_event

DEFAULT_SLO_PATH = Path(__file__).resolve().parents[1] / "config" / "rag_slo.yaml"
DEFAULT_WINDOW_SIZE = 100
LOW_RELEVANCE_THRESHOLD = 0.2
OFFLINE_RELEVANCE_THRESHOLD = 0.1


class RAGFailureType(str, Enum):
    """Structured RAG failure reasons for observability and RagContext."""

    EMPTY_RETRIEVAL = "empty_retrieval"
    TIMEOUT = "timeout"
    EMBEDDING_FAILURE = "embedding_failure"
    VECTOR_DB_ERROR = "vector_db_error"
    FALLBACK_TRIGGERED = "fallback_triggered"
    LOW_RELEVANCE = "low_relevance"


class RAGAlertLevel(str, Enum):
    """Alert severity routed to rag.alert events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RAGFallbackMode(str, Enum):
    """Stratified fallback tiers beyond binary degraded flags."""

    NONE = "none"
    LIGHT = "light_fallback"
    HEAVY = "heavy_fallback"
    OFFLINE = "offline_mode"


@dataclass(frozen=True)
class RAGSLO:
    """Hard reliability contract for rolling-window RAG health."""

    retrieval_latency_ms_p95: float = 120.0
    retrieval_success_rate_min: float = 0.95
    fallback_rate_max: float = 0.10
    empty_retrieval_rate_max: float = 0.05
    hallucination_risk_score_max: float = 0.2
    window_max_requests: int = DEFAULT_WINDOW_SIZE
    low_relevance_score: float = LOW_RELEVANCE_THRESHOLD
    offline_relevance_score: float = OFFLINE_RELEVANCE_THRESHOLD

    @classmethod
    def load(cls, path: Path | None = None) -> RAGSLO:
        """Parse RAG SLO YAML; missing keys fall back to dataclass defaults."""
        p = path or DEFAULT_SLO_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        slo = raw.get("slo", {})
        window = raw.get("window", {})
        thresholds = raw.get("thresholds", {})
        return cls(
            retrieval_latency_ms_p95=float(
                slo.get("retrieval_latency_ms_p95", cls.retrieval_latency_ms_p95)
            ),
            retrieval_success_rate_min=float(
                slo.get("retrieval_success_rate_min", cls.retrieval_success_rate_min)
            ),
            fallback_rate_max=float(slo.get("fallback_rate_max", cls.fallback_rate_max)),
            empty_retrieval_rate_max=float(
                slo.get("empty_retrieval_rate_max", cls.empty_retrieval_rate_max)
            ),
            hallucination_risk_score_max=float(
                slo.get("hallucination_risk_score_max", cls.hallucination_risk_score_max)
            ),
            window_max_requests=int(window.get("max_requests", cls.window_max_requests)),
            low_relevance_score=float(
                thresholds.get("low_relevance_score", cls.low_relevance_score)
            ),
            offline_relevance_score=float(
                thresholds.get("offline_relevance_score", cls.offline_relevance_score)
            ),
        )


@dataclass
class RAGMetrics:
    """Per-request RAG observation signals."""

    latency_ms: float
    retrieved_docs: int
    fallback_used: bool
    query_embedding_norm: float
    retrieval_score_max: float
    error_type: str | None = None
    grounded: bool = True
    guard_rejection: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGWindowMetrics:
    """Aggregated RAG health over a rolling request window."""

    total_requests: int
    fallback_rate: float
    empty_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    retrieval_success_rate: float = 1.0
    hallucination_risk_score: float = 0.0


@dataclass
class RAGSLOViolation:
    """Result of comparing window metrics against RAGSLO."""

    violated: bool
    violated_fields: list[str]
    severity: str
    explanation: str


def metrics_from_hardened_result(
    result: Any,
    *,
    fallback_used: bool = False,
    error_type: str | None = None,
    query_embedding_norm: float = 0.0,
) -> RAGMetrics:
    """Build RAGMetrics from a HardenedRagResult and pipeline context."""
    retrieved = getattr(result, "retrieved", []) or []
    scores = [float(r.score) for r in retrieved if getattr(r, "score", None) is not None]
    retrieval_score_max = max(scores, default=0.0)
    guard_reason = str(getattr(result, "guard_reason", "") or "")
    guard_rejection = guard_reason.startswith("hallucination_guard")
    return RAGMetrics(
        latency_ms=float(getattr(result, "latency_ms", 0.0)),
        retrieved_docs=len(retrieved),
        fallback_used=fallback_used,
        query_embedding_norm=query_embedding_norm,
        retrieval_score_max=retrieval_score_max,
        error_type=error_type,
        grounded=bool(getattr(result, "grounded", False)),
        guard_rejection=guard_rejection,
    )


def classify_rag_failure(
    metrics: RAGMetrics,
    *,
    low_relevance_threshold: float = LOW_RELEVANCE_THRESHOLD,
) -> RAGFailureType | None:
    """Map raw per-request metrics to a structured failure type."""
    if metrics.error_type == "timeout":
        return RAGFailureType.TIMEOUT
    if metrics.error_type == "embedding_failure":
        return RAGFailureType.EMBEDDING_FAILURE
    if metrics.error_type in ("vector_db_error", "retriever_error"):
        return RAGFailureType.VECTOR_DB_ERROR

    if metrics.retrieved_docs == 0:
        return RAGFailureType.EMPTY_RETRIEVAL

    if metrics.fallback_used:
        return RAGFailureType.FALLBACK_TRIGGERED

    if metrics.retrieval_score_max < low_relevance_threshold:
        return RAGFailureType.LOW_RELEVANCE

    return None


def decide_fallback(
    metrics: RAGMetrics,
    *,
    offline_threshold: float = OFFLINE_RELEVANCE_THRESHOLD,
) -> RAGFallbackMode:
    """Choose fallback tier from per-request metrics."""
    if metrics.retrieved_docs == 0 and metrics.fallback_used:
        return RAGFallbackMode.HEAVY

    if metrics.retrieval_score_max < offline_threshold:
        return RAGFallbackMode.OFFLINE

    if metrics.fallback_used:
        return RAGFallbackMode.LIGHT

    return RAGFallbackMode.NONE
