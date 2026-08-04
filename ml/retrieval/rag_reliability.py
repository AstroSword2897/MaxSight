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


def compute_window(
    metrics: Sequence[RAGMetrics],
) -> RAGWindowMetrics:
    """Aggregate per-request metrics into window-level health signals."""
    if not metrics:
        return RAGWindowMetrics(
            total_requests=0,
            fallback_rate=0.0,
            empty_rate=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            retrieval_success_rate=1.0,
            hallucination_risk_score=0.0,
        )

    total = len(metrics)
    fallback_count = sum(1 for m in metrics if m.fallback_used)
    empty_count = sum(1 for m in metrics if m.retrieved_docs == 0)
    error_count = sum(1 for m in metrics if m.error_type is not None)
    guard_reject_count = sum(1 for m in metrics if m.guard_rejection)
    latencies = sorted(m.latency_ms for m in metrics)
    if latencies:
        p95_idx = min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))
        p95 = latencies[p95_idx]
    else:
        p95 = 0.0
    success_rate = 1.0 - (error_count + empty_count) / max(1, total)
    hallucination_risk = guard_reject_count / max(1, total)

    return RAGWindowMetrics(
        total_requests=total,
        fallback_rate=fallback_count / total,
        empty_rate=empty_count / total,
        avg_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_ms=p95,
        retrieval_success_rate=success_rate,
        hallucination_risk_score=hallucination_risk,
    )


def evaluate_slo(window: RAGWindowMetrics, slo: RAGSLO) -> RAGSLOViolation:
    """Compare window metrics to RAGSLO; severity escalates with violation count."""
    violations: list[str] = []

    if window.total_requests > 0 and window.p95_latency_ms > slo.retrieval_latency_ms_p95:
        violations.append("latency_p95")

    if window.total_requests > 0 and window.fallback_rate > slo.fallback_rate_max:
        violations.append("fallback_rate")

    if window.total_requests > 0 and window.empty_rate > slo.empty_retrieval_rate_max:
        violations.append("empty_retrieval_rate")

    if (
        window.total_requests > 0
        and window.retrieval_success_rate < slo.retrieval_success_rate_min
    ):
        violations.append("retrieval_success_rate")

    if (
        window.total_requests > 0
        and window.hallucination_risk_score > slo.hallucination_risk_score_max
    ):
        violations.append("hallucination_risk_score")

    if len(violations) >= 2:
        severity = "critical"
    elif len(violations) == 1:
        severity = "high"
    else:
        severity = "ok"

    return RAGSLOViolation(
        violated=len(violations) > 0,
        violated_fields=violations,
        severity=severity,
        explanation=";".join(violations) if violations else "ok",
    )


def map_violation_to_alert(violation: RAGSLOViolation) -> RAGAlertLevel:
    """Map SLO violation severity to operational alert level."""
    if violation.severity == "critical":
        return RAGAlertLevel.CRITICAL
    if violation.severity == "high":
        return RAGAlertLevel.WARNING
    return RAGAlertLevel.INFO


def emit_rag_failure(failure_type: RAGFailureType, metrics: RAGMetrics, **context: Any) -> None:
    """Emit structured rag.failure when classification is non-null."""
    emit_event(
        "rag.failure",
        failure_type=failure_type.value,
        latency_ms=round(metrics.latency_ms, 2),
        retrieved_docs=metrics.retrieved_docs,
        retrieval_score_max=round(metrics.retrieval_score_max, 4),
        **context,
    )


def emit_rag_alert(
    level: RAGAlertLevel,
    violation: RAGSLOViolation,
    context: dict[str, Any] | None = None,
) -> None:
    """Emit rag.alert for SLO violations with routing metadata."""
    emit_event(
        "rag.alert",
        system="rag",
        level=level.value,
        violations=list(violation.violated_fields),
        severity=violation.severity,
        explanation=violation.explanation,
        **(context or {}),
    )


@dataclass
class RAGReliabilityWrapper:
    """Observation + SLO + decision layer around HardenedRagPipeline."""

    base: Any
    slo: RAGSLO = field(default_factory=RAGSLO.load)
    _window: list[RAGMetrics] = field(default_factory=list)

    def query(self, query: dict[str, Any], temporal_reliability: float = 1.0) -> Any:
        """Delegate to base pipeline, classify failures, evaluate SLO, emit alerts."""
        result = self.base.query(query, temporal_reliability=temporal_reliability)
        reliability = dict(getattr(result, "reliability", None) or {})
        metrics_dict = reliability.get("metrics")
        if metrics_dict:
            metrics = RAGMetrics(**metrics_dict)
        else:
            metrics = metrics_from_hardened_result(result)

        failure = classify_rag_failure(
            metrics, low_relevance_threshold=self.slo.low_relevance_score
        )
        fallback_mode = decide_fallback(
            metrics, offline_threshold=self.slo.offline_relevance_score
        )

        if failure is not None:
            emit_rag_failure(failure, metrics, query_keys=list(query.keys())[:8])
            if failure == RAGFailureType.EMPTY_RETRIEVAL:
                emit_event(
                    "rag.degraded",
                    guard_reason=getattr(result, "guard_reason", "") or "empty_retrieval",
                    latency_ms=round(metrics.latency_ms, 2),
                )

        self._append_window(metrics)
        window_metrics = compute_window(self._window)
        violation = evaluate_slo(window_metrics, self.slo)

        if violation.violated:
            level = map_violation_to_alert(violation)
            emit_rag_alert(
                level,
                violation,
                {
                    "window_total": window_metrics.total_requests,
                    "fallback_rate": round(window_metrics.fallback_rate, 4),
                    "empty_rate": round(window_metrics.empty_rate, 4),
                    "p95_latency_ms": round(window_metrics.p95_latency_ms, 2),
                },
            )

        reliability.update(
            {
                "failure_type": failure.value if failure else "",
                "fallback_mode": fallback_mode.value,
                "metrics": metrics.to_dict(),
                "slo_violated": violation.violated,
                "slo_severity": violation.severity,
            }
        )
        result.reliability = reliability
        return result

    def _append_window(self, metrics: RAGMetrics) -> None:
        self._window.append(metrics)
        if len(self._window) > self.slo.window_max_requests:
            self._window.pop(0)

    @property
    def window_metrics(self) -> RAGWindowMetrics:
        """Expose current window aggregates for tests and diagnostics."""
        return compute_window(self._window)

    @property
    def slo_stats(self) -> Any:
        """Delegate legacy RagSloStats on the base pipeline when present."""
        return getattr(self.base, "slo", None)


def wrap_rag_pipeline(base: Any, slo: RAGSLO | None = None) -> RAGReliabilityWrapper:
