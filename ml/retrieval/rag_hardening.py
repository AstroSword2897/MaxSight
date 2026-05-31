"""Industrial RAG hardening: hallucination guard, query debouncing, SLO monitoring."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HALLUCINATION_THRESHOLD = 0.45
DEFAULT_DEBOUNCE_S = 0.1
DEFAULT_TIMEOUT_S = 0.5


@dataclass
class RetrievalResult:
    """Single retrieved document with groundedness score."""

    payload: dict[str, Any]
    score: float
    grounded: bool = True
    source_id: str = ""


@dataclass
class HardenedRagResult:
    """Full RAG response with guard annotations."""

    guidance: str
    advisory_score: float
    retrieved: list[RetrievalResult]
    grounded: bool
    guard_reason: str
    latency_ms: float
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "guidance": self.guidance,
            "advisory_score": self.advisory_score,
            "retrieved_count": len(self.retrieved),
            "grounded": self.grounded,
            "guard_reason": self.guard_reason,
            "latency_ms": round(self.latency_ms, 2),
            "cache_hit": self.cache_hit,
        }


@dataclass
class RagSloStats:
    """Running SLO compliance counters."""

    total: int = 0
    within_latency: int = 0
    grounded: int = 0
    cache_hits: int = 0
    slo_latency_ms: float = DEFAULT_TIMEOUT_S * 1000

    @property
    def latency_slo_rate(self) -> float:
        return self.within_latency / max(1, self.total)

    @property
    def groundedness_rate(self) -> float:
        return self.grounded / max(1, self.total)

    def record(self, result: HardenedRagResult) -> None:
        self.total += 1
        if result.latency_ms <= self.slo_latency_ms:
            self.within_latency += 1
        if result.grounded:
            self.grounded += 1
        if result.cache_hit:
            self.cache_hits += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "latency_slo_rate": round(self.latency_slo_rate, 4),
            "groundedness_rate": round(self.groundedness_rate, 4),
            "cache_hits": self.cache_hits,
            "slo_latency_ms": self.slo_latency_ms,
        }


class HallucinationGuard:
    """Reject retrieved results that lack groundedness evidence."""

    def __init__(self, threshold: float = DEFAULT_HALLUCINATION_THRESHOLD) -> None:
        self.threshold = threshold

    def filter(self, results: list[RetrievalResult]) -> tuple[list[RetrievalResult], str]:
        """Return (grounded_results, guard_reason)."""
        grounded = [r for r in results if r.score >= self.threshold and r.grounded]
        if not grounded and results:
            return [], "hallucination_guard_all_rejected"
        if len(grounded) < len(results):
            return grounded, "hallucination_guard_partial_rejection"
        return grounded, ""


class QueryDebouncer:
    """Suppress duplicate RAG queries within a short window."""

    def __init__(self, debounce_s: float = DEFAULT_DEBOUNCE_S) -> None:
        self.debounce_s = debounce_s
        self._cache: dict[str, tuple[float, HardenedRagResult]] = {}

    def _key(self, query: dict[str, Any]) -> str:
        raw = str(sorted(query.items()))
        return hashlib.sha1(raw.encode()).hexdigest()

    def get(self, query: dict[str, Any]) -> HardenedRagResult | None:
        key = self._key(query)
        if key in self._cache:
            ts, result = self._cache[key]
            if time.monotonic() - ts < self.debounce_s:
                return result
        return None

    def store(self, query: dict[str, Any], result: HardenedRagResult) -> None:
        self._cache[self._key(query)] = (time.monotonic(), result)


class HardenedRagPipeline:
    """Production RAG: debounce → retrieve → guard → respond with SLO tracking."""

    def __init__(
        self,
        retriever: Any,
        *,
        hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
        debounce_s: float = DEFAULT_DEBOUNCE_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.guard = HallucinationGuard(threshold=hallucination_threshold)
        self.debouncer = QueryDebouncer(debounce_s=debounce_s)
        self.timeout_s = timeout_s
        self.top_k = top_k
        self.slo = RagSloStats(slo_latency_ms=timeout_s * 1000)

    def query(self, query: dict[str, Any], temporal_reliability: float = 1.0) -> HardenedRagResult:
        """Execute hardened RAG with debounce, guard, timeout, and SLO tracking."""
        cached = self.debouncer.get(query)
        if cached is not None:
            return HardenedRagResult(
                guidance=cached.guidance,
                advisory_score=cached.advisory_score,
                retrieved=cached.retrieved,
                grounded=cached.grounded,
                guard_reason=cached.guard_reason,
                latency_ms=0.0,
                cache_hit=True,
            )

        t0 = time.perf_counter()
        raw_results: list[RetrievalResult] = []
        try:
            raw_results = self.retriever.retrieve(query, top_k=self.top_k)
        except Exception as exc:
            logger.error("rag_retriever_error: %s", exc)

        filtered, guard_reason = self.guard.filter(raw_results)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not raw_results:
            from ml.training.observability import emit_event

            emit_event(
                "rag.degraded",
                guard_reason=guard_reason or "no_results",
                latency_ms=round(elapsed_ms, 2),
            )

        reliability = max(0.0, min(1.0, float(temporal_reliability)))
        top_score = max((r.score for r in filtered), default=0.0)
        advisory_score = reliability * top_score if filtered else reliability * 0.3

        if reliability < 0.45:
            guidance = "advisory_only_unstable_perception"
        elif advisory_score > 0.7:
            guidance = "therapy_prompt_high_confidence"
        else:
            guidance = "therapy_prompt_low_intensity"

        result = HardenedRagResult(
            guidance=guidance,
            advisory_score=float(advisory_score),
            retrieved=filtered,
            grounded=len(filtered) > 0 and not guard_reason.startswith("hallucination_guard_all"),
            guard_reason=guard_reason,
            latency_ms=elapsed_ms,
            cache_hit=False,
        )
        self.debouncer.store(query, result)
        self.slo.record(result)
        return result
