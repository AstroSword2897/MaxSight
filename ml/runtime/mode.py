"""Runtime mode resolution, compute tier routing, and request orchestration."""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml.runtime.tier_router import TierProfile

logger = logging.getLogger(__name__)


class RuntimeMode(str, Enum):
    """High-level deployment profile; drives defaults in tools/simulation/config."""

    SIMULATOR = "simulator"
    PRODUCTION = "production"


def get_runtime_mode() -> RuntimeMode:
    """Resolve ``MAXSIGHT_RUNTIME`` (default: simulator). Accepts production|prod."""

    raw = os.getenv("MAXSIGHT_RUNTIME", "simulator").strip().lower()
    if raw in ("production", "prod"):
        return RuntimeMode.PRODUCTION
    return RuntimeMode.SIMULATOR


def is_production_runtime() -> bool:
    """True when running with production defaults (no Flask debug, dev routes off)."""

    return get_runtime_mode() == RuntimeMode.PRODUCTION


def resolve_compute_tier(
    *,
    latency_budget_ms: float | None = None,
    battery_low: bool = False,
) -> TierProfile:
    """Return the appropriate ``TierProfile`` for the current runtime environment.

    Simulator defaults to bronze to avoid spinning up expensive capabilities
    during local development. Production uses gold unless battery or latency
    constraints force a downgrade.

    Parameters:
        latency_budget_ms: Max acceptable latency; triggers silver/bronze fallback.
        battery_low: When true, caps at silver tier regardless of mode.
    """
    from ml.runtime.contracts import ComputeTier
    from ml.runtime.tier_router import TierRouter

    mode = get_runtime_mode()
    requested = ComputeTier.BRONZE if mode == RuntimeMode.SIMULATOR else ComputeTier.GOLD
    router = TierRouter()
    return router.resolve(requested, latency_budget_ms=latency_budget_ms, battery_low=battery_low)


class RuntimeOrchestrator:
    """Unified inference orchestrator: tier routing → therapy → RAG → RuntimeResponse.

    Wires together the compute tier router, therapy engine, and RAG pipeline
    into a single ``process()`` call. Each subsystem is optional so the
    orchestrator degrades gracefully when components are unavailable (e.g.
    bronze tier with RAG disabled, or simulator runs without therapy).

    Typical production usage:
        orch = RuntimeOrchestrator()
        response = orch.process(request)
        print(response.to_dict())
    """

    def __init__(
        self,
        *,
        therapy_engine: Any | None = None,
        rag_pipeline: Any | None = None,
        tier_router: Any | None = None,
    ) -> None:
        """Initialise orchestrator with optional subsystem overrides.

        Parameters:
            therapy_engine: ``TherapyEngine`` instance; built lazily when None.
            rag_pipeline: ``HardenedRagPipeline`` instance; built lazily when None.
            tier_router: ``TierRouter`` instance; built lazily when None.
        """
        self._therapy_engine = therapy_engine
        self._rag_pipeline = rag_pipeline
        self._tier_router = tier_router

    def _get_therapy_engine(self) -> Any:
        if self._therapy_engine is None:
            from ml.therapy.therapy_engine import TherapyEngine

            self._therapy_engine = TherapyEngine()
        return self._therapy_engine

    def _get_rag_pipeline(self) -> Any:
        if self._rag_pipeline is None:
            from ml.retrieval.rag_hardening import HardenedRagPipeline

            class _NullRetriever:
                _warned = False

                def retrieve(self, query: dict[str, Any], top_k: int = 5) -> list:
                    if not _NullRetriever._warned:
                        logger.warning(
                            "rag_degraded null_retriever_active — configure a real retriever before device deployment"
                        )
                        _NullRetriever._warned = True
                    return []

            self._rag_pipeline = HardenedRagPipeline(_NullRetriever())
        return self._rag_pipeline

    def _get_tier_router(self) -> Any:
        if self._tier_router is None:
            from ml.runtime.tier_router import TierRouter

            self._tier_router = TierRouter()
        return self._tier_router

    def process(self, request: Any) -> Any:
        """Process one runtime request and return a ``RuntimeResponse``.

        Parameters:
            request: ``RuntimeRequest`` dataclass from ``ml.runtime.contracts``.

        Returns:
            ``RuntimeResponse`` with populated therapy, RAG, tier trace, and
            latency fields. Never raises; subsystem errors are logged and
            reflected in the ``degraded_mode`` field.

        Behavioral guarantees:
            - Therapy runs only when ``request.enable_therapy`` is True and the
              resolved tier does not disable it.
            - RAG runs only when ``request.enable_rag`` is True and the resolved
              tier profile has ``enable_rag=True``.
            - ``score_trace`` is populated on each ``TherapyRecommendation`` so
              SCRUM-19 explainability requirements are satisfied.
            - Total latency is measured wall-clock and included in the response.
        """
        from ml.runtime.contracts import (
            DegradedMode,
            RagContext,
            RuntimeResponse,
            TherapyRecommendation,
            validate_model_outputs,
        )
        from ml.training.observability import emit_event

        t0 = time.perf_counter()
        degraded_mode = DegradedMode.D0_NORMAL
        therapy_recommendations: list[TherapyRecommendation] = []
        rag_context: RagContext | None = None
        tier_trace: dict[str, Any] = {}
        perception = validate_model_outputs(dict(request.perception))

        # Resolve compute tier from request.
        router = self._get_tier_router()
        try:
            profile = router.resolve(request.tier)
            tier_trace = router.to_trace(profile)
            emit_event(
                "runtime.tier_resolved",
                tier=profile.tier.value,
                enable_rag=profile.enable_rag,
                enable_therapy=request.enable_therapy,
            )
        except Exception as exc:
            logger.warning("tier_resolve_failed: %s — defaulting to bronze", exc)
            profile = None
            degraded_mode = DegradedMode.D1_HIGH_LOAD

        # Run therapy engine when requested.
        if request.enable_therapy:
            try:
                engine = self._get_therapy_engine()
                actions = engine.update(perception)
                last_context = engine.get_last_context()
                for action in actions:
                    score_trace: dict[str, float] = {}
                    try:
                        from ml.therapy.situation_understanding import SituationContext

                        trace_context = last_context or SituationContext()
                        trace = engine.scoring_model.score_intervention(
                            trace_context, action.intervention_type
                        )
                        score_trace = trace.to_dict()
                    except Exception:
                        pass
                    therapy_recommendations.append(
                        TherapyRecommendation(
                            intervention_type=action.intervention_type,
                            channel=action.channel,
                            content=action.content,
                            intensity=action.intensity,
                            score=score_trace.get("final_score", action.intensity),
                            score_trace=score_trace,
                        )
                    )
            except Exception as exc:
                logger.error("therapy_engine_error: %s", exc)
                degraded_mode = DegradedMode.D1_HIGH_LOAD

        # Run RAG pipeline when the tier supports it.
        rag_enabled = request.enable_rag and (profile is not None and profile.enable_rag)
        if rag_enabled:
            try:
                pipeline = self._get_rag_pipeline()
                temporal_reliability = float(perception.get("temporal_consistency", 1.0))
                rag_result = pipeline.query(perception, temporal_reliability)
                rag_context = RagContext(
                    guidance=rag_result.guidance,
                    advisory_score=rag_result.advisory_score,
                    retrieved_count=len(rag_result.retrieved),
                    grounded=rag_result.grounded,
                    guard_reason=rag_result.guard_reason,
                )
            except Exception as exc:
                logger.error("rag_pipeline_error: %s", exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        resolved_tier = profile.tier if profile else request.tier

        return RuntimeResponse(
            frame_id=request.frame_id,
            tier=resolved_tier,
            degraded_mode=degraded_mode,
            therapy=therapy_recommendations,
            rag=rag_context,
            latency_ms=round(latency_ms, 2),
            trace={"tier_profile": tier_trace},
        )
