"""Non-blocking advisory logic for retrieval-augmented therapy guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """Compact retrieval result used by advisory generation."""

    payload: Dict[str, Any]
    score: float


class AdvisoryRetriever(Protocol):
    """Retriever contract for production advisory enrichment."""

    def retrieve(self, query: Dict[str, Any], top_k: int = 5) -> List[RetrievalResult]:
        """Return ranked retrieval results without blocking critical path."""


def generate_therapy_advisory(
    clip_manifest: Dict[str, Any],
    temporal_reliability: float,
    retriever: Optional[AdvisoryRetriever],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Create advisory payload from temporal reliability and retrieval context.

    This method is intentionally advisory-only and does not gate core hazards.
    """
    reliability = max(0.0, min(1.0, float(temporal_reliability)))
    query = {
        "video_id": clip_manifest.get("video_id"),
        "clip_id": clip_manifest.get("clip_id"),
        "temporal_window": clip_manifest.get("temporal_window"),
        "segments_count": sum(len(s) for s in clip_manifest.get("frames_segments", [])),
        "temporal_reliability": reliability,
    }

    retrieved: List[RetrievalResult] = []
    if retriever is not None:
        try:
            retrieved = retriever.retrieve(query, top_k=top_k)
        except Exception:
            retrieved = []

    retrieval_score = max((r.score for r in retrieved), default=0.0)
    advisory_score = reliability * retrieval_score if retrieved else reliability
    if reliability < 0.45:
        guidance = "advisory_only_unstable_perception"
    elif advisory_score > 0.7:
        guidance = "therapy_prompt_high_confidence"
    else:
        guidance = "therapy_prompt_low_intensity"

    return {
        "guidance": guidance,
        "temporal_reliability": reliability,
        "advisory_score": float(advisory_score),
        "retrieved_count": len(retrieved),
        "retrieved": [
            {"payload": r.payload, "score": float(r.score)}
            for r in retrieved
        ],
    }

