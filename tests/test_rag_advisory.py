"""Strict tests for ml.pipeline.rag_advisory (RAG as advisory layer before therapy copy policy)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.rag_advisory import (  # noqa: E402
    RetrievalResult,
    generate_therapy_advisory,
)


def _clip(video_id: str = "v1", clip_id: str = "c1", segments=None) -> dict:
    if segments is None:
        segments = [[{"bbox": [0, 0, 10, 10]}], [{"bbox": [1, 0, 10, 10]}]]
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "temporal_window": 8,
        "frames_segments": segments,
    }


def test_generate_therapy_advisory_unstable_perception_low_reliability() -> None:
    out = generate_therapy_advisory(_clip(), temporal_reliability=0.2, retriever=None)
    assert out["guidance"] == "advisory_only_unstable_perception"
    assert out["temporal_reliability"] == pytest.approx(0.2)
    assert out["retrieved_count"] == 0
    assert out["advisory_score"] == pytest.approx(0.2)


def test_generate_therapy_advisory_clamps_reliability() -> None:
    out = generate_therapy_advisory(_clip(), temporal_reliability=-1.0, retriever=None)
    assert out["temporal_reliability"] == 0.0
    out2 = generate_therapy_advisory(_clip(), temporal_reliability=2.0, retriever=None)
    assert out2["temporal_reliability"] == 1.0


def test_generate_therapy_advisory_no_retriever_uses_reliability_only() -> None:
    out = generate_therapy_advisory(_clip(), temporal_reliability=0.9, retriever=None)
    assert out["guidance"] == "therapy_prompt_high_confidence"
    assert out["advisory_score"] == pytest.approx(0.9)
    assert out["retrieved"] == []

    low = generate_therapy_advisory(_clip(), temporal_reliability=0.5, retriever=None)
    assert low["guidance"] == "therapy_prompt_low_intensity"
    assert low["advisory_score"] == pytest.approx(0.5)


class _FakeRetriever:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def retrieve(self, query: dict, top_k: int = 5):  # noqa: ARG002
        assert "temporal_reliability" in query
        assert query["video_id"] == "v1"
        assert query["clip_id"] == "c1"
        return [
            RetrievalResult(payload={"k": i}, score=s)
            for i, s in enumerate(self.scores[:top_k])
        ]


def test_generate_therapy_advisory_high_confidence_with_retriever() -> None:
    ret = _FakeRetriever([0.95, 0.1])
    out = generate_therapy_advisory(_clip(), temporal_reliability=0.95, retriever=ret, top_k=5)
    assert out["retrieved_count"] == 2
    assert out["guidance"] == "therapy_prompt_high_confidence"
    assert out["advisory_score"] == pytest.approx(0.95 * 0.95, rel=1e-5)
    assert len(out["retrieved"]) == 2
    assert out["retrieved"][0]["score"] == pytest.approx(0.95)


def test_generate_therapy_advisory_retriever_exception_falls_back() -> None:
    class Boom:
        def retrieve(self, query, top_k=5):  # noqa: ARG002
            raise RuntimeError("index offline")

    out = generate_therapy_advisory(_clip(), temporal_reliability=0.55, retriever=Boom())
    assert out["retrieved_count"] == 0
    assert out["guidance"] == "therapy_prompt_low_intensity"
    assert out["advisory_score"] == pytest.approx(0.55)


def test_segments_count_in_query_matches_manifest() -> None:
    captured: dict = {}

    class CaptureRetriever:
        def retrieve(self, query, top_k=5):  # noqa: ARG002
            captured.update(query)
            return []

    generate_therapy_advisory(_clip(segments=[[], [{"a": 1}], [{"b": 1}]]), 0.6, CaptureRetriever())
    assert captured["segments_count"] == 2
