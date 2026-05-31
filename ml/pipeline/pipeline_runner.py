"""Core offline pipeline logic: temporal preprocessing + advisory generation.

This module is importable by tests and other modules. The SageMaker Processing
entrypoint (ml/pipeline/sagemaker_entrypoint.py) calls run_sagemaker_pipeline
from here — do not import the entrypoint itself, only this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.data.video_preprocessing import PanopticSegmenter, VideoPanopticPreprocessor
from ml.pipeline.rag_advisory import AdvisoryRetriever, generate_therapy_advisory
from ml.pipeline.sagemaker_config import SageMakerPipelineConfig
from ml.training.loss_weighting import build_temporal_weight_updates

try:
    from ml.retrieval.rag_hardening import HardenedRagPipeline
    from ml.retrieval.rag_hardening import RetrievalResult as HardenedRetrievalResult

    _HARDENED_RAG_AVAILABLE = True
except ImportError:
    _HARDENED_RAG_AVAILABLE = False


def _make_hardened_retriever(base_retriever: AdvisoryRetriever | None) -> Any | None:
    """Wrap an AdvisoryRetriever in HardenedRagPipeline for production use.

    Returns ``None`` when rag_hardening is unavailable, so callers fall back
    to the lightweight advisory path.
    """
    if not _HARDENED_RAG_AVAILABLE or base_retriever is None:
        return None

    class _AdaptedRetriever:
        """Adapts AdvisoryRetriever to the HardenedRagPipeline retriever protocol."""

        def __init__(self, inner: AdvisoryRetriever) -> None:
            self._inner = inner

        def retrieve(self, query: dict[str, Any], top_k: int = 5):
            results = self._inner.retrieve(query, top_k=top_k)
            return [HardenedRetrievalResult(payload=r.payload, score=r.score) for r in results]

    return HardenedRagPipeline(_AdaptedRetriever(base_retriever))


@dataclass(frozen=True)
class PrecomputedVideoRecord:
    """Input record expected by the offline processing pipeline."""

    video_id: str
    frame_paths: list[str]
    frames_segments: list[list[dict[str, Any]]]


class _PrecomputedSegmenter(PanopticSegmenter):
    """Segmenter adapter that serves precomputed pseudo-panoptic segments."""

    def __init__(self, by_path: dict[str, list[dict[str, Any]]]):
        self.by_path = by_path

    def segment(self, frame: Any) -> list[dict[str, Any]]:
        key = str(frame)
        return list(self.by_path.get(key, []))


def _load_records(input_dir: Path) -> list[PrecomputedVideoRecord]:
    manifest_path = input_dir / "video_records.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing required input manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    records: list[PrecomputedVideoRecord] = []
    for row in data:
        records.append(
            PrecomputedVideoRecord(
                video_id=str(row["video_id"]),
                frame_paths=[str(p) for p in row["frame_paths"]],
                frames_segments=[[dict(s) for s in frame] for frame in row["frames_segments"]],
            )
        )
    return records


def run_sagemaker_pipeline(
    cfg: SageMakerPipelineConfig | None = None,
    retriever: AdvisoryRetriever | None = None,
) -> dict[str, Any]:
    """Execute the offline preprocessing pipeline in a SageMaker-compatible environment."""

    config = cfg or SageMakerPipelineConfig.from_env()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(config.input_dir)
    all_outputs: list[dict[str, Any]] = []

    # Prefer hardened RAG pipeline for production resilience.
    hardened = _make_hardened_retriever(retriever)

    for rec in records:
        by_path = {
            p: rec.frames_segments[i]
            for i, p in enumerate(rec.frame_paths)
            if i < len(rec.frames_segments)
        }
        preprocessor = VideoPanopticPreprocessor(
            segmenter=_PrecomputedSegmenter(by_path),
            frame_loader=lambda p: p,
            config=config.preprocessing,
        )
        processed = preprocessor.process_video(rec.video_id, rec.frame_paths)

        clips_out: list[dict[str, Any]] = []
        for clip in processed["clips"]:
            segments = clip.get("frames_segments", [])
            frame_count = max(1, len(segments))
            temporal_reliability = max(
                0.0,
                min(1.0, 1.0 - (sum(1 for s in segments if not s) / float(frame_count))),
            )
            if hardened is not None:
                # Use the industrial RAG path with hallucination guard + SLO.
                rag_result = hardened.query(
                    {"clip_id": clip.get("clip_id"), "temporal_reliability": temporal_reliability},
                    temporal_reliability=temporal_reliability,
                )
                advisory = rag_result.to_dict()
                advisory["temporal_reliability"] = temporal_reliability
            else:
                advisory = generate_therapy_advisory(
                    clip_manifest=clip,
                    temporal_reliability=temporal_reliability,
                    retriever=retriever,
                    top_k=config.advisory_retrieval_top_k,
                )
            clip_weight_updates = build_temporal_weight_updates(
                epoch=0,
                schedule=config.temporal_weight_schedule,
                temporal_heads={"motion": 1.0, "temporal_consistency": 1.0},
            )
            clip_enriched = dict(clip)
            clip_enriched["temporal_reliability"] = temporal_reliability
            clip_enriched["advisory"] = advisory
            clip_enriched["temporal_weight_updates"] = clip_weight_updates
            clips_out.append(clip_enriched)

        all_outputs.append(
            {
                "video_id": rec.video_id,
                "stats": processed["stats"],
                "clips": clips_out,
            }
        )

    out_path = config.output_dir / "phase3_pipeline_output.json"
    out_path.write_text(json.dumps(all_outputs, indent=2))
    return {"output_path": str(out_path), "num_videos": len(all_outputs)}
