"""SageMaker-ready entrypoint for adaptive temporal preprocessing + advisory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.data.video_preprocessing import PanopticSegmenter, VideoPanopticPreprocessor
from ml.pipeline.rag_advisory import AdvisoryRetriever, generate_therapy_advisory
from ml.pipeline.sagemaker_config import SageMakerPipelineConfig
from ml.training.loss_weighting import build_temporal_weight_updates


@dataclass(frozen=True)
class PrecomputedVideoRecord:
    """Input record expected by this production entrypoint."""

    video_id: str
    frame_paths: List[str]
    frames_segments: List[List[Dict[str, Any]]]


class _PrecomputedSegmenter(PanopticSegmenter):
    """Segmenter adapter that serves precomputed pseudo-panoptic segments."""

    def __init__(self, by_path: Dict[str, List[Dict[str, Any]]]):
        self.by_path = by_path

    def segment(self, frame: Any) -> List[Dict[str, Any]]:
        key = str(frame)
        return list(self.by_path.get(key, []))


def _load_records(input_dir: Path) -> List[PrecomputedVideoRecord]:
    manifest_path = input_dir / "video_records.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing required input manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    records: List[PrecomputedVideoRecord] = []
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
    cfg: Optional[SageMakerPipelineConfig] = None,
    retriever: Optional[AdvisoryRetriever] = None,
) -> Dict[str, Any]:
    """Execute production pipeline in SageMaker-compatible environment."""

    config = cfg or SageMakerPipelineConfig.from_env()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(config.input_dir)
    all_outputs: List[Dict[str, Any]] = []

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

        clips_out: List[Dict[str, Any]] = []
        for clip in processed["clips"]:
            segments = clip.get("frames_segments", [])
            frame_count = max(1, len(segments))
            temporal_reliability = max(
                0.0,
                min(1.0, 1.0 - (sum(1 for s in segments if not s) / float(frame_count))),
            )
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


if __name__ == "__main__":
    result = run_sagemaker_pipeline()
    print(json.dumps(result))

