"""Production video preprocessing pipeline for panoptic temporal training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from ml.data.video_panoptic import (
    AdaptiveTemporalConfig,
    PseudoPanopticQualityConfig,
    VideoSamplingConfig,
    associate_tracks_multi_frame,
    build_adaptive_windows,
    build_fixed_stride_windows,
    iter_chunks,
    prune_pseudo_segments,
)


class PanopticSegmenter(Protocol):
    """Interface for panoptic segmentation backends."""

    def segment(self, frame: Any) -> list[dict[str, Any]]:
        """Return pseudo-panoptic segments for one frame."""
        ...


@dataclass(frozen=True)
class PreprocessingConfig:
    """Pipeline configuration for scalable open-video preprocessing."""

    sampling: VideoSamplingConfig = VideoSamplingConfig()
    quality: PseudoPanopticQualityConfig = PseudoPanopticQualityConfig()
    chunk_size: int = 64
    segmentation_workers: int = 4
    temporal_lookback: int = 2
    temporal_iou_threshold: float = 0.3
    enable_frame_jitter: bool = False
    enable_speed_perturbation: bool = False
    enable_adaptive_windowing: bool = False
    adaptive: AdaptiveTemporalConfig = AdaptiveTemporalConfig()

    def validate(self) -> None:
        self.sampling.validate()
        self.quality.validate()
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.segmentation_workers < 1:
            raise ValueError("segmentation_workers must be >= 1")
        if self.temporal_lookback < 1:
            raise ValueError("temporal_lookback must be >= 1")
        if not (0.0 <= self.temporal_iou_threshold <= 1.0):
            raise ValueError("temporal_iou_threshold must be in [0, 1]")
        if self.enable_adaptive_windowing:
            self.adaptive.validate()


class VideoPanopticPreprocessor:
    """Chunked and parallel panoptic preprocessing for production use."""

    def __init__(
        self,
        segmenter: PanopticSegmenter,
        frame_loader: Callable[[str], Any],
        config: PreprocessingConfig | None = None,
    ):
        self.segmenter = segmenter
        self.frame_loader = frame_loader
        self.config = config or PreprocessingConfig()
        self.config.validate()

    def _segment_frame_path(self, frame_path: str) -> list[dict[str, Any]]:
        frame = self.frame_loader(frame_path)
        segments = self.segmenter.segment(frame)
        return prune_pseudo_segments(segments, self.config.quality)

    def process_video(
        self,
        video_id: str,
        frame_paths: Sequence[str],
    ) -> dict[str, Any]:
        """Build sequence-ready clip manifest with pseudo-panoptic supervision."""

        if not video_id:
            raise ValueError("video_id is required")
        if not frame_paths:
            return {
                "video_id": video_id,
                "clips": [],
                "stats": {"frames_total": 0, "frames_kept": 0, "segments_kept": 0},
            }

        # Parallel segmentation with chunked submission to avoid unbounded memory.
        per_frame_segments: list[list[dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=self.config.segmentation_workers) as executor:
            for frame_chunk in iter_chunks(list(frame_paths), self.config.chunk_size):
                results = list(executor.map(self._segment_frame_path, frame_chunk))
                per_frame_segments.extend(results)

        associated = associate_tracks_multi_frame(
            per_frame_segments,
            lookback=self.config.temporal_lookback,
            iou_threshold=self.config.temporal_iou_threshold,
        )

        if self.config.enable_adaptive_windowing:
            windows = build_adaptive_windows(associated, self.config.adaptive)
        else:
            windows = build_fixed_stride_windows(len(frame_paths), self.config.sampling)
        clips: list[dict[str, Any]] = []
        for clip_idx, (start, end) in enumerate(windows):
            clip_frames = list(frame_paths[start:end])
            clip_segments = associated[start:end]
            span = int(end - start)
            clips.append(
                {
                    "clip_id": f"{video_id}_clip_{clip_idx:06d}",
                    "video_id": video_id,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    # Actual clip length (matches len(frame_paths)); may differ from sampling.temporal_window when adaptive.
                    "temporal_window": span,
                    "temporal_stride": int(self.config.sampling.temporal_stride),
                    "temporal_overlap": int(self.config.sampling.temporal_overlap),
                    "frame_paths": clip_frames,
                    "frames_segments": clip_segments,
                    "augmentation_policy": {
                        "frame_jitter": bool(self.config.enable_frame_jitter),
                        "speed_perturbation": bool(self.config.enable_speed_perturbation),
                    },
                }
            )

        total_segments = sum(len(s) for s in associated)
        return {
            "video_id": video_id,
            "clips": clips,
            "stats": {
                "frames_total": len(frame_paths),
                "frames_kept": len(associated),
                "segments_kept": int(total_segments),
                "num_clips": len(clips),
            },
        }
