import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_preprocessing import (  # noqa: E402
    PanopticSegmenter,
    PreprocessingConfig,
    VideoPanopticPreprocessor,
)
from ml.data.video_panoptic import (  # noqa: E402
    AdaptiveTemporalConfig,
    PseudoPanopticQualityConfig,
    VideoSamplingConfig,
)


class DummySegmenter(PanopticSegmenter):
    def segment(self, frame: Any) -> List[Dict[str, Any]]:
        idx = int(frame)
        return [
            {"bbox": [idx, 0, 10, 10], "score": 0.9, "area": 100},
            {"bbox": [idx, 0, 1, 1], "score": 0.2, "area": 1},
        ]


def test_video_panoptic_preprocessor_builds_clips() -> None:
    cfg = PreprocessingConfig(
        sampling=VideoSamplingConfig(temporal_window=4, temporal_stride=1, temporal_overlap=2),
        quality=PseudoPanopticQualityConfig(min_confidence=0.5, min_area_pixels=10.0),
        chunk_size=3,
        segmentation_workers=2,
        temporal_lookback=2,
        temporal_iou_threshold=0.2,
    )
    loader = lambda p: p  # noqa: E731
    pipeline = VideoPanopticPreprocessor(segmenter=DummySegmenter(), frame_loader=loader, config=cfg)
    frame_paths = [str(i) for i in range(10)]
    out = pipeline.process_video("vid-1", frame_paths)

    assert out["video_id"] == "vid-1"
    assert out["stats"]["frames_total"] == 10
    assert out["stats"]["num_clips"] > 0
    first_clip = out["clips"][0]
    assert len(first_clip["frame_paths"]) == 4
    assert len(first_clip["frames_segments"]) == 4
    # Low-quality segment must be pruned.
    assert all(len(frame_segments) == 1 for frame_segments in first_clip["frames_segments"])
    assert "track_proxy_id" in first_clip["frames_segments"][0][0]


def test_video_panoptic_preprocessor_adaptive_windows() -> None:
    cfg = PreprocessingConfig(
        sampling=VideoSamplingConfig(temporal_window=8, temporal_stride=1, temporal_overlap=2),
        quality=PseudoPanopticQualityConfig(min_confidence=0.5, min_area_pixels=10.0),
        chunk_size=2,
        segmentation_workers=2,
        temporal_lookback=2,
        temporal_iou_threshold=0.2,
        enable_adaptive_windowing=True,
        adaptive=AdaptiveTemporalConfig(t_min=3, t_max=6, smooth_factor=0.0, overlap_ratio=0.0),
    )
    loader = lambda p: p  # noqa: E731
    pipeline = VideoPanopticPreprocessor(segmenter=DummySegmenter(), frame_loader=loader, config=cfg)
    out = pipeline.process_video("vid-adaptive", [str(i) for i in range(12)])
    assert out["stats"]["num_clips"] > 0
    for clip in out["clips"]:
        t = len(clip["frame_paths"])
        assert 1 <= t <= 6

