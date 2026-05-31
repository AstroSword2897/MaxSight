import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_panoptic import (  # noqa: E402
    AdaptiveTemporalConfig,
    PseudoPanopticQualityConfig,
    VideoSamplingConfig,
    associate_tracks_multi_frame,
    build_adaptive_windows,
    build_fixed_stride_windows,
    compute_motion_score,
    iter_chunks,
    motion_to_temporal_window,
    prune_pseudo_segments,
)


def test_build_fixed_stride_windows_with_overlap() -> None:
    cfg = VideoSamplingConfig(temporal_window=8, temporal_stride=1, temporal_overlap=4)
    windows = build_fixed_stride_windows(20, cfg)
    assert windows[0] == (0, 8)
    assert windows[1] == (4, 12)
    assert windows[-1] == (12, 20)


def test_prune_pseudo_segments_filters_low_quality() -> None:
    cfg = PseudoPanopticQualityConfig(
        min_confidence=0.5,
        min_area_pixels=10.0,
        min_bbox_width=3.0,
        min_bbox_height=3.0,
    )
    segments = [
        {"id": 1, "score": 0.9, "area": 30, "bbox": [1, 1, 5, 5]},
        {"id": 2, "score": 0.2, "area": 30, "bbox": [1, 1, 5, 5]},
        {"id": 3, "score": 0.9, "area": 2, "bbox": [1, 1, 5, 5]},
        {"id": 4, "score": 0.9, "area": 30, "bbox": [1, 1, 1, 5]},
    ]
    kept = prune_pseudo_segments(segments, cfg)
    assert [s["id"] for s in kept] == [1]


def test_associate_tracks_multi_frame_maintains_ids() -> None:
    frames = [
        [{"bbox": [10, 10, 10, 10]}],
        [{"bbox": [11, 10, 10, 10]}],
        [{"bbox": [12, 10, 10, 10]}],
    ]
    out = associate_tracks_multi_frame(frames, lookback=2, iou_threshold=0.3)
    ids = [f[0]["track_proxy_id"] for f in out]
    assert ids[0] == ids[1] == ids[2]


def test_iter_chunks_splits_sequence() -> None:
    chunks = list(iter_chunks(list(range(10)), chunk_size=4))
    assert chunks == [list(range(4)), list(range(4, 8)), list(range(8, 10))]


def test_compute_motion_score_low_when_stable() -> None:
    prev_frame = [{"bbox": [10, 10, 20, 20]}]
    curr_frame = [{"bbox": [10, 10, 20, 20]}]
    score = compute_motion_score(prev_frame, curr_frame)
    assert 0.0 <= score < 0.1


def test_motion_to_temporal_window_mapping() -> None:
    cfg = AdaptiveTemporalConfig(t_min=4, t_max=16, smooth_factor=0.0)
    t_low_motion = motion_to_temporal_window(0.0, cfg)
    t_high_motion = motion_to_temporal_window(1.0, cfg)
    assert t_low_motion == 16
    assert t_high_motion == 4


def test_build_adaptive_windows_basic() -> None:
    cfg = AdaptiveTemporalConfig(t_min=4, t_max=8, smooth_factor=0.0, overlap_ratio=0.0)
    frames = [
        [{"bbox": [10, 10, 20, 20]}],
        [{"bbox": [10, 10, 20, 20]}],
        [{"bbox": [40, 10, 20, 20]}],
        [{"bbox": [70, 10, 20, 20]}],
        [{"bbox": [71, 10, 20, 20]}],
        [{"bbox": [71, 10, 20, 20]}],
    ]
    windows = build_adaptive_windows(frames, cfg)
    assert len(windows) >= 1
    for start, end in windows:
        assert 0 <= start < end <= len(frames)
