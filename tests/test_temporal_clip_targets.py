import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.temporal_clip_targets import derive_temporal_clip_targets  # noqa: E402


def test_derive_temporal_single_frame() -> None:
    t = derive_temporal_clip_targets([[{"bbox": [0, 0, 10, 10], "track_proxy_id": 1}]])
    assert t.temporal_consistency == 1.0
    assert t.flicker_proxy == 0.0


def test_derive_temporal_stable_track() -> None:
    b = {"bbox": [10, 10, 20, 20], "track_proxy_id": 1}
    t = derive_temporal_clip_targets([[b], [b], [b]])
    assert t.temporal_consistency > 0.99
    assert t.flicker_proxy < 0.05


def test_derive_temporal_empty_all_frames() -> None:
    t = derive_temporal_clip_targets([[], [], []])
    assert t.temporal_consistency == 1.0
    assert t.flicker_proxy == 0.0


def test_derive_temporal_large_displacement_lowers_consistency() -> None:
    a = {"bbox": [10, 10, 20, 20], "track_proxy_id": 1}
    b = {"bbox": [100, 100, 20, 20], "track_proxy_id": 1}
    stable = derive_temporal_clip_targets([[a], [a]])
    jumped = derive_temporal_clip_targets([[a], [b]])
    assert jumped.temporal_consistency < stable.temporal_consistency
    assert jumped.flicker_proxy > stable.flicker_proxy
