import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_manifest import validate_manifest_v1  # noqa: E402


def _valid_clip(**kwargs):
    base = {
        "clip_id": "c0",
        "video_id": "v0",
        "start_frame": 0,
        "end_frame": 8,
        "temporal_window": 8,
        "temporal_stride": 1,
        "temporal_overlap": 0,
        "frame_paths": [f"f{i}.jpg" for i in range(8)],
        "frames_segments": [[] for _ in range(8)],
    }
    base.update(kwargs)
    return base


def test_validate_manifest_ok() -> None:
    data = {"schema_version": "1.0", "clips": [_valid_clip()]}
    assert validate_manifest_v1(data) == []


def test_validate_manifest_wrong_version() -> None:
    data = {"schema_version": "0.9", "clips": [_valid_clip()]}
    errs = validate_manifest_v1(data)
    assert len(errs) >= 1
    assert any("schema_version" in e for e in errs)


def test_validate_fixed_t8_fails_on_length() -> None:
    clip = _valid_clip()
    clip["frame_paths"] = clip["frame_paths"][:7]
    data = {"schema_version": "1.0", "clips": [clip]}
    errs = validate_manifest_v1(data, require_fixed_t8=True)
    assert errs
    assert any("fixed_stride_t8" in e or "temporal_window" in e for e in errs)


def test_validate_manifest_rejects_end_before_start() -> None:
    c = _valid_clip()
    c["start_frame"] = 10
    c["end_frame"] = 4
    errs = validate_manifest_v1({"schema_version": "1.0", "clips": [c]})
    assert any("end_frame" in e.lower() for e in errs)


def test_validate_manifest_rejects_tw_mismatch() -> None:
    c = _valid_clip()
    c["temporal_window"] = 7
    errs = validate_manifest_v1({"schema_version": "1.0", "clips": [c]})
    assert any("temporal_window" in e for e in errs)


def test_validate_manifest_rejects_bad_segment_bbox() -> None:
    c = _valid_clip()
    bad_seg = [{"bbox": [1, 2, 3]}]
    c["frames_segments"] = [bad_seg if i == 0 else [] for i in range(8)]
    errs = validate_manifest_v1({"schema_version": "1.0", "clips": [c]})
    assert any("bbox" in e for e in errs)


def test_validate_manifest_empty_clips_allowed() -> None:
    assert validate_manifest_v1({"schema_version": "1.0", "clips": []}) == []


def test_validate_manifest_missing_clip_field() -> None:
    c = _valid_clip()
    del c["clip_id"]
    errs = validate_manifest_v1({"schema_version": "1.0", "clips": [c]})
    assert any("clip_id" in e for e in errs)
