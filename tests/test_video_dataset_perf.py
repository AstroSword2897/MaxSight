import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_dataset_perf import (  # noqa: E402
    profile_video_clip_dataset,
    summarize_manifest_frame_files,
    time_manifest_parse_and_validate_ms,
)


def _tiny_manifest(tmp_path: Path) -> Path:
    img_dir = tmp_path / "v"
    img_dir.mkdir()
    for i in range(8):
        p = img_dir / f"f{i:03d}.jpg"
        Image.new("RGB", (64, 64), color=(i * 10, 40, 80)).save(p, format="JPEG")
    rel = [f"v/f{i:03d}.jpg" for i in range(8)]
    manifest = {
        "schema_version": "1.0",
        "contract": "fixed_stride_t8",
        "clips": [
            {
                "clip_id": "c0",
                "video_id": "v",
                "start_frame": 0,
                "end_frame": 8,
                "temporal_window": 8,
                "temporal_stride": 1,
                "temporal_overlap": 0,
                "frame_paths": rel,
                "frames_segments": [
                    [{"bbox": [10, 10, 20, 20], "class_idx": 0, "track_proxy_id": 1}]
                    for _ in range(8)
                ],
            }
        ],
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return mp


def test_summarize_manifest_frame_files_counts(tmp_path: Path) -> None:
    mp = _tiny_manifest(tmp_path)
    s = summarize_manifest_frame_files(mp, manifest_root=tmp_path)
    assert s["clips"] == 1
    assert s["frame_paths_total"] == 8
    assert s["frame_paths_existing"] == 8
    assert s["frame_paths_missing"] == 0


def test_time_manifest_parse_and_validate_ms_positive(tmp_path: Path) -> None:
    mp = _tiny_manifest(tmp_path)
    t = time_manifest_parse_and_validate_ms(mp)
    assert t["json_parse_ms"] >= 0
    assert t["validate_ms"] >= 0
    assert t["parse_and_validate_ms"] > 0


def test_profile_video_clip_dataset_reports_throughput(tmp_path: Path) -> None:
    mp = _tiny_manifest(tmp_path)
    data = json.loads(mp.read_text(encoding="utf-8"))
    clip0 = data["clips"][0]
    data["clips"].append({**clip0, "clip_id": "c1"})
    mp.write_text(json.dumps(data), encoding="utf-8")

    r = profile_video_clip_dataset(
        mp,
        manifest_root=tmp_path,
        warmup_samples=1,
        timed_getitem_count=4,
        dataloader_batches=2,
        batch_size=1,
        num_workers=0,
        seed=0,
    )
    assert r["clips"] == 2
    assert r["manifest_frame_summary"]["frame_paths_existing"] == 16
    assert r["getitem_sequential"]["clips_per_s"] > 0
    assert r["dataloader_collate"]["batches_timed"] == 2
    assert r["dataloader_collate"]["clips_per_s"] > 0
