import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_clip_dataset import VideoClipManifestDataset  # noqa: E402


def test_video_clip_dataset_loads_and_collate_keys(tmp_path: Path) -> None:
    img_dir = tmp_path / "v"
    img_dir.mkdir()
    for i in range(8):
        p = img_dir / f"f{i:03d}.jpg"
        Image.new("RGB", (128, 128), color=(i * 10, 40, 80)).save(p, format="JPEG")

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
                    [{"bbox": [32, 32, 64, 64], "class_idx": 0, "track_proxy_id": 1}]
                ]
                + [
                    [{"bbox": [33, 32, 64, 64], "class_idx": 0, "track_proxy_id": 1}]
                    for _ in range(7)
                ],
            }
        ],
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    ds = VideoClipManifestDataset(mp, manifest_root=tmp_path)
    assert len(ds) == 1
    row = ds[0]
    assert row["frames"].shape[0] == 8
    assert row["frames"].shape[1] == 3
    assert "temporal_consistency" in row
    assert "flicker" in row

    from ml.data.data_pipeline import collate_fn  # noqa: E402

    batch = collate_fn([row])
    assert batch["images"].shape[0] == 1
    assert batch["images"].shape[1] == 8
    assert "frame_lengths" in batch
    assert batch["frame_lengths"].tolist() == [8]
    assert "temporal_consistency" in batch
    assert "flicker" in batch
    assert batch["clip_ids"] == ["c0"]
    assert batch["temporal_consistency"].dtype == torch.float32
    assert batch["flicker"].dtype == torch.float32
    assert 0.0 <= float(batch["temporal_consistency"][0]) <= 1.0
    assert 0.0 <= float(batch["flicker"][0]) <= 1.0
    assert row["num_objects"].item() >= 1
    assert row["labels"][0].item() == 0


def test_video_clip_dataset_invalid_manifest_raises(tmp_path: Path) -> None:
    mp = tmp_path / "bad.json"
    mp.write_text(json.dumps({"schema_version": "1.0", "clips": []}), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema_version": "0.5", "clips": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid manifest|schema_version"):
        VideoClipManifestDataset(broken, manifest_root=tmp_path)
