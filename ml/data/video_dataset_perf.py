"""Manifest frame coverage and VideoClipManifestDataset throughput (no model)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ml.data.data_pipeline import collate_fn
from ml.data.video_clip_dataset import VideoClipManifestDataset
from ml.data.video_manifest import validate_manifest_v1


def summarize_manifest_frame_files(
    manifest_path: Path,
    *,
    manifest_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Count how many manifest frame_paths resolve to existing files."""

    mp = Path(manifest_path)
    root = Path(manifest_root) if manifest_root is not None else mp.parent
    with open(mp, "r", encoding="utf-8") as f:
        data = json.load(f)
    clips = data.get("clips", [])
    if not isinstance(clips, list):
        return {
            "clips": 0,
            "frame_paths_total": 0,
            "frame_paths_existing": 0,
            "frame_paths_missing": 0,
        }

    total = 0
    existing = 0
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        paths = clip.get("frame_paths")
        if not isinstance(paths, list):
            continue
        for p in paths:
            total += 1
            path = Path(p)
            resolved = path if path.is_absolute() else (root / path).resolve()
            if resolved.exists():
                existing += 1

    return {
        "clips": sum(1 for c in clips if isinstance(c, dict)),
        "frame_paths_total": total,
        "frame_paths_existing": existing,
        "frame_paths_missing": total - existing,
    }


def time_manifest_parse_and_validate_ms(manifest_path: Path) -> Dict[str, float]:
    """Split timing for JSON parse vs validate (milliseconds)."""

    mp = Path(manifest_path)
    raw = mp.read_bytes()
    t0 = time.perf_counter()
    data = json.loads(raw.decode("utf-8"))
    t1 = time.perf_counter()
    _ = validate_manifest_v1(data)
    t2 = time.perf_counter()
    return {
        "json_parse_ms": (t1 - t0) * 1000.0,
        "validate_ms": (t2 - t1) * 1000.0,
        "parse_and_validate_ms": (t2 - t0) * 1000.0,
    }


def profile_video_clip_dataset(
    manifest_path: Path,
    *,
    manifest_root: Optional[Path] = None,
    warmup_samples: int = 2,
    timed_getitem_count: int = 32,
    dataloader_batches: int = 10,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 0,
    shuffle_indices: bool = True,
) -> Dict[str, Any]:
    """Measure init, sequential __getitem__, and DataLoader+collate throughput."""

    mp = Path(manifest_path)
    summary = summarize_manifest_frame_files(mp, manifest_root=manifest_root)
    parse_times = time_manifest_parse_and_validate_ms(mp)

    t_init0 = time.perf_counter()
    ds = VideoClipManifestDataset(mp, manifest_root=manifest_root)
    t_init1 = time.perf_counter()
    n = len(ds)
    if n == 0:
        return {
            "clips": 0,
            "manifest_frame_summary": summary,
            "manifest_parse_ms": parse_times,
            "dataset_init_s": t_init1 - t_init0,
            "error": "empty dataset",
        }

    rng = random.Random(seed)
    idx_list = list(range(n))
    if shuffle_indices:
        rng.shuffle(idx_list)

    w = max(0, int(warmup_samples))
    for j in range(w):
        _ = ds[idx_list[j % n]]

    k = max(1, int(timed_getitem_count))
    t_g0 = time.perf_counter()
    for j in range(k):
        _ = ds[idx_list[j % n]]
    t_g1 = time.perf_counter()
    get_wall = t_g1 - t_g0

    from torch.utils.data import DataLoader

    bs = max(1, int(batch_size))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )
    nb = max(1, int(dataloader_batches))
    t_d0 = time.perf_counter()
    batches_done = 0
    clips_in_batches = 0
    for batch in loader:
        _ = batch["images"].shape
        batches_done += 1
        clips_in_batches += int(batch["images"].shape[0])
        if batches_done >= nb:
            break
    t_d1 = time.perf_counter()
    dl_wall = t_d1 - t_d0

    return {
        "clips": n,
        "manifest_frame_summary": summary,
        "manifest_parse_ms": parse_times,
        "dataset_init_s": t_init1 - t_init0,
        "getitem_sequential": {
            "samples_timed": k,
            "wall_s": get_wall,
            "clips_per_s": k / get_wall if get_wall > 0 else 0.0,
            "ms_per_clip_mean": (get_wall / k) * 1000.0 if k else 0.0,
        },
        "dataloader_collate": {
            "batch_size": bs,
            "num_workers": int(num_workers),
            "batches_timed": batches_done,
            "clips_timed": clips_in_batches,
            "wall_s": dl_wall,
            "batches_per_s": batches_done / dl_wall if dl_wall > 0 else 0.0,
            "clips_per_s": clips_in_batches / dl_wall if dl_wall > 0 else 0.0,
        },
    }
