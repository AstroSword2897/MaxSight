#!/usr/bin/env python3
"""Build a v1 video panoptic clip manifest from extracted frames on disk.

This is the standard ingestion step for video datasets (Kinetics-700, Epic-Kitchens,
YouTube-VOS, BDD100K video) before they are usable by `VideoClipManifestDataset`.

Input layout (frames_root)
-------------------------
frames_root/
  <video_id_1>/
    frame_000000.jpg
    frame_000001.jpg
    ...
  <video_id_2>/
    ...

Outputs a JSON file matching docs/schemas/video_panoptic_manifest_v1.schema.json.

Note
----
This builder does not derive segmentation/masks. It emits empty `frames_segments`
lists, which is still valid and useful for representation/temporal consistency
training. If you have boxes/masks, extend this builder to populate `frames_segments`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.video_manifest import CONTRACT_FIXED_STRIDE_T8, validate_manifest_v1  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Root directory containing per-video frame folders.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output manifest JSON path.")
    p.add_argument(
        "--contract", default=CONTRACT_FIXED_STRIDE_T8, help="Contract label for the manifest."
    )
    p.add_argument("--window", type=int, default=8, help="Temporal window size (T).")
    p.add_argument("--stride", type=int, default=1, help="Frame stride within a window.")
    p.add_argument(
        "--overlap", type=int, default=0, help="Overlap in frames between consecutive windows."
    )
    p.add_argument(
        "--ext", default="jpg,png,jpeg", help="Comma-separated frame extensions to include."
    )
    p.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Optional cap for number of videos processed (0 = no cap).",
    )
    p.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="Optional cap for number of clips written (0 = no cap).",
    )
    return p.parse_args()


def _iter_frame_files(video_dir: Path, exts: list[str]) -> list[Path]:
    frames: list[Path] = []
    for e in exts:
        frames.extend(sorted(video_dir.glob(f"*.{e}")))
    # If naming is inconsistent, a stable sort by name is still deterministic.
    return sorted({p.resolve(): p for p in frames}.values(), key=lambda p: p.name)


def _build_windows(
    frames: list[Path], *, window: int, stride: int, overlap: int
) -> list[list[Path]]:
    if window < 2:
        raise ValueError("--window must be >= 2")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if overlap < 0:
        raise ValueError("--overlap must be >= 0")
    step_between_windows = max(1, window - overlap)
    windows: list[list[Path]] = []
    # Build a list of indices sampled with given stride.
    sampled = frames[::stride] if stride > 1 else frames
    for start in range(0, len(sampled) - window + 1, step_between_windows):
        windows.append(sampled[start : start + window])
    return windows


def main() -> int:
    args = parse_args()
    frames_root = args.frames_root.resolve()
    if not frames_root.is_dir():
        print(f"Error: --frames-root not a directory: {frames_root}", file=sys.stderr)
        return 1

    exts = [x.strip().lstrip(".").lower() for x in str(args.ext).split(",") if x.strip()]
    if not exts:
        print("Error: --ext must include at least one extension", file=sys.stderr)
        return 1

    clips: list[dict[str, Any]] = []
    video_dirs = [p for p in sorted(frames_root.iterdir()) if p.is_dir()]
    if args.limit_videos and args.limit_videos > 0:
        video_dirs = video_dirs[: args.limit_videos]

    for vdir in video_dirs:
        frames = _iter_frame_files(vdir, exts)
        if len(frames) < args.window:
            continue
        windows = _build_windows(
            frames, window=args.window, stride=args.stride, overlap=args.overlap
        )
        for wi, win in enumerate(windows):
            if args.limit_clips and args.limit_clips > 0 and len(clips) >= args.limit_clips:
                break
            rel_paths = [str(p.relative_to(frames_root)) for p in win]
            clip_id = f"{vdir.name}:{wi:06d}"
            clips.append(
                {
                    "clip_id": clip_id,
                    "video_id": vdir.name,
                    "start_frame": wi * (args.window - args.overlap),
                    "end_frame": wi * (args.window - args.overlap) + args.window,
                    "temporal_window": args.window,
                    "temporal_stride": args.stride,
                    "temporal_overlap": args.overlap,
                    "frame_paths": rel_paths,
                    "frames_segments": [[] for _ in range(args.window)],
                }
            )
        if args.limit_clips and args.limit_clips > 0 and len(clips) >= args.limit_clips:
            break

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "contract": str(args.contract),
        "clips": clips,
        "stats": {
            "frames_root": str(frames_root),
            "videos_seen": len(video_dirs),
            "clips_written": len(clips),
            "window": args.window,
            "stride": args.stride,
            "overlap": args.overlap,
        },
    }

    errs = validate_manifest_v1(
        manifest, require_fixed_t8=(args.contract == CONTRACT_FIXED_STRIDE_T8)
    )
    if errs:
        print("Error: generated manifest failed validation:", file=sys.stderr)
        for e in errs[:20]:
            print(f"- {e}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {args.out} (clips={len(clips)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
