#!/usr/bin/env python3
"""Build a v1 clip manifest from a video file or a directory of frames (paths-only segments)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.video_manifest import CONTRACT_FIXED_STRIDE_T8, MANIFEST_SCHEMA_VERSION, validate_manifest_v1
from ml.data.video_panoptic import VideoSamplingConfig, build_fixed_stride_windows


def _extract_video_frames(video_path: Path, out_dir: Path, image_ext: str = ".jpg") -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    paths: list[Path] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fp = out_dir / f"frame_{idx:06d}{image_ext}"
        cv2.imwrite(str(fp), frame)
        paths.append(fp)
        idx += 1
    cap.release()
    if not paths:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return paths


def _frames_from_dir(frames_dir: Path, image_ext: str | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(frames_dir.iterdir())
    out: list[Path] = []
    for f in files:
        if not f.is_file():
            continue
        if image_ext and f.suffix.lower() != image_ext.lower():
            continue
        if f.suffix.lower() in exts:
            out.append(f)
    return out


def build_manifest(
    frame_paths: list[Path],
    *,
    video_id: str,
    sampling: VideoSamplingConfig,
    relative_to: Path,
    contract_t8: bool,
) -> dict:
    rel_paths = []
    for p in frame_paths:
        try:
            rel_paths.append(str(p.resolve().relative_to(relative_to.resolve())))
        except ValueError:
            rel_paths.append(str(p.resolve()))

    windows = build_fixed_stride_windows(len(rel_paths), sampling)
    clips = []
    for clip_idx, (start, end) in enumerate(windows):
        span = end - start
        clip_paths = rel_paths[start:end]
        clips.append(
            {
                "clip_id": f"{video_id}_clip_{clip_idx:06d}",
                "video_id": video_id,
                "start_frame": int(start),
                "end_frame": int(end),
                "temporal_window": span,
                "temporal_stride": int(sampling.temporal_stride),
                "temporal_overlap": int(sampling.temporal_overlap),
                "frame_paths": clip_paths,
                "frames_segments": [[] for _ in range(span)],
            }
        )

    payload: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "clips": clips,
        "stats": {
            "frames_total": len(rel_paths),
            "num_clips": len(clips),
        },
    }
    if contract_t8 and sampling.temporal_window == 8:
        payload["contract"] = CONTRACT_FIXED_STRIDE_T8
    errs = validate_manifest_v1(payload, require_fixed_t8=contract_t8 and sampling.temporal_window == 8)
    if errs:
        raise ValueError("Manifest validation failed: " + "; ".join(errs))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Input video file")
    src.add_argument("--frames-dir", type=Path, help="Directory of ordered frame images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write extracted frames (required with --video)",
    )
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output JSON manifest path")
    parser.add_argument("--video-id", type=str, default=None, help="Logical video id (default: stem)")
    parser.add_argument("--temporal-window", type=int, default=8)
    parser.add_argument("--temporal-overlap", type=int, default=0)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Directory frame_paths are relative to (default: manifest parent dir)",
    )
    parser.add_argument(
        "--require-fixed-t8",
        action="store_true",
        help="Fail unless contract is strict T=8 (temporal_window==8 and all clips length 8)",
    )
    parser.add_argument("--image-ext", type=str, default=None, help="Filter extension e.g. .jpg")
    args = parser.parse_args()

    sampling = VideoSamplingConfig(
        temporal_window=args.temporal_window,
        temporal_stride=args.temporal_stride,
        temporal_overlap=args.temporal_overlap,
    )
    sampling.validate()

    if args.video:
        if not args.output_dir:
            parser.error("--output-dir is required with --video")
        vid = args.video
        video_id = args.video_id or vid.stem
        out_sub = args.output_dir / video_id
        paths = _extract_video_frames(vid, out_sub)
        relative_to = args.manifest_root or args.manifest_out.parent
    else:
        d = args.frames_dir
        assert d is not None
        paths = _frames_from_dir(d, args.image_ext)
        if not paths:
            raise SystemExit(f"No frames found under {d}")
        video_id = args.video_id or d.name
        relative_to = args.manifest_root or args.manifest_out.parent

    contract_t8 = args.require_fixed_t8 or (sampling.temporal_window == 8)
    manifest = build_manifest(
        paths,
        video_id=video_id,
        sampling=sampling,
        relative_to=relative_to,
        contract_t8=contract_t8 and sampling.temporal_window == 8,
    )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.manifest_out} with {len(manifest['clips'])} clips.")


if __name__ == "__main__":
    main()
