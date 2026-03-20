#!/usr/bin/env python3
"""Run pseudo-panoptic segmentation over each frame and emit a v1 manifest (full frames_segments)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.video_manifest import CONTRACT_FIXED_STRIDE_T8, MANIFEST_SCHEMA_VERSION, validate_manifest_v1
from ml.data.video_preprocessing import PreprocessingConfig, VideoPanopticPreprocessor
from ml.data.video_panoptic import VideoSamplingConfig


class StubPanopticSegmenter:
    """Deterministic box for smoke tests when no real segmenter is available."""

    def segment(self, frame: Any) -> List[dict]:
        from PIL import Image

        if hasattr(frame, "size"):
            w, h = frame.size
        else:
            w, h = 224, 224
        bw, bh = max(16, w // 4), max(16, h // 4)
        x, y = (w - bw) // 2, (h - bh) // 2
        return [
            {
                "id": 1,
                "class_idx": 0,
                "score": 0.95,
                "area": float(bw * bh),
                "bbox": [float(x), float(y), float(bw), float(bh)],
            }
        ]


def _pil_loader(path: str) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        required=True,
        help="Paths-only v1 manifest from sample_video_clips.py",
    )
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Resolve relative frame_paths (default: sample manifest parent)",
    )
    parser.add_argument("--use-stub-segmenter", action="store_true", help="Use centered box (no torch model)")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temporal-lookback", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--require-fixed-t8",
        action="store_true",
        help="Validate strict T=8 contract after build",
    )
    args = parser.parse_args()

    if not args.use_stub_segmenter:
        print("Only --use-stub-segmenter is implemented in-repo; pass that flag for offline runs.", file=sys.stderr)
        raise SystemExit(2)

    root = args.manifest_root or args.sample_manifest.parent
    with open(args.sample_manifest, "r", encoding="utf-8") as f:
        sample = json.load(f)

    clips_in = sample.get("clips", [])
    if not isinstance(clips_in, list):
        raise SystemExit("sample manifest has no clips array")

    all_clips: List[dict] = []
    stats_frames = 0
    global_tw8 = True
    for clip in clips_in:
        if not isinstance(clip, dict):
            continue
        paths = clip.get("frame_paths")
        if not isinstance(paths, list):
            continue
        resolved = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = (Path(root) / pp).resolve()
            resolved.append(str(pp))
        video_id = str(clip.get("video_id", "video"))
        tw = len(resolved)
        if tw != 8:
            global_tw8 = False
        sampling = VideoSamplingConfig(
            temporal_window=tw,
            temporal_stride=int(clip.get("temporal_stride", 1)),
            temporal_overlap=int(clip.get("temporal_overlap", 0)),
        )
        config = PreprocessingConfig(
            sampling=sampling,
            chunk_size=args.chunk_size,
            segmentation_workers=args.workers,
            temporal_lookback=args.temporal_lookback,
            temporal_iou_threshold=args.iou_threshold,
        )
        preprocessor = VideoPanopticPreprocessor(
            segmenter=StubPanopticSegmenter(),
            frame_loader=_pil_loader,
            config=config,
        )
        out = preprocessor.process_video(video_id, resolved)
        stats_frames += int(out.get("stats", {}).get("frames_total", len(resolved)))
        for c in out.get("clips", []):
            if isinstance(c, dict):
                rel_frames = []
                for p, orig in zip(c.get("frame_paths", []), paths):
                    pth = Path(p)
                    try:
                        rel_frames.append(str(pth.resolve().relative_to(Path(root).resolve())))
                    except ValueError:
                        rel_frames.append(orig)
                c = dict(c)
                c["frame_paths"] = rel_frames
                all_clips.append(c)

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "clips": all_clips,
        "stats": {
            "source_manifest": str(args.sample_manifest),
            "frames_processed": stats_frames,
            "num_clips": len(all_clips),
        },
    }
    if global_tw8 and args.require_fixed_t8:
        payload["contract"] = CONTRACT_FIXED_STRIDE_T8
    errs = validate_manifest_v1(payload, require_fixed_t8=args.require_fixed_t8)
    if errs:
        raise SystemExit("Validation failed: " + "; ".join(errs))

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.manifest_out} with {len(all_clips)} clips.")


if __name__ == "__main__":
    main()
