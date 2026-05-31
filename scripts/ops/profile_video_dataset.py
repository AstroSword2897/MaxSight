#!/usr/bin/env python3
"""Benchmark video clip manifest loading and DataLoader throughput (no model)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.video_dataset_perf import (  # noqa: E402
    profile_video_clip_dataset,
    summarize_manifest_frame_files,
    time_manifest_parse_and_validate_ms,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSON")
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Directory for resolving relative frame_paths (default: manifest parent)",
    )
    p.add_argument(
        "--summary-only", action="store_true", help="Only frame file coverage + parse times"
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--getitem-samples", type=int, default=32)
    p.add_argument("--dataloader-batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None, help="Write full report as JSON")
    args = p.parse_args()

    mp = args.manifest
    root = args.manifest_root
    if args.summary_only:
        cov = summarize_manifest_frame_files(mp, manifest_root=root)
        times = time_manifest_parse_and_validate_ms(mp)
        out = {"manifest_frame_summary": cov, "manifest_parse_ms": times}
        print(json.dumps(out, indent=2))
        if args.json_out:
            args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return

    report = profile_video_clip_dataset(
        mp,
        manifest_root=root,
        warmup_samples=args.warmup,
        timed_getitem_count=args.getitem_samples,
        dataloader_batches=args.dataloader_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
