#!/usr/bin/env python3
"""Acquire a class-capped dataset subset into data/raw and write a manifest.

Examples:
  python scripts/acquire_data.py coco --root data/raw/coco --per-class 4000 --max-gb 4 --dry-run
  python scripts/acquire_data.py coco --root data/raw/coco --per-class 4000 --max-gb 4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.datasets.acquisition import coco  # noqa: E402
from ml.datasets.label_map import DEFAULT_PATH as DEFAULT_LABEL_MAP, LabelMap  # noqa: E402
from ml.datasets.manifest import write_manifest  # noqa: E402


def cmd_coco(args: argparse.Namespace) -> int:
    root = Path(args.root)
    lm = LabelMap.load(args.label_map)
    ann_path = coco.fetch_annotations(root / "annotations", split=args.split)
    records = coco.select_records(coco.load_annotations(ann_path), lm, args.per_class, seed=args.seed)
    counts = Counter(b.cls for r in records for b in r.boxes)
    est_gb = len(records) * 0.16 / 1024  # COCO images average ~160 KB
    print(f"selected {len(records)} images; estimated download {est_gb:.1f} GB (budget {args.max_gb} GB)")
    for cls in lm.classes:
        print(f"  {cls:14s} {counts.get(cls, 0):7d}")
    if args.dry_run:
        return 0
    ok, failed = coco.download_images(records, root, max_gb=args.max_gb, workers=args.workers)
    out = Path(args.manifest or root / "manifest.jsonl")
    write_manifest(ok, out)
    print(f"downloaded {len(ok)} images, {len(failed)} failures -> {out}")
    for image_id, err in failed[:10]:
        print(f"  FAILED {image_id}: {err}")
    return 0 if not failed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coco", help="COCO 2017 class-capped subset")
    c.add_argument("--root", default="data/raw/coco")
    c.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    c.add_argument("--per-class", type=int, default=4000)
    c.add_argument("--max-gb", type=float, default=4.0, help="hard download budget")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    c.add_argument("--manifest", default=None)
    c.add_argument("--dry-run", action="store_true", help="select and report only; download nothing")
    c.set_defaults(func=cmd_coco)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
