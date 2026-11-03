#!/usr/bin/env python3
"""Pre-training dataset gate: clean (optional), split, audit a manifest.

  python scripts/validate_data.py --manifest data/raw/coco/manifest.jsonl --root data/raw/coco \
      --clean --split --out data/gold/coco_clean.jsonl --report artifacts/evaluation-reports/data_audit.json
Exit code 0 only if the audit has no errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.datasets.audit import audit  # noqa: E402
from ml.datasets.cleaning import clean  # noqa: E402
from ml.datasets.label_map import DEFAULT_PATH, LabelMap  # noqa: E402
from ml.datasets.manifest import read_manifest, write_manifest  # noqa: E402
from ml.datasets.splits import assign_splits  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True, help="directory the manifest paths are relative to")
    p.add_argument("--label-map", default=str(DEFAULT_PATH))
    p.add_argument("--clean", action="store_true", help="decode check, small-box filter, dedup, class cap")
    p.add_argument("--per-class-cap", type=int, default=None)
    p.add_argument("--split", action="store_true", help="assign train/val/test by group (70/15/15)")
    p.add_argument("--holdout", nargs="*", default=[], help="group or environment names held out entirely")
    p.add_argument("--max-class-share", type=float, default=0.15)
    p.add_argument("--out", help="write the cleaned/split manifest here")
    p.add_argument("--report", help="write the audit JSON here")
    args = p.parse_args()

    lm = LabelMap.load(args.label_map)
    root = Path(args.root)
    records = list(read_manifest(args.manifest))
    clean_info = None
    if args.clean:
        records, rep = clean(records, root, per_class_cap=args.per_class_cap)
        clean_info = rep.as_dict()
        print("clean:", json.dumps(clean_info))
    if args.split:
        records = assign_splits(records, holdout_groups=frozenset(args.holdout))
    res = audit(records, lm, root=root, max_class_share=args.max_class_share)
    for cls in lm.classes:
        print(f"  {cls:14s} {res.class_counts.get(cls, 0):7d}")
    print("splits:", dict(res.split_counts))
    for w in res.warnings:
        print("WARN ", w)
    for e in res.errors[:25]:
        print("ERROR", e)
    if len(res.errors) > 25:
        print(f"... {len(res.errors) - 25} more errors")
    if args.out:
        write_manifest(records, args.out)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({**res.as_dict(), "clean": clean_info}, indent=2))
    print("AUDIT", "PASSED" if res.ok else "FAILED")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
