#!/usr/bin/env python3
"""Acquire a class-capped dataset subset into data/raw and write a manifest.

Examples:
  python scripts/acquire_data.py coco --root data/raw/coco --per-class 4000 --max-gb 4 --dry-run
  python scripts/acquire_data.py coco --root data/raw/coco --per-class 4000 --max-gb 4
  python scripts/acquire_data.py openimages --split validation --dry-run
  python scripts/acquire_data.py openimages --split validation --max-images 2000 --approve-gb 1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.datasets.acquisition import coco, fetch  # noqa: E402
from ml.datasets.acquisition import openimages_manual as oim  # noqa: E402
from ml.datasets.label_map import DEFAULT_PATH as DEFAULT_LABEL_MAP  # noqa: E402
from ml.datasets.label_map import LabelMap  # noqa: E402
from ml.datasets.manifest import write_manifest  # noqa: E402
from ml.datasets.sources import write_acquired_marker  # noqa: E402


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
    write_acquired_marker(root, verified=False, reason="COCO publishes no per-image checksum",
                          images=len(ok), failures=len(failed), manifest=str(out))
    print(f"downloaded {len(ok)} images, {len(failed)} failures -> {out}")
    for image_id, err in failed[:10]:
        print(f"  FAILED {image_id}: {err}")
    return 0 if not failed else 1


def cmd_openimages(args: argparse.Namespace) -> int:
    """Official manual route: boxable class list + bbox CSV -> capped image list -> anonymous S3 download."""
    root = Path(args.root)
    ann = root / "annotations"
    size = oim.BBOX_BYTES.get(args.split)
    need_gb = (size or 0) / 1024**3
    if (size is None or need_gb > 0.5) and args.approve_gb < max(need_gb, 0.01):
        print(f"Refusing: the {args.split} bbox CSV is {'of unverified size' if size is None else f'{need_gb:.2f} GB'}; "
              f"pass --approve-gb to allow it.")
        return 2
    lm = LabelMap.load(args.label_map)
    classes = fetch.download_resumable(oim.URLS["class_descriptions"], ann / "class-descriptions-boxable.csv")
    bbox = fetch.download_resumable(oim.URLS[f"bbox_{args.split}"], ann / f"{args.split}-annotations-bbox.csv")
    display_to_id = oim.parse_class_descriptions(classes.read_text(encoding="utf-8"))
    requested = args.classes or sorted(lm.sources["openimages"])
    wanted, missing = oim.resolve_classes(display_to_id, requested)
    for name in missing:
        print(f"  UNAVAILABLE (not an Open Images boxable class): {name}")
    sel = oim.select_images(oim.iter_bbox_rows(bbox, wanted), args.per_class, seed=args.seed,
                            max_images=args.max_images)
    print(f"selected {len(sel.images)} {args.split} images")
    for label_id, name in sorted(wanted.items(), key=lambda kv: kv[1]):
        print(f"  {name:18s} {sel.class_counts.get(label_id, 0):7d}")
    list_path = root / f"image_list_{args.split}.txt"
    oim.write_image_list(args.split, sel.images, list_path)
    print(f"image list: {list_path}")
    if args.dry_run:
        return 0
    ok, failed = oim.download_images(oim.read_image_list(list_path), root / "images", workers=args.workers)
    records = oim.to_records(sel, wanted, args.split, root, lm)
    out = Path(args.manifest or root / f"manifest_{args.split}.jsonl")
    write_manifest(records, out)
    write_acquired_marker(root, verified=False, reason="Open Images S3 publishes no per-image checksum",
                          split=args.split, images=len(ok), failures=len(failed), records=len(records),
                          manifest=str(out))
    print(f"downloaded {len(ok)} images, {len(failed)} failures, {len(records)} records -> {out}")
    for key, err in failed[:10]:
        print(f"  FAILED {key}: {err}")
    return 0 if not failed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coco", help="COCO 2017 class-capped subset")
    c.add_argument("--root", default="datasets/raw/coco2017")
    c.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    c.add_argument("--per-class", type=int, default=4000)
    c.add_argument("--max-gb", type=float, default=4.0, help="hard download budget")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    c.add_argument("--manifest", default=None)
    c.add_argument("--dry-run", action="store_true", help="select and report only; download nothing")
    c.set_defaults(func=cmd_coco)
    o = sub.add_parser("openimages", help="Open Images V7 subset (official CSVs + anonymous S3, needs boto3)")
    o.add_argument("--root", default="datasets/raw/open_images_v7")
    o.add_argument("--split", default="validation", choices=list(oim.SPLITS))
    o.add_argument("--classes", nargs="*", help="display names; default: every class in the label map")
    o.add_argument("--per-class", type=int, default=4000)
    o.add_argument("--max-images", type=int, default=20000, help="hard cap on images downloaded")
    o.add_argument("--approve-gb", type=float, default=0.0, help="allow a large annotation CSV (train is 2.3 GB)")
    o.add_argument("--workers", type=int, default=5)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    o.add_argument("--manifest", default=None)
    o.add_argument("--dry-run", action="store_true", help="select and write the image list; download no images")
    o.set_defaults(func=cmd_openimages)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
