#!/usr/bin/env python3
"""Build bronze/silver/gold layout: COCO splits into silver, gold training_index.json; optional video manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.download_datasets import verify_coco_dataset  # noqa: E402
from ml.data.medallion_layout import (  # noqa: E402
    bronze_coco_dir,
    bronze_video_dir,
    build_coco_training_index,
    default_gold_index_path,
    default_medallion_root,
    load_training_index,
    merge_video_into_index,
    path_relative_to_repo,
    silver_coco_splits_dir,
    silver_video_dir,
    write_training_index,
)
from ml.data.video_manifest import validate_manifest_v1  # noqa: E402


def _ensure_layout(mroot: Path) -> None:
    for d in (
        bronze_coco_dir(mroot),
        bronze_video_dir(mroot),
        silver_coco_splits_dir(mroot),
        silver_video_dir(mroot),
        mroot / "gold",
    ):
        d.mkdir(parents=True, exist_ok=True)


def _find_coco_ann_and_images(bronze_coco: Path) -> tuple[Path, Path]:
    ann_file = bronze_coco / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        ann_file = bronze_coco / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"No instances_train2017.json / instances_val2017.json under {bronze_coco}/annotations")

    image_dir = bronze_coco / "train2017"
    if not image_dir.exists():
        image_dir = bronze_coco / "val2017"
    if not image_dir.exists():
        image_dir = bronze_coco
    return ann_file, image_dir


def cmd_promote_coco(args: argparse.Namespace) -> int:
    mroot = Path(args.medallion_root).resolve()
    _ensure_layout(mroot)
    bronze_coco = Path(args.bronze_coco).resolve()
    if not bronze_coco.exists():
        print(f"Bronze COCO dir missing: {bronze_coco}", file=sys.stderr)
        return 1

    status = verify_coco_dataset(bronze_coco, check_coco_raw=(bronze_coco.name == "coco_raw"))
    if not (status.get("train_images") or status.get("val_images")) or not status.get("annotations"):
        print("COCO looks incomplete. Download/extract first (see docs/medallion_data.md).", file=sys.stderr)
        return 1

    ann_file, image_dir = _find_coco_ann_and_images(bronze_coco)
    splits_dir = silver_coco_splits_dir(mroot)
    splits_dir.mkdir(parents=True, exist_ok=True)

    from ml.data.coco_dataset_splitter import create_maxsight_splits_from_coco

    train_file, val_file, test_file = create_maxsight_splits_from_coco(
        coco_annotation_file=ann_file,
        image_dir=image_dir,
        output_dir=splits_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        num_samples=args.train_samples + args.val_samples + args.test_samples,
    )

    idx = build_coco_training_index(
        REPO,
        bronze_coco_data_dir=bronze_coco,
        train_annotation=train_file,
        val_annotation=val_file,
        test_annotation=test_file,
        image_dir=bronze_coco,
    )
    gold_path = Path(args.gold_index_out).resolve() if args.gold_index_out else default_gold_index_path(mroot)
    write_training_index(gold_path, idx)
    print(json.dumps({"gold_index": str(gold_path), "train": str(train_file), "val": str(val_file)}, indent=2))
    return 0


def cmd_promote_video(args: argparse.Namespace) -> int:
    mroot = Path(args.medallion_root).resolve()
    _ensure_layout(mroot)
    sv = silver_video_dir(mroot)
    sv.mkdir(parents=True, exist_ok=True)

    gold_path = Path(args.gold_index_out).resolve() if args.gold_index_out else default_gold_index_path(mroot)
    if gold_path.exists():
        base = load_training_index(gold_path)
    else:
        print("Gold index missing; run --promote-coco first.", file=sys.stderr)
        return 1

    updates: dict[str, str | None] = {
        "train_manifest": None,
        "val_manifest": None,
        "manifest_root": None,
    }
    mr = Path(args.video_manifest_root).resolve() if args.video_manifest_root else None

    def _ingest(src: Path, dest_name: str) -> Path:
        if not src.exists():
            raise FileNotFoundError(src)
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        errs = validate_manifest_v1(data)
        if errs:
            raise ValueError("Invalid manifest: " + "; ".join(errs[:5]))
        dest = sv / dest_name
        shutil.copy2(src, dest)
        return dest

    if args.video_train_manifest:
        p = _ingest(Path(args.video_train_manifest), "train_manifest.json")
        updates["train_manifest"] = path_relative_to_repo(p, REPO)
    if args.video_val_manifest:
        p = _ingest(Path(args.video_val_manifest), "val_manifest.json")
        updates["val_manifest"] = path_relative_to_repo(p, REPO)
    if mr is not None:
        updates["manifest_root"] = path_relative_to_repo(mr, REPO)

    if not args.video_train_manifest and not args.video_val_manifest and mr is None:
        print("Nothing to do: pass --video-train-manifest, --video-val-manifest, and/or --video-manifest-root.", file=sys.stderr)
        return 1

    merged = merge_video_into_index(base, updates)
    write_training_index(gold_path, merged)
    print(json.dumps({"gold_index": str(gold_path), "video": merged.get("video")}, indent=2))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    mroot = Path(args.medallion_root).resolve()
    _ensure_layout(mroot)
    print(f"Medallion dirs ready under {mroot}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medallion-root",
        type=Path,
        default=default_medallion_root(REPO),
        help="Medallion root (default: datasets/medallion)",
    )
    parser.add_argument(
        "--gold-index-out",
        type=Path,
        default=None,
        help="Write training_index.json here (default: <medallion-root>/gold/training_index.json)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create bronze/silver/gold directory tree")
    p_init.set_defaults(func=cmd_init)

    p_coco = sub.add_parser("promote-coco", help="Verify bronze COCO, write MaxSight splits to silver, gold index")
    p_coco.add_argument(
        "--bronze-coco",
        type=Path,
        default=REPO / "datasets" / "coco_raw",
        help="Bronze COCO root (train2017, annotations, …)",
    )
    p_coco.add_argument("--train-samples", type=int, default=10000)
    p_coco.add_argument("--val-samples", type=int, default=2000)
    p_coco.add_argument("--test-samples", type=int, default=1000)
    p_coco.add_argument("--seed", type=int, default=42)
    p_coco.add_argument(
        "--publish-cleaned-splits",
        action="store_true",
        help="Copy split JSONs to datasets/cleaned_splits/maxsight_*.json (dataset registry paths)",
    )
    p_coco.set_defaults(func=cmd_promote_coco)

    p_vid = sub.add_parser("promote-video", help="Validate and copy v1 manifests into silver/video; update gold index")
    p_vid.add_argument("--video-train-manifest", type=Path, default=None)
    p_vid.add_argument("--video-val-manifest", type=Path, default=None)
    p_vid.add_argument(
        "--video-manifest-root",
        type=Path,
        default=None,
        help="Directory used to resolve relative frame_paths in manifests",
    )
    p_vid.set_defaults(func=cmd_promote_video)

    p_all = sub.add_parser("all", help="init + promote-coco (same flags as promote-coco)")
    p_all.add_argument("--bronze-coco", type=Path, default=REPO / "datasets" / "coco_raw")
    p_all.add_argument("--train-samples", type=int, default=10000)
    p_all.add_argument("--val-samples", type=int, default=2000)
    p_all.add_argument("--test-samples", type=int, default=1000)
    p_all.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "all":
        mroot = Path(args.medallion_root).resolve()
        _ensure_layout(mroot)
        ns = argparse.Namespace(
            medallion_root=mroot,
            gold_index_out=args.gold_index_out,
            bronze_coco=args.bronze_coco,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            seed=args.seed,
        )
        return cmd_promote_coco(ns)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())