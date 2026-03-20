#!/usr/bin/env python3
"""Register a dataset directory into bronze and write an ingest record.

This script does NOT move or copy data — it records where the raw data lives
so the cleaning pipeline knows where to find it.

Prefer canonical roots from ``ml.data.medallion_layout.default_raw_dataset_dir``
(``datasets/coco_raw``, ``datasets/raw/<key>/``); see ``docs/medallion_data.md``.

Usage
-----
# Register COCO
python scripts/ops/ingest_datasets.py coco --path /data/coco_raw

# Register BDD100K
python scripts/ops/ingest_datasets.py bdd100k --path /data/BDD100K

# Register Epic-Kitchens-100
python scripts/ops/ingest_datasets.py epic_kitchens --path /data/epic_kitchens

# Register MOSE
python scripts/ops/ingest_datasets.py mose --path /data/MOSE

# Register YouTube-VOS
python scripts/ops/ingest_datasets.py youtube_vos --path /data/youtube_vos

# Register Kinetics-700
python scripts/ops/ingest_datasets.py kinetics700 --path /data/kinetics700

# List all registered datasets
python scripts/ops/ingest_datasets.py list

# Show record for one dataset
python scripts/ops/ingest_datasets.py show bdd100k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.medallion_layout import (  # noqa: E402
    DATASET_KEYS,
    default_medallion_root,
    ensure_medallion_dirs,
    load_ingest_record,
    write_ingest_record,
)


# ── Per-dataset ingest probes ─────────────────────────────────────────────────

def _probe_coco(path: Path) -> dict:
    ann_dir = path / "annotations"
    train = (ann_dir / "instances_train2017.json").exists()
    val = (ann_dir / "instances_val2017.json").exists()
    img_train = (path / "train2017").exists()
    img_val = (path / "val2017").exists()
    return {
        "has_train_annotation": train,
        "has_val_annotation": val,
        "has_train_images": img_train,
        "has_val_images": img_val,
        "ready": train or val,
    }


def _probe_bdd100k(path: Path) -> dict:
    has_images = (path / "images").exists() or any(path.glob("**/*.jpg"))
    has_labels = (path / "labels").exists() or any(path.glob("**/*.json"))
    return {"has_images": has_images, "has_labels": has_labels, "ready": has_images}


def _probe_kinetics700(path: Path) -> dict:
    has_videos = any(path.rglob("*.mp4"))
    has_csv = any(path.glob("*.csv"))
    return {"has_videos": has_videos, "has_csv_index": has_csv, "ready": has_videos}


def _probe_youtube8m(path: Path) -> dict:
    has_tfrecord = any(path.rglob("*.tfrecord"))
    has_feature = any(path.rglob("*.pkl")) or any(path.rglob("*.npy"))
    return {"has_tfrecord": has_tfrecord, "has_features": has_feature, "ready": has_tfrecord or has_feature}


def _probe_howto100m(path: Path) -> dict:
    has_videos = any(path.rglob("*.mp4")) or any(path.rglob("*.mkv"))
    has_subtitles = any(path.rglob("*.vtt")) or any(path.rglob("*.srt")) or any(path.rglob("*.json"))
    return {"has_videos": has_videos, "has_subtitles": has_subtitles, "ready": has_videos}


def _probe_webvid10m(path: Path) -> dict:
    has_videos = any(path.rglob("*.mp4"))
    has_captions = any(path.rglob("*.csv")) or any(path.rglob("*.json"))
    return {"has_videos": has_videos, "has_captions": has_captions, "ready": has_videos}


def _probe_epic_kitchens(path: Path) -> dict:
    has_videos = any(path.rglob("*.mp4"))
    has_ann = any(path.rglob("*.csv"))
    return {"has_videos": has_videos, "has_annotation_csv": has_ann, "ready": has_videos}


def _probe_mose(path: Path) -> dict:
    has_jpgs = any(path.rglob("*.jpg")) or any(path.rglob("*.png"))
    has_masks = any(path.rglob("*.png"))
    return {"has_frames": has_jpgs, "has_masks": has_masks, "ready": has_jpgs}


def _probe_youtube_vos(path: Path) -> dict:
    has_frames = any(path.rglob("*.jpg")) or any(path.rglob("*.png"))
    has_ann = (path / "train" / "meta.json").exists() or any(path.rglob("meta.json"))
    return {"has_frames": has_frames, "has_annotation_json": has_ann, "ready": has_frames}


PROBES = {
    "coco": _probe_coco,
    "bdd100k": _probe_bdd100k,
    "kinetics700": _probe_kinetics700,
    "youtube8m": _probe_youtube8m,
    "howto100m": _probe_howto100m,
    "webvid10m": _probe_webvid10m,
    "epic_kitchens": _probe_epic_kitchens,
    "mose": _probe_mose,
    "youtube_vos": _probe_youtube_vos,
}


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        return 1

    mroot = Path(args.medallion_root).resolve()
    ensure_medallion_dirs(mroot, [args.dataset])

    probe_fn = PROBES.get(args.dataset)
    probe_result = probe_fn(path) if probe_fn else {}

    record = {
        "dataset_key": args.dataset,
        "source_path": str(path),
        "medallion_root": str(mroot),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": args.notes or "",
        "probe": probe_result,
    }

    out_path = write_ingest_record(mroot, args.dataset, record)
    print(json.dumps({"status": "ok", "record": str(out_path), "probe": probe_result}, indent=2))
    if not probe_result.get("ready", True):
        print(
            f"\nWarning: probe indicates {args.dataset} at {path} may be incomplete.",
            file=sys.stderr,
        )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    mroot = Path(args.medallion_root).resolve()
    rows = []
    for key in DATASET_KEYS:
        try:
            rec = load_ingest_record(mroot, key)
            rows.append({"dataset": key, "source": rec.get("source_path"), "ready": rec.get("probe", {}).get("ready", "?")})
        except FileNotFoundError:
            rows.append({"dataset": key, "source": None, "ready": False})
    print(json.dumps(rows, indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    mroot = Path(args.medallion_root).resolve()
    try:
        rec = load_ingest_record(mroot, args.dataset)
        print(json.dumps(rec, indent=2))
    except FileNotFoundError:
        print(f"No ingest record for {args.dataset}. Run: python scripts/ops/ingest_datasets.py {args.dataset} --path ...", file=sys.stderr)
        return 1
    return 0


# ── Parser ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--medallion-root", type=Path,
        default=default_medallion_root(REPO),
        help="Medallion root dir (default: datasets/medallion)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    for key in DATASET_KEYS:
        p = sub.add_parser(key, help=f"Register {key} dataset into bronze")
        p.add_argument("--path", required=True, help="Path to raw dataset directory")
        p.add_argument("--notes", default="", help="Optional freeform notes")
        p.set_defaults(func=cmd_ingest, dataset=key)

    p_list = sub.add_parser("list", help="List all registered datasets")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show ingest record for a dataset")
    p_show.add_argument("dataset", choices=DATASET_KEYS)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
