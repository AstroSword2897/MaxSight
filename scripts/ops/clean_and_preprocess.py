#!/usr/bin/env python3
"""Bronze → silver: clean and preprocess every registered dataset.

Reads ingest records from bronze, runs cleaning + preprocessing per dataset,
and writes results to silver.  Saves a per-run report JSON under gold/.

Usage
-----
# All registered datasets
python scripts/ops/clean_and_preprocess.py all

# Single dataset
python scripts/ops/clean_and_preprocess.py coco
python scripts/ops/clean_and_preprocess.py bdd100k
python scripts/ops/clean_and_preprocess.py mose
python scripts/ops/clean_and_preprocess.py youtube_vos

# Video datasets (need opencv-python)
python scripts/ops/clean_and_preprocess.py kinetics700 --video-fps 1.0

# Force re-run (overwrite existing silver files)
python scripts/ops/clean_and_preprocess.py coco --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.dataset_cleaning import (  # noqa: E402
    clean_bdd100k,
    clean_coco,
    clean_video_dataset,
    clean_vos_dataset,
)
from ml.data.dataset_preprocessing import (  # noqa: E402
    ImagePreprocessingPipeline,
    VideoFrameExtractor,
    adapt_bdd100k_to_coco,
    build_vos_coco_annotation,
)
from ml.data.medallion_layout import (  # noqa: E402
    DATASET_KEYS,
    default_medallion_root,
    ensure_medallion_dirs,
    gold_dir,
    load_ingest_record,
    silver_annotations_dir,
    silver_dataset_dir,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Per-dataset pipelines ─────────────────────────────────────────────────────


def pipeline_coco(
    mroot: Path,
    source_path: Path,
    target_size: tuple,
    force: bool,
) -> list[dict[str, Any]]:
    silver = silver_dataset_dir(mroot, "coco")
    reports_raw = clean_coco(source_path, silver)
    pipe = ImagePreprocessingPipeline(target_size=target_size, dataset_key="coco")
    rpt = pipe.run(silver / "images", silver / "images_resized", force=force)
    return [r.to_dict() for r in reports_raw] + [rpt.to_dict()]


def pipeline_bdd100k(
    mroot: Path,
    source_path: Path,
    target_size: tuple,
    force: bool,
) -> list[dict[str, Any]]:
    silver = silver_dataset_dir(mroot, "bdd100k")
    reports_raw = clean_bdd100k(source_path, silver)
    pipe = ImagePreprocessingPipeline(target_size=target_size, dataset_key="bdd100k")
    rpt = pipe.run(silver / "images", silver / "images_resized", force=force)

    ann_dir = silver / "annotations"
    coco_ann_dir = silver_annotations_dir(mroot, "bdd100k")
    coco_ann_dir.mkdir(parents=True, exist_ok=True)
    for ann_file in ann_dir.glob("*.json"):
        try:
            adapt_bdd100k_to_coco(ann_file, coco_ann_dir / ("coco_" + ann_file.name))
        except Exception as exc:
            logger.warning("BDD100K adapter skipped %s: %s", ann_file.name, exc)

    return [r.to_dict() for r in reports_raw] + [rpt.to_dict()]


def pipeline_video(
    mroot: Path,
    source_path: Path,
    dataset_key: str,
    target_size: tuple,
    fps: float,
    force: bool,
) -> list[dict[str, Any]]:
    silver = silver_dataset_dir(mroot, dataset_key)
    reports_raw = clean_video_dataset(source_path, silver, dataset_key=dataset_key)
    extractor = VideoFrameExtractor(fps=fps, target_size=target_size, dataset_key=dataset_key)
    rpt = extractor.run_batch(silver / "videos", silver / "frames", force=force)
    return [r.to_dict() for r in reports_raw] + [rpt.to_dict()]


def pipeline_vos(
    mroot: Path,
    source_path: Path,
    dataset_key: str,
    target_size: tuple,
    force: bool,
) -> list[dict[str, Any]]:
    silver = silver_dataset_dir(mroot, dataset_key)
    reports_raw = clean_vos_dataset(source_path, silver, dataset_key=dataset_key)
    pipe = ImagePreprocessingPipeline(target_size=target_size, dataset_key=dataset_key)
    rpt = pipe.run(silver / "frames", silver / "frames_resized", force=force)
    coco_path = silver_annotations_dir(mroot, dataset_key) / "scaffold.json"
    try:
        build_vos_coco_annotation(silver / "frames", None, coco_path, dataset_key=dataset_key)
    except Exception as exc:
        logger.warning("VOS scaffold failed for %s: %s", dataset_key, exc)
    return [r.to_dict() for r in reports_raw] + [rpt.to_dict()]


PIPELINE_MAP = {
    "coco": lambda mroot, src, sz, fps, force: pipeline_coco(mroot, src, sz, force),
    "bdd100k": lambda mroot, src, sz, fps, force: pipeline_bdd100k(mroot, src, sz, force),
    "kinetics700": lambda mroot, src, sz, fps, force: pipeline_video(
        mroot, src, "kinetics700", sz, fps, force
    ),
    "youtube8m": lambda mroot, src, sz, fps, force: pipeline_video(
        mroot, src, "youtube8m", sz, fps, force
    ),
    "howto100m": lambda mroot, src, sz, fps, force: pipeline_video(
        mroot, src, "howto100m", sz, fps, force
    ),
    "webvid10m": lambda mroot, src, sz, fps, force: pipeline_video(
        mroot, src, "webvid10m", sz, fps, force
    ),
    "epic_kitchens": lambda mroot, src, sz, fps, force: pipeline_video(
        mroot, src, "epic_kitchens", sz, fps, force
    ),
    "mose": lambda mroot, src, sz, fps, force: pipeline_vos(mroot, src, "mose", sz, force),
    "youtube_vos": lambda mroot, src, sz, fps, force: pipeline_vos(
        mroot, src, "youtube_vos", sz, force
    ),
}


# ── Run entry ─────────────────────────────────────────────────────────────────


def run_dataset(
    dataset_key: str,
    mroot: Path,
    target_size: tuple,
    fps: float,
    force: bool,
) -> dict[str, Any]:
    try:
        rec = load_ingest_record(mroot, dataset_key)
    except FileNotFoundError:
        logger.warning("No ingest record for %s — run ingest_datasets.py first", dataset_key)
        return {"dataset": dataset_key, "status": "skipped", "reason": "no ingest record"}

    source_path = Path(rec["source_path"])
    if not source_path.exists():
        logger.error("Source path missing for %s: %s", dataset_key, source_path)
        return {
            "dataset": dataset_key,
            "status": "error",
            "reason": f"source missing: {source_path}",
        }

    pipeline_fn = PIPELINE_MAP[dataset_key]
    t0 = time.perf_counter()
    try:
        reports = pipeline_fn(mroot, source_path, target_size, fps, force)
        elapsed = time.perf_counter() - t0
        logger.info("Done [%s] in %.1fs", dataset_key, elapsed)
        return {
            "dataset": dataset_key,
            "status": "ok",
            "elapsed_s": round(elapsed, 2),
            "reports": reports,
        }
    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", dataset_key, exc, exc_info=True)
        return {"dataset": dataset_key, "status": "error", "reason": str(exc)}


def save_run_report(mroot: Path, results: list[dict[str, Any]]) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = gold_dir(mroot) / f"clean_preprocess_run_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_at": ts, "results": results}, indent=2), encoding="utf-8")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--medallion-root",
        type=Path,
        default=default_medallion_root(REPO),
        help="Medallion root (default: datasets/medallion)",
    )
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"))
    parser.add_argument(
        "--video-fps", type=float, default=1.0, help="FPS for frame extraction (video datasets)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing silver files")
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=DATASET_KEYS + ["all"],
        help="Dataset key(s) to process, or 'all'",
    )
    args = parser.parse_args()

    mroot = args.medallion_root.resolve()
    ensure_medallion_dirs(mroot)
    target_size = tuple(args.target_size)  # (W, H)
    keys = DATASET_KEYS if "all" in args.datasets else args.datasets

    results = []
    for key in keys:
        logger.info("=== Processing: %s ===", key)
        result = run_dataset(key, mroot, target_size, args.video_fps, args.force)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))

    report_path = save_run_report(mroot, results)
    print(f"\nFull report: {report_path}")

    errors = [r for r in results if r.get("status") == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
