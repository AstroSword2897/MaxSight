#!/usr/bin/env python3
"""Create canonical raw-data dirs and medallion layout; optionally gather COCO and ingest.

Vendor datasets (Kinetics, YouTube-8M, etc.) must be obtained manually; this script
keeps paths stable under ``datasets/coco_raw`` and ``datasets/raw/<key>/``.

Usage
-----
  python scripts/ops/ensure_medallion_dataset_paths.py
  python scripts/ops/ensure_medallion_dataset_paths.py --gather-coco
  python scripts/ops/ensure_medallion_dataset_paths.py --ingest-nonempty
  python scripts/ops/ensure_medallion_dataset_paths.py --gather-coco --skip-download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.medallion_layout import (  # noqa: E402
    DATASET_KEYS,
    all_default_raw_dataset_dirs,
    bronze_video_dir,
    default_medallion_root,
    default_raw_dataset_dir,
    ensure_medallion_dirs,
)

# Pointers only; large or gated corpora are not fetched by this repo.
ACQUISITION_HINTS: dict[str, str] = {
    "coco": "http://cocodataset.org — or scripts/ops/gather_training_data.py",
    "kinetics700": "https://deepmind.google/datasets/kinetics/",
    "youtube8m": "https://research.google.com/youtube8m/",
    "howto100m": "https://www.di.ens.fr/willow/research/howto100m/",
    "webvid10m": "WebVid-10M project / paper (mirrors vary)",
    "bdd100k": "https://bdd-data.berkeley.edu/",
    "epic_kitchens": "https://epic-kitchens.github.io/",
    "mose": "MOSE project page (video object segmentation)",
    "youtube_vos": "https://youtube-vos.org/",
}


def _dir_nonempty(p: Path) -> bool:
    if not p.is_dir():
        return False
    try:
        next(p.iterdir())
    except StopIteration:
        return False
    return True


def _run_ingest(medallion_root: Path, key: str, path: Path) -> int:
    ingest = REPO / "scripts" / "ops" / "ingest_datasets.py"
    cmd = [
        sys.executable,
        str(ingest),
        key,
        "--path",
        str(path.resolve()),
        "--medallion-root",
        str(medallion_root.resolve()),
    ]
    print(" ", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--medallion-root",
        type=Path,
        default=None,
        help="Medallion root (default: <repo>/datasets/medallion)",
    )
    parser.add_argument(
        "--gather-coco",
        action="store_true",
        help="Run scripts/ops/gather_training_data.py with default COCO data dir.",
    )
    parser.add_argument(
        "--ingest-nonempty",
        action="store_true",
        help="Run ingest_datasets.py for each canonical path that already has files.",
    )
    args, gather_rest = parser.parse_known_args()
    repo = args.repo_root.resolve()
    mroot = args.medallion_root.resolve() if args.medallion_root else default_medallion_root(repo)

    paths = all_default_raw_dataset_dirs(repo)
    for key, p in paths.items():
        p.mkdir(parents=True, exist_ok=True)
    ensure_medallion_dirs(mroot)
    bronze_video_dir(mroot).mkdir(parents=True, exist_ok=True)

    print("Canonical raw dataset directories:")
    for key in DATASET_KEYS:
        print(f"  {key:16} {paths[key]}")
    print()
    print("Medallion root:", mroot)
    print("Acquire data (manual except COCO automation below):")
    for key in DATASET_KEYS:
        print(f"  {key}: {ACQUISITION_HINTS.get(key, 'see docs/video_and_navigation_datasets.md')}")
    print()

    if gather_rest and not args.gather_coco:
        print("Unknown arguments (use --gather-coco to forward extras to gather_training_data.py):", gather_rest, file=sys.stderr)
        return 2

    if args.gather_coco:
        gather = repo / "scripts" / "ops" / "gather_training_data.py"
        data_dir = default_raw_dataset_dir(repo, "coco")
        extra = list(gather_rest)
        while extra and extra[0] == "--":
            extra.pop(0)
        cmd = [sys.executable, str(gather), "--data-dir", str(data_dir), *extra]
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=repo)
        if rc != 0:
            return rc

    if args.ingest_nonempty:
        print("ingest (non-empty paths only):")
        any_ingest = False
        for key in DATASET_KEYS:
            p = paths[key]
            if not _dir_nonempty(p):
                continue
            any_ingest = True
            rc = _run_ingest(mroot, key, p)
            if rc != 0:
                return rc
        if not any_ingest:
            print("  (no non-empty canonical dirs; populate paths then re-run with --ingest-nonempty)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
