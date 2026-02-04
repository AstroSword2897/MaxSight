#!/usr/bin/env python3
"""
Gather all data required for MaxSight training and AutoML.

Runs in order: download COCO (optional), extract zips, create train/val/test splits.
Use this once to satisfy data requirements for train_maxsight.py and tune_hyperparameters.py.
Works on x86_64 and arm64 (Apple Silicon); extraction and splits are platform-agnostic.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command; return completed process."""
    return subprocess.run(cmd, cwd=cwd, check=check)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gather COCO data and create training splits for MaxSight"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "datasets" / "coco_raw",
        help="COCO data directory (default: datasets/coco_raw)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "datasets" / "cleaned_splits",
        help="Output directory for train/val/test JSON (default: datasets/cleaned_splits)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing zips or extracted data)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extract step (already extracted)",
    )
    parser.add_argument(
        "--download-auto",
        action="store_true",
        help="Run download with --auto to attempt automatic download",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=10000,
        help="Max training samples for splits (default: 10000)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2000,
        help="Max validation samples (default: 2000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="Max test samples (default: 1000)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    splits_dir = args.splits_dir.resolve()

    print("=" * 70)
    print("MaxSight: Gather training data (x86_64 / arm64)")
    print("=" * 70)
    print(f"Data dir:   {data_dir}")
    print(f"Splits dir: {splits_dir}")
    print()

    # 1. Download (optional)
    if not args.skip_download:
        print("Step 1: Download COCO (if missing)...")
        try:
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "download_coco.py"),
                    "--data_dir", str(data_dir),
                    *(["--auto"] if args.download_auto else []),
                ],
                check=False,
            )
        except Exception as e:
            print(f"  Warning: download step failed: {e}")
            print("  You can run manually: python scripts/download_coco.py --data_dir ... [--auto]")
        print()
    else:
        print("Step 1: Skip download (--skip-download)")
        print()

    # 2. Extract
    if not args.skip_extract:
        print("Step 2: Extract COCO zips...")
        extract_script = ROOT / "scripts" / "extract_coco.py"
        if not extract_script.exists():
            print("  Warning: scripts/extract_coco.py not found")
        else:
            # extract_coco.py uses hardcoded datasets/coco_raw; we need to support custom dir
            from ml.data.download_datasets import verify_coco_dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            # Run extract in project root; extract_coco expects datasets/coco_raw
            if data_dir == ROOT / "datasets" / "coco_raw":
                try:
                    run([sys.executable, str(extract_script)], check=False)
                except Exception as e:
                    print(f"  Warning: extract failed: {e}")
            else:
                print(f"  Note: extract_coco.py uses datasets/coco_raw. If your data is elsewhere, run:")
                print(f"        python scripts/extract_coco.py  # then copy/move to {data_dir}")
        print()
    else:
        print("Step 2: Skip extract (--skip-extract)")
        print()

    # 3. Setup splits (create_maxsight_splits_from_coco)
    print("Step 3: Create train/val/test splits...")
    from ml.data.download_datasets import verify_coco_dataset
    status = verify_coco_dataset(data_dir, check_coco_raw=(data_dir.name == "coco_raw"))
    if not (status.get("train_images") or status.get("val_images")) or not status.get("annotations"):
        print("  COCO data missing or incomplete. Run download and extract first.")
        print("  Verify: python scripts/download_coco.py --verify-only --data_dir", data_dir)
        return 1

    from ml.data.coco_dataset_splitter import create_maxsight_splits_from_coco

    ann_file = data_dir / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        ann_file = data_dir / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        print("  Error: No COCO annotation file found under", data_dir)
        return 1

    image_dir = data_dir / "train2017"
    if not image_dir.exists():
        image_dir = data_dir / "val2017"
    if not image_dir.exists():
        image_dir = data_dir

    splits_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_file, val_file, test_file = create_maxsight_splits_from_coco(
            coco_annotation_file=ann_file,
            image_dir=image_dir,
            output_dir=splits_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=42,
            num_samples=args.train_samples + args.val_samples + args.test_samples,
        )
    except Exception as e:
        print(f"  Error creating splits: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("Data ready")
    print("=" * 70)
    print("Splits (MaxSight format):")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")
    print()
    print("Training (use annotation files and image dir):")
    print(f"  python scripts/train_maxsight.py \\")
    print(f"    --data-dir {data_dir} \\")
    print(f"    --train-annotation {train_file} \\")
    print(f"    --val-annotation {val_file} \\")
    print(f"    --image-dir {image_dir} \\")
    print(f"    --epochs 2 --device cpu")
    print()
    print("AutoML (Optuna) after data is ready:")
    print(f"  python scripts/tune_hyperparameters.py \\")
    print(f"    --data-dir {data_dir} \\")
    print(f"    --train-annotation {train_file} \\")
    print(f"    --val-annotation {val_file} \\")
    print(f"    --image-dir {image_dir} \\")
    print(f"    --n-trials 5 --epochs-per-trial 2 --device cpu")
    print()
    print("arm64 (Apple Silicon): use --device mps for inference/benchmarks;")
    print("  use --device cpu for training if MPS has unsupported ops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
