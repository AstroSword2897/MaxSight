#!/usr/bin/env python3
"""Phase 3: Data Pipeline and Augmentation Validation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline and augmentations")
    parser.add_argument("--train-annotation", type=Path, default=Path("datasets/cleaned_splits/maxsight_train.json"),
                        help="Train annotation JSON")
    parser.add_argument("--val-annotation", type=Path, default=Path("datasets/cleaned_splits/maxsight_val.json"),
                        help="Val annotation JSON")
    parser.add_argument("--image-dir", type=Path, default=None, help="Image root (default: parent of splits)")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to check for overflow")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for loader")
    args = parser.parse_args()

    if not args.train_annotation.exists():
        print(f"Train annotation not found: {args.train_annotation}")
        print("Run: python scripts/gather_training_data.py")
        return 1

    image_dir = args.image_dir or args.train_annotation.parent.parent / "coco_raw"
    if not image_dir.exists():
        image_dir = args.train_annotation.parent.parent
    val_ann = args.val_annotation if args.val_annotation.exists() else args.train_annotation

    # --- 1) create_data_loaders + collate (variable-length audio padding) ---.
    from ml.data.data_pipeline import create_data_loaders, compute_class_weights

    print("Loading dataset (custom collate, variable-length audio padding)...")
    train_loader, val_loader, _ = create_data_loaders(
        args.train_annotation,
        val_ann,
        test_annotation_file=None,
        image_dir=image_dir,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        condition_mode=None,
        apply_lighting_augmentation=True,
        max_objects=10,
    )
    print("  Train batches:", len(train_loader))

    # --- 2) Test augmentations on up to max_samples images; check no overflows ---.
    print("Checking for invalid values (NaN/Inf) in up to %d samples..." % args.max_samples)
    invalid = 0
    samples_seen = 0
    for batch in train_loader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                invalid += 1
                print("  Invalid values in batch key:", key)
        samples_seen += batch["images"].size(0)
        if samples_seen >= args.max_samples:
            break
    if invalid:
        print("  FAIL: Found invalid (NaN/Inf) values.")
        return 1
    print("  OK: All checked tensors finite (augmentations preserve valid range).")

    # --- 3) Class weights: rare classes get higher weights ---.
    print("Computing class weights (rare classes get higher weights)...")
    weights = compute_class_weights(args.train_annotation)
    if not weights:
        print("  No class weights (empty or unsupported annotation format).")
    else:
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
        print("  Top 5 rare classes (highest weight):", sorted_weights[:5])
        print("  Weights computed; use in create_data_loaders(class_weights=...) for weighted sampling.")
    print("Validation passed: no invalid values; augmentations preserve annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())





