#!/usr/bin/env python3
"""Setup script for COCO dataset train/test/validation splits.

Creates properly split COCO datasets for MaxSight training and evaluation."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.coco_dataset_splitter import (
    split_coco_dataset,
    create_maxsight_splits_from_coco
)


def main():
    parser = argparse.ArgumentParser(
        description='Setup COCO dataset splits for MaxSight training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--coco_dir',
        type=Path,
        required=True,
        help='Root directory containing COCO dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for split annotation files'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['coco', 'maxsight'],
        default='maxsight',
        help='Output format: coco (original) or maxsight (converted)'
    )
    
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.7,
        help='Fraction for training'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.15,
        help='Fraction for validation'
    )
    
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.15,
        help='Fraction for testing'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Optional limit on total samples (for faster iteration)'
    )
    
    parser.add_argument(
        '--train_samples',
        type=int,
        default=None,
        help='Absolute number of training samples (overrides --train_split if set)'
    )
    
    parser.add_argument(
        '--val_samples',
        type=int,
        default=None,
        help='Absolute number of validation samples (overrides --val_split if set)'
    )
    
    parser.add_argument(
        '--annotation_file',
        type=Path,
        default=None,
        help='Specific COCO annotation file (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect annotation file if not provided
    if args.annotation_file is None:
        # Try common COCO annotation locations
        possible_locations = [
            args.coco_dir / 'annotations' / 'instances_train2017.json',
            args.coco_dir / 'annotations' / 'instances_val2017.json',
            args.coco_dir / 'instances_train2017.json',
            args.coco_dir / 'instances_val2017.json',
        ]
        
        for loc in possible_locations:
            if loc.exists():
                args.annotation_file = loc
                print(f"Auto-detected annotation file: {args.annotation_file}")
                break
        
        if args.annotation_file is None:
            print("Error: Could not find COCO annotation file.")
            print("Please specify --annotation_file or ensure COCO dataset is properly structured.")
            sys.exit(1)
    
    # Auto-detect image directory
    image_dir = args.coco_dir / 'train2017'
    if not image_dir.exists():
        image_dir = args.coco_dir / 'val2017'
    if not image_dir.exists():
        image_dir = args.coco_dir / 'images'
    
    if not image_dir.exists():
        print(f"Warning: Could not find image directory. Using: {image_dir}")
    
    print("="*60)
    print("COCO Dataset Split Setup")
    print("="*60)
    print(f"COCO directory: {args.coco_dir}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Format: {args.format}")
    if args.train_samples is not None and args.val_samples is not None:
        print(f"Splits: Train={args.train_samples:,} samples, Val={args.val_samples:,} samples")
    else:
        print(f"Splits: Train={args.train_split:.1%}, Val={args.val_split:.1%}, Test={args.test_split:.1%}")
    print("="*60)
    print()
    
    # Create splits
    try:
        if args.format == 'coco':
            train_file, val_file, test_file = split_coco_dataset(
                coco_annotation_file=args.annotation_file,
                output_dir=args.output_dir,
                train_split=args.train_split,
                val_split=args.val_split,
                test_split=args.test_split,
                seed=args.seed
            )
        else:
            train_file, val_file, test_file = create_maxsight_splits_from_coco(
                coco_annotation_file=args.annotation_file,
                image_dir=image_dir,
                output_dir=args.output_dir,
                train_split=args.train_split if args.train_samples is None else None,
                val_split=args.val_split if args.val_samples is None else None,
                test_split=args.test_split if args.train_samples is None else None,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                seed=args.seed,
                num_samples=args.num_samples
            )
        
        print()
        print(f"Train: {train_file}")
        print(f"Val:   {val_file}")
        print(f"Test:  {test_file}")
        print()
        print("Next steps:")
        print("1. Use these annotation files with MaxSightDataset")
        print("2. For inference, use inference datasets (see ml/data/inference_datasets.py)")
        print("   - Open Images V6: Broad semantic diversity")
        print("   - BDD100K: Motion / outdoor / hazard realism")
        print("   - ADE20K: Indoor structure & objects")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
