#!/usr/bin/env python3
"""
Setup training data pipeline for MaxSight.

Creates train/val/test splits from COCO and sets up data loaders.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.coco_dataset_splitter import create_maxsight_splits_from_coco
from ml.data.data_pipeline import create_data_loaders, get_data_info, compute_class_weights
from ml.data.download_datasets import verify_coco_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Setup training data pipeline for MaxSight',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--coco_dir',
        type=Path,
        default=None,
        help='COCO dataset directory (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('datasets/cleaned_splits'),
        help='Output directory for split annotation files'
    )
    
    parser.add_argument(
        '--train_samples',
        type=int,
        default=10000,
        help='Number of training samples'
    )
    
    parser.add_argument(
        '--val_samples',
        type=int,
        default=2000,
        help='Number of validation samples'
    )
    
    parser.add_argument(
        '--test_samples',
        type=int,
        default=1000,
        help='Number of test samples'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing splits, do not create new ones'
    )
    
    parser.add_argument(
        '--test-loaders',
        action='store_true',
        help='Test data loaders after setup'
    )
    
    args = parser.parse_args()
    
    # Auto-detect COCO directory
    if args.coco_dir is None:
        if Path('datasets/coco_raw').exists():
            args.coco_dir = Path('datasets/coco_raw')
        elif Path('datasets/coco').exists():
            args.coco_dir = Path('datasets/coco')
        else:
            print("Error: COCO dataset not found. Please specify --coco_dir")
            print("Or download COCO first: python scripts/download_coco.py --auto")
            sys.exit(1)
    
    print("="*70)
    print("MaxSight Training Data Setup")
    print("="*70)
    print(f"COCO directory: {args.coco_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples: Train={args.train_samples:,}, Val={args.val_samples:,}, Test={args.test_samples:,}")
    print("="*70)
    print()
    
    # Verify COCO dataset
    print("Verifying COCO dataset...")
    status = verify_coco_dataset(args.coco_dir, check_coco_raw=False)
    
    if not all([status['val_images'], status['annotations'], status['val_annotations']]):
        print("\n⚠️  COCO dataset is incomplete. Missing components:")
        for key, value in status.items():
            if not value:
                print(f"  ❌ {key}")
        print("\nPlease run: python scripts/setup_coco_data.py")
        sys.exit(1)
    
    print("✅ COCO dataset verified")
    print()
    
    # Check if splits already exist
    output_dir = args.output_dir
    train_file = output_dir / 'train.json'
    val_file = output_dir / 'val.json'
    test_file = output_dir / 'test.json'
    
    if args.verify_only:
        if all(f.exists() for f in [train_file, val_file, test_file]):
            print("✅ Training splits already exist:")
            print(f"  Train: {train_file}")
            print(f"  Val:   {val_file}")
            print(f"  Test:  {test_file}")
            
            if args.test_loaders:
                print("\nTesting data loaders...")
                test_data_loaders(train_file, val_file, test_file, args.coco_dir)
            return
        else:
            print("❌ Training splits not found. Run without --verify-only to create them.")
            sys.exit(1)
    
    # Create splits
    print("Creating MaxSight splits from COCO...")
    
    # Find annotation file
    ann_file = args.coco_dir / 'annotations' / 'instances_train2017.json'
    if not ann_file.exists():
        ann_file = args.coco_dir / 'annotations' / 'instances_val2017.json'
    
    if not ann_file.exists():
        print(f"Error: Could not find COCO annotation file in {args.coco_dir}")
        sys.exit(1)
    
    # Find image directory
    image_dir = args.coco_dir / 'train2017'
    if not image_dir.exists():
        image_dir = args.coco_dir / 'val2017'
    
    if not image_dir.exists():
        print(f"Warning: Image directory not found: {image_dir}")
        image_dir = args.coco_dir
    
    try:
        train_file, val_file, test_file = create_maxsight_splits_from_coco(
            coco_annotation_file=ann_file,
            image_dir=image_dir,
            output_dir=output_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed,
            num_samples=args.train_samples + args.val_samples + args.test_samples
        )
        
        print()
        print("✅ Splits created successfully:")
        print(f"  Train: {train_file}")
        print(f"  Val:   {val_file}")
        print(f"  Test:  {test_file}")
        
        # Compute class weights
        print("\nComputing class weights...")
        class_weights = compute_class_weights(train_file)
        print(f"  Found {len(class_weights)} classes")
        
        # Test data loaders if requested
        if args.test_loaders:
            print("\nTesting data loaders...")
            test_data_loaders(train_file, val_file, test_file, args.coco_dir)
        
        print("\n✅ Training data pipeline ready!")
        print("\nNext steps:")
        print("1. Review training configs in ml/training/configs/")
        print("2. Run training: python scripts/train_maxsight.py --config <config_file>")
        
    except Exception as e:
        print(f"\n❌ Error creating splits: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_data_loaders(train_file: Path, val_file: Path, test_file: Path, coco_dir: Path):
    """Test that data loaders can be created and batches can be loaded."""
    try:
        from ml.data.data_pipeline import create_data_loaders
        
        print("  Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_annotation_file=train_file,
            val_annotation_file=val_file,
            test_annotation_file=test_file,
            image_dir=coco_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            condition_mode=None,
            apply_lighting_augmentation=False
        )
        
        print("  ✅ Data loaders created")
        
        # Get info
        train_info = get_data_info(train_loader)
        val_info = get_data_info(val_loader)
        
        print(f"  Train: {train_info['dataset_size']} samples, {train_info['num_batches']} batches")
        print(f"  Val:   {val_info['dataset_size']} samples, {val_info['num_batches']} batches")
        
        # Try loading a batch
        print("  Loading sample batch...")
        batch = next(iter(train_loader))
        print(f"  ✅ Batch loaded: {list(batch.keys())}")
        print(f"     Images shape: {batch['images'].shape}")
        print(f"     Labels shape: {batch['labels'].shape}")
        print(f"     Boxes shape:  {batch['boxes'].shape}")
        
    except Exception as e:
        print(f"  ❌ Error testing loaders: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

