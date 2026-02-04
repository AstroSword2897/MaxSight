#!/usr/bin/env python3
"""
Helper script to download COCO dataset with multiple fallback methods.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.download_datasets import download_coco_dataset, verify_coco_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Download COCO dataset for MaxSight training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=None,  # Will auto-detect coco_raw or use datasets/coco
        help='Directory to save/check COCO dataset (default: auto-detect)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Attempt automatic download (tries wget, curl, then requests)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset, do not download'
    )
    
    args = parser.parse_args()
    
    # Auto-detect data directory
    if args.data_dir is None:
        # Check common locations
        if Path('datasets/coco_raw').exists():
            args.data_dir = Path('datasets/coco_raw')
            print(f"Auto-detected COCO dataset at: {args.data_dir}")
        elif Path('datasets/coco').exists():
            args.data_dir = Path('datasets/coco')
            print(f"Auto-detected COCO dataset at: {args.data_dir}")
        else:
            args.data_dir = Path('datasets/coco')
            print(f"Using default directory: {args.data_dir}")
    
    if args.verify_only:
        print("Verifying COCO dataset...")
        status = verify_coco_dataset(args.data_dir, check_coco_raw=True)
        
        if all(status.values()):
            print("\n✅ COCO dataset is complete and verified!")
            return 0
        else:
            print("\n⚠️  COCO dataset is incomplete:")
            for key, value in status.items():
                status_str = "✅" if value else "❌"
                print(f"  {status_str} {key}")
            print("\nRun with --auto to download missing components.")
            return 1
    else:
        print("="*70)
        print("COCO Dataset Download for MaxSight")
        print("="*70)
        print(f"\nTarget directory: {args.data_dir}")
        print(f"Auto-download: {args.auto}")
        print()
        
        download_coco_dataset(data_dir=args.data_dir, auto_download=args.auto)
        
        # Verify after download attempt
        print("\n" + "="*70)
        print("Verification")
        print("="*70)
        status = verify_coco_dataset(args.data_dir)
        
        if all(status.values()):
            print("\n✅ COCO dataset is complete and ready for training!")
            return 0
        else:
            print("\n⚠️  Some components are missing. Please download manually if needed.")
            return 1


if __name__ == "__main__":
    sys.exit(main())

