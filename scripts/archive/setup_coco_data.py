#!/usr/bin/env python3
"""Setup COCO dataset - Extract zips and verify structure.
Handles both coco_raw and coco directories."""

import sys
import zipfile
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.download_datasets import verify_coco_dataset


def extract_zip(zip_path: Path, extract_to: Path, description: str) -> bool:
    """Extract zip file with progress."""
    if not zip_path.exists():
        print(f"❌ {description} zip not found: {zip_path}")
        return False
    
    extract_dir = extract_to.parent
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {description}...")
    print(f"  From: {zip_path}")
    print(f"  To: {extract_to.parent}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"  Files: {total_files:,}")
            
            # Extract with progress
            for i, member in enumerate(file_list):
                if (i + 1) % 1000 == 0 or i == 0:
                    print(f"  Progress: {i+1:,}/{total_files:,} files", end='\r')
                zip_ref.extract(member, extract_to.parent)
            
            print(f"\n✅ {description} extracted successfully")
            return True
    except Exception as e:
        print(f"❌ Failed to extract {description}: {e}")
        return False


def setup_coco_data(data_dir: Path = None) -> bool:
    """Setup COCO dataset by extracting zips and verifying."""
    # Auto-detect location
    if data_dir is None:
        if Path('datasets/coco_raw').exists():
            data_dir = Path('datasets/coco_raw')
        elif Path('datasets/coco').exists():
            data_dir = Path('datasets/coco')
        else:
            data_dir = Path('datasets/coco')
            data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COCO Dataset Setup")
    print("="*70)
    print(f"Data directory: {data_dir}\n")
    
    # Check what we have
    val_zip = data_dir / "val2017.zip"
    train_zip = data_dir / "train2017.zip"
    ann_zip = data_dir / "annotations_trainval2017.zip"
    
    val_dir = data_dir / "val2017"
    train_dir = data_dir / "train2017"
    ann_dir = data_dir / "annotations"
    
    extracted = False
    
    # Extract val images if zip exists but directory doesn't
    if val_zip.exists() and not val_dir.exists():
        if extract_zip(val_zip, val_dir, "Val images"):
            extracted = True
    
    # Extract train images if zip exists but directory doesn't
    if train_zip.exists() and not train_dir.exists():
        if extract_zip(train_zip, train_dir, "Train images"):
            extracted = True
    
    # Extract annotations if zip exists but directory is incomplete
    if ann_zip.exists():
        if not ann_dir.exists() or len(list(ann_dir.glob("*.json"))) < 4:
            if extract_zip(ann_zip, ann_dir, "Annotations"):
                extracted = True
    
    if extracted:
        print("\n✅ Extraction complete!")
    else:
        print("\nℹ️  No extraction needed (directories already exist or zips missing)")
    
    # Verify dataset
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    status = verify_coco_dataset(data_dir, check_coco_raw=False)
    
    all_good = all(status.values())
    
    if all_good:
        print("\n✅ COCO dataset is complete and ready for training!")
        return True
    else:
        print("\n⚠️  COCO dataset is incomplete:")
        for key, value in status.items():
            status_str = "✅" if value else "❌"
            print(f"  {status_str} {key}")
        
        # Provide specific instructions
        if not status['train_images']:
            print("\n📥 To download train images:")
            print("  1. Visit: http://images.cocodataset.org/zips/train2017.zip")
            print("  2. Download (~18GB) to:", data_dir)
            print("  3. Run this script again to extract")
            print("  Or use: python scripts/download_coco.py --auto")
        
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup COCO dataset')
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=None,
        help='COCO dataset directory (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    success = setup_coco_data(args.data_dir)
    sys.exit(0 if success else 1)

