#!/usr/bin/env python3
"""Extract COCO dataset zip files and verify extraction."""

import sys
import zipfile
from pathlib import Path
import shutil

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.download_datasets import verify_coco_dataset


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file to directory."""
    print(f"Extracting {zip_path.name}...")
    print(f"  Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  Destination: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            print(f"  Files: {total_files}")
            
            # Extract with progress.
            extracted = 0
            for member in zip_ref.namelist():
                zip_ref.extract(member, extract_to)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f"  Progress: {extracted}/{total_files} files ({100*extracted/total_files:.1f}%)")
            
            print(f"  [ok] Extracted {extracted} files")
            return True
    except zipfile.BadZipFile:
        print(f"  [fail] Error: {zip_path.name} is corrupted or incomplete")
        return False
    except Exception as e:
        print(f"  [fail] Error: {e}")
        return False


def main():
    data_dir = Path("datasets/coco_raw")
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return 1
    
    print("="*70)
    print("COCO Dataset Extraction")
    print("="*70)
    print(f"\nData directory: {data_dir}\n")
    
    # Check what needs to be extracted.
    zip_files = {
        'train2017.zip': data_dir / 'train2017.zip',
        'val2017.zip': data_dir / 'val2017.zip',
        'annotations_trainval2017.zip': data_dir / 'annotations_trainval2017.zip'
    }
    
    # Check existing directories.
    train_dir = data_dir / 'train2017'
    val_dir = data_dir / 'val2017'
    ann_dir = data_dir / 'annotations'
    
    extracted = []
    to_extract = []
    
    # Check train images.
    if train_dir.exists() and len(list(train_dir.glob("*.jpg"))) > 100000:
        print("[ok] train2017 already extracted")
        extracted.append('train2017')
    elif zip_files['train2017.zip'].exists():
        to_extract.append(('train2017.zip', zip_files['train2017.zip'], data_dir))
    else:
        print("WARNING train2017.zip not found")
    
    # Check val images.
    if val_dir.exists() and len(list(val_dir.glob("*.jpg"))) > 4000:
        print("[ok] val2017 already extracted")
        extracted.append('val2017')
    elif zip_files['val2017.zip'].exists():
        to_extract.append(('val2017.zip', zip_files['val2017.zip'], data_dir))
    else:
        print("WARNING val2017.zip not found")
    
    # Check annotations.
    if ann_dir.exists() and len(list(ann_dir.glob("*.json"))) >= 5:
        print("[ok] annotations already extracted")
        extracted.append('annotations')
    elif zip_files['annotations_trainval2017.zip'].exists():
        to_extract.append(('annotations_trainval2017.zip', zip_files['annotations_trainval2017.zip'], data_dir))
    else:
        print("WARNING annotations_trainval2017.zip not found")
    
    if not to_extract:
        print("\n[ok] All files already extracted!")
    else:
        print(f"\nExtracting {len(to_extract)} file(s)...\n")
        for name, zip_path, extract_to in to_extract:
            if not extract_zip(zip_path, extract_to):
                print(f"\n[fail] Failed to extract {name}")
                return 1
            print()
    
    # Verify extraction.
    print("="*70)
    print("Verification")
    print("="*70)
    status = verify_coco_dataset(data_dir, check_coco_raw=True)
    
    if all(status.values()):
        print("\nOK COCO dataset is complete and ready for training!")
        return 0
    else:
        print("\nWARNING  Dataset is incomplete:")
        for key, value in status.items():
            status_str = "OK" if value else "FAIL"
            print(f"  {status_str} {key}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


