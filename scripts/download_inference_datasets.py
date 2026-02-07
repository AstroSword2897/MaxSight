#!/usr/bin/env python3
"""Download all inference datasets for MaxSight evaluation.

Downloads:
- Open Images V6 (validation set for inference)
- BDD100K (validation set for inference)
- ADE20K (validation set for inference)"""

import sys
import argparse
import subprocess
import shutil
import zipfile
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def download_file(url: str, dest: Path, resume: bool = True) -> bool:
    """Download a file with progress bar and resume capability...."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Checks if file already exists
    if dest.exists():
        print(f"  ✓ File already exists: {dest}")
        return True
    
    # Uses curl when available (supports resume)
    if shutil.which('curl'):
        try:
            cmd = ['curl', '-L', '-C', '-', '--progress-bar', '--retry', '3', '-o', str(dest), url]
            result = subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            pass
    
    # Fallback to requests with progress
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        mode = 'ab' if resume and dest.exists() else 'wb'
        
        with open(dest, mode) as f:
            if resume and dest.exists():
                # Resume download
                downloaded = dest.stat().st_size
                headers = {'Range': f'bytes={downloaded}-'}
                response = requests.get(url, headers=headers, stream=True, timeout=30)
            else:
                downloaded = 0
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"  Downloading {dest.name}") as pbar:
                pbar.update(downloaded)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file with progress."""
    if not zip_path.exists():
        return False
    
    print(f"  Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total = len(zip_ref.namelist())
            extracted = 0
            for member in zip_ref.namelist():
                zip_ref.extract(member, extract_to)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f"    Progress: {extracted}/{total} ({100*extracted/total:.1f}%)")
            print(f"  ✓ Extracted {extracted} files")
            return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_open_images_v6(data_dir: Path) -> bool:
    """Download Open Images V6 validation set...."""
    print("\n" + "="*70)
    print("Downloading Open Images V6 (Validation Set)")
    print("="*70)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = data_dir / "validation"
    csv_path = data_dir / "validation-annotations-bbox.csv"
    
    # Checks if already downloaded
    if validation_dir.exists():
        img_count = len(list(validation_dir.rglob("*.jpg")))
        if img_count > 0 and csv_path.exists():
            print(f"  ✓ Open Images V6 already downloaded ({img_count} images)")
            return True
    
    # Uses FiftyOne method when available
    try:
        import fiftyone as fo
        print("\n  Using FiftyOne to download Open Images V6...")
        print("  This will download ~41K validation images (~2GB)")
        print("  This may take 30-60 minutes depending on your connection...")
        
        # Download using FiftyOne
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="validation",
            dataset_dir=str(data_dir.parent),  # FiftyOne will create open_images_v6 subdir
            label_types=["detections"],
            max_samples=None  # Download all validation images
        )
        
        print(f"  ✓ Downloaded {len(dataset)} images")
        
        # Reorganize to expected structure
        print("  Reorganizing files to expected structure...")
        
        # FiftyOne stores images in its own structure, we need to reorganize
        fo_dataset_dir = data_dir.parent / "open-images-v6-validation"
        if not fo_dataset_dir.exists():
            # Uses alternative location
            fo_dataset_dir = data_dir.parent / "open-images-v6" / "validation"
        
        if fo_dataset_dir.exists():
            # Move images to validation/
            validation_dir.mkdir(parents=True, exist_ok=True)
            for img_path in fo_dataset_dir.rglob("*.jpg"):
                # Open Images uses subdirectories by image ID prefix
                img_id = img_path.stem
                subdir = validation_dir / img_id[:2]
                subdir.mkdir(exist_ok=True)
                img_path.rename(subdir / img_path.name)
        
        # Export annotations CSV
        print("  Exporting annotations...")
        try:
            dataset.export(
                export_dir=str(data_dir),
                dataset_type=fo.types.COCODetectionDataset,
            )
            # Convert COCO format to Open Images CSV format if needed
            # For now, we'll download the CSV separately
        except:
            pass
        
        # Download annotation CSV
        csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
        print(f"  Downloading annotation CSV...")
        if download_file(csv_url, csv_path):
            print(f"  ✓ Annotation CSV downloaded")
        
        # Verify
        img_count = len(list(validation_dir.rglob("*.jpg")))
        if img_count > 0 and csv_path.exists():
            print(f"\n  ✅ Open Images V6 download complete!")
            print(f"     Images: {img_count}")
            print(f"     Annotations: ✓")
            return True
        else:
            print(f"  ⚠ Download incomplete - some files missing")
            return False
            
    except ImportError:
        print("\n  FiftyOne not installed. Installing...")
        print("  Run: pip install fiftyone")
        print("\n  Falling back to manual download instructions...")
    except Exception as e:
        print(f"\n  FiftyOne download failed: {e}")
        print("  Falling back to manual download instructions...")
    
    # Fallback: Manual download instructions
    print("\n" + "="*70)
    print("Manual Download Instructions for Open Images V6")
    print("="*70)
    print("\n  Option 1: Install FiftyOne and retry:")
    print("    pip install fiftyone")
    print("    python scripts/download_inference_datasets.py --skip-bdd100k --skip-ade20k")
    print("\n  Option 2: Manual download from Google Cloud Storage:")
    print("    1. Visit: https://storage.googleapis.com/openimages/web/index.html")
    print("    2. Download: 'Validation Images' (~2GB)")
    print("    3. Extract to: datasets/open_images_v6/validation/")
    print("    4. Download: 'Validation Annotations' (validation-annotations-bbox.csv)")
    print("    5. Place CSV in: datasets/open_images_v6/")
    
    # Download annotation CSV when possible
    csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
    print(f"\n  Attempting to download annotation CSV...")
    if download_file(csv_url, csv_path):
        print(f"  ✓ Annotation CSV downloaded")
        print(f"  ⚠ Images still need to be downloaded manually")
        return False
    else:
        print(f"  ⚠ Annotation CSV download failed (manual download required)")
        return False


def download_bdd100k(data_dir: Path) -> bool:
    """Download BDD100K validation set.
    
    Note: Requires registration at bdd-data.berkeley.edu"""
    print("\n" + "="*70)
    print("Downloading BDD100K (Validation Set)")
    print("="*70)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images" / "100k"
    labels_dir = data_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Data is available at: https://dl.cv.ethz.ch/bdd100k/data/
    print("\n  BDD100K Download:")
    print("  Data available at: https://dl.cv.ethz.ch/bdd100k/data/")
    print("  No registration required - direct download available!")
    
    # BDD100K direct download URLs (from https://dl.cv.ethz.ch/bdd100k/data/)
    # Correct URLs from the actual directory listing
    images_url = "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"  # 542MB - validation images only
    labels_url = "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"  # 53MB - Detection 2020 labels (train+val)
    
    print(f"\n  Downloading from: https://dl.cv.ethz.ch/bdd100k/data/")
    print(f"  Validation images: 542MB")
    print(f"  Detection labels: 53MB")
    
    import shutil
    
    # Download labels first (smaller, 53MB)
    print(f"\n  Step 1: Downloading Detection 2020 labels...")
    labels_zip = data_dir / "det_20_labels.zip"
    labels_path = labels_dir / "bdd100k_labels_images_val.json"
    
    if download_file(labels_url, labels_zip):
        print(f"  ✓ Labels downloaded, extracting...")
        # Extract to temp location
        temp_labels = data_dir / "temp_labels"
        if extract_zip(labels_zip, temp_labels):
            # Find det_val.json in the extracted structure
            # Structure: bdd100k/labels/det_20/det_val.json
            det_val_source = None
            for path in temp_labels.rglob("det_val.json"):
                det_val_source = path
                break
            
            if det_val_source and det_val_source.exists():
                # Copy to expected location
                shutil.copy(det_val_source, labels_path)
                print(f"  ✓ Labels extracted to: {labels_path}")
                labels_success = True
            else:
                print(f"  ⚠ Could not find det_val.json in extracted files")
                labels_success = False
            
            # Cleanup
            shutil.rmtree(temp_labels)
            labels_zip.unlink()
        else:
            labels_success = False
    else:
        print(f"  ⚠ Labels download failed")
        labels_success = False
    
    # Download validation images (542MB)
    print(f"\n  Step 2: Downloading validation images (542MB)...")
    images_zip = data_dir / "100k_images_val.zip"
    
    if download_file(images_url, images_zip):
        print(f"  ✓ Images downloaded, extracting...")
        # Extract to temp location
        temp_extract = data_dir / "temp_extract"
        if extract_zip(images_zip, temp_extract):
            # Find val folder in extracted structure
            # Structure: bdd100k/images/100k/val/
            val_source = None
            for path in temp_extract.rglob("val"):
                if path.is_dir() and len(list(path.glob("*.jpg"))) > 0:
                    val_source = path
                    break
            
            if val_source and val_source.exists():
                val_dest = images_dir / "val"
                if val_dest.exists():
                    shutil.rmtree(val_dest)
                shutil.move(str(val_source), str(val_dest))
                print(f"  ✓ Validation images extracted to: {val_dest}")
                images_success = True
            else:
                print(f"  ⚠ Could not find val folder with images")
                images_success = False
            
            # Cleanup
            shutil.rmtree(temp_extract)
            images_zip.unlink()
        else:
            images_success = False
    else:
        print(f"  ⚠ Images download failed")
        images_success = False
    
    if images_success and labels_success:
        return True
    
    # If download failed, provide manual instructions
    if not images_success or not labels_success:
        print(f"\n" + "="*70)
        print(f"⚠️  BDD100K Download Failed")
        print(f"="*70)
        print(f"\nPossible reasons:")
        print(f"  - DNS resolution failed (dl.cv.ethz.ch not accessible)")
        print(f"  - Network/firewall blocking access")
        print(f"  - Server temporarily unavailable")
        
        print(f"\n📋 Alternative Options:")
        print(f"\n1. Use VPN/Proxy:")
        print(f"   - Enable VPN that can access ETH Zurich domains")
        print(f"   - Retry this script or download manually")
        
        print(f"\n2. Manual Download (if VPN works):")
        print(f"   - Visit: https://dl.cv.ethz.ch/bdd100k/data/")
        if not images_success:
            print(f"   - Download: '100k_images_val.zip' (542MB)")
            print(f"   - Extract val folder to: {images_dir / 'val'}")
        if not labels_success:
            print(f"   - Download: 'bdd100k_det_20_labels_trainval.zip' (53MB)")
            print(f"   - Extract det_val.json to: {labels_path}")
        
        print(f"\n3. Alternative Sources:")
        print(f"   - Hugging Face: https://huggingface.co/datasets/kd7/graid-bdd100k")
        print(f"   - FiftyOne: https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html")
        print(f"   - GitHub: https://github.com/bdd100k/bdd100k")
        
        print(f"\n4. Skip for Now:")
        print(f"   - ADE20K is already downloaded ✅ (indoor scenes)")
        print(f"   - BDD100K is for outdoor/driving scenarios - can add later")
    
    return images_success and labels_success


def download_ade20k(data_dir: Path) -> bool:
    """Download ADE20K validation set.
    
    ADE20K is available from MIT Vision Group."""
    print("\n" + "="*70)
    print("Downloading ADE20K (Validation Set)")
    print("="*70)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ADE20K download URLs (MIT Vision Group)
    # Direct download links
    urls = {
        'validation_images': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
        # Note: This zip contains both training and validation
    }
    
    print("\n  Downloading ADE20K dataset...")
    print("  Note: This includes both training and validation sets (~2GB)")
    
    zip_path = data_dir / "ADEChallengeData2016.zip"
    
    # Download the main zip file
    if download_file(urls['validation_images'], zip_path):
        print(f"  ✓ Download complete, extracting...")
        
        # Extract to temp location first
        temp_extract = data_dir / "temp_extract"
        if extract_zip(zip_path, temp_extract):
            # Move to correct structure
            ade_data = temp_extract / "ADEChallengeData2016"
            if ade_data.exists():
                # Move images
                if (ade_data / "images").exists():
                    images_dest = data_dir / "images"
                    if images_dest.exists():
                        shutil.rmtree(images_dest)
                    shutil.move(str(ade_data / "images"), str(images_dest))
                
                # Move annotations
                if (ade_data / "annotations").exists():
                    ann_dest = data_dir / "annotations"
                    if ann_dest.exists():
                        shutil.rmtree(ann_dest)
                    shutil.move(str(ade_data / "annotations"), str(ann_dest))
                
                # Cleanup
                shutil.rmtree(temp_extract)
                zip_path.unlink()
                
                print(f"  ✓ ADE20K extracted successfully")
                return True
    
    print(f"  ✗ ADE20K download/extraction failed")
    return False


def verify_dataset(dataset_name: str, data_dir: Path) -> Dict[str, bool]:
    """Verify that a dataset is properly downloaded."""
    status = {
        'images': False,
        'annotations': False,
        'complete': False
    }
    
    if dataset_name == 'open_images_v6':
        validation_dir = data_dir / "validation"
        csv_file = data_dir / "validation-annotations-bbox.csv"
        
        if validation_dir.exists():
            img_count = len(list(validation_dir.rglob("*.jpg")))
            status['images'] = img_count > 0
            if status['images']:
                print(f"    Images: {img_count} found")
        
        if csv_file.exists():
            status['annotations'] = True
            print(f"    Annotations: ✓")
    
    elif dataset_name == 'bdd100k':
        val_images = data_dir / "images" / "100k" / "val"
        val_labels = data_dir / "labels" / "bdd100k_labels_images_val.json"
        
        if val_images.exists():
            img_count = len(list(val_images.glob("*.jpg")))
            status['images'] = img_count > 0
            if status['images']:
                print(f"    Images: {img_count} found")
        
        if val_labels.exists():
            status['annotations'] = True
            print(f"    Annotations: ✓")
    
    elif dataset_name == 'ade20k':
        val_images = data_dir / "images" / "validation"
        val_annotations = data_dir / "annotations" / "validation"
        
        if val_images.exists():
            img_count = len(list(val_images.glob("*.jpg")))
            status['images'] = img_count > 0
            if status['images']:
                print(f"    Images: {img_count} found")
        
        if val_annotations.exists():
            ann_count = len(list(val_annotations.glob("*.png")))
            status['annotations'] = ann_count > 0
            if status['annotations']:
                print(f"    Annotations: {ann_count} found")
    
    status['complete'] = status['images'] and status['annotations']
    return status


def main():
    parser = argparse.ArgumentParser(
        description='Download all inference datasets for MaxSight'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=ROOT / "datasets",
        help='Base directory for datasets (default: datasets/)'
    )
    parser.add_argument(
        '--skip-open-images',
        action='store_true',
        help='Skip Open Images V6 download'
    )
    parser.add_argument(
        '--skip-bdd100k',
        action='store_true',
        help='Skip BDD100K download'
    )
    parser.add_argument(
        '--skip-ade20k',
        action='store_true',
        help='Skip ADE20K download'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets, do not download'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MaxSight Inference Datasets Downloader")
    print("="*70)
    print(f"\nBase directory: {args.base_dir}")
    print("\nThis script will download:")
    print("  1. Open Images V6 (validation set) - Broad semantic diversity")
    print("  2. BDD100K (validation set) - Motion/outdoor/hazard realism")
    print("  3. ADE20K (validation set) - Indoor structure & objects")
    print("\n⚠️  Note: Some datasets require manual download due to:")
    print("  - Registration requirements (BDD100K)")
    print("  - Large size distributed across many files (Open Images)")
    print("  - Authentication/rate limits")
    
    if args.verify_only:
        print("\n" + "="*70)
        print("Verifying Existing Datasets")
        print("="*70)
        
        datasets = {
            'open_images_v6': args.base_dir / "open_images_v6",
            'bdd100k': args.base_dir / "bdd100k",
            'ade20k': args.base_dir / "ade20k"
        }
        
        all_complete = True
        for name, path in datasets.items():
            print(f"\n{name}:")
            if path.exists():
                status = verify_dataset(name, path)
                if status['complete']:
                    print(f"  ✓ Complete")
                else:
                    print(f"  ⚠ Incomplete")
                    all_complete = False
            else:
                print(f"  ✗ Not found")
                all_complete = False
        
        if all_complete:
            print("\n✅ All inference datasets are complete!")
            return 0
        else:
            print("\n⚠️  Some datasets are incomplete. Run without --verify-only to download.")
            return 1
    
    # Download datasets
    results = {}
    
    if not args.skip_open_images:
        results['open_images_v6'] = download_open_images_v6(args.base_dir / "open_images_v6")
    else:
        print("\n⏭️  Skipping Open Images V6")
    
    if not args.skip_bdd100k:
        results['bdd100k'] = download_bdd100k(args.base_dir / "bdd100k")
    else:
        print("\n⏭️  Skipping BDD100K")
    
    if not args.skip_ade20k:
        results['ade20k'] = download_ade20k(args.base_dir / "ade20k")
    else:
        print("\n⏭️  Skipping ADE20K")
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    
    for name, success in results.items():
        status = "✓ Success" if success else "⚠ Partial/Manual download required"
        print(f"  {name}: {status}")
    
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    datasets = {
        'open_images_v6': args.base_dir / "open_images_v6",
        'bdd100k': args.base_dir / "bdd100k",
        'ade20k': args.base_dir / "ade20k"
    }
    
    for name, path in datasets.items():
        if args.skip_open_images and name == 'open_images_v6':
            continue
        if args.skip_bdd100k and name == 'bdd100k':
            continue
        if args.skip_ade20k and name == 'ade20k':
            continue
        
        print(f"\n{name}:")
        if path.exists():
            verify_dataset(name, path)
        else:
            print("  ✗ Not found")
    
    print("\n✅ Download process complete!")
    print("\nNote: Some datasets may require manual download. See instructions above.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
