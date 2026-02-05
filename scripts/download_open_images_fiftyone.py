#!/usr/bin/env python3
"""
Download Open Images V6 using FiftyOne (recommended method).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("="*70)
    print("Open Images V6 Download via FiftyOne")
    print("="*70)
    
    try:
        import fiftyone as fo
        print(f"\n✓ FiftyOne {fo.__version__} installed")
    except ImportError:
        print("\n✗ FiftyOne not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "fiftyone", "-q"], check=True)
        import fiftyone as fo
        print("✓ FiftyOne installed")
    
    data_dir = ROOT / "datasets" / "open_images_v6"
    validation_dir = data_dir / "validation"
    
    # Check if already downloaded
    if validation_dir.exists():
        img_count = len(list(validation_dir.rglob("*.jpg")))
        if img_count > 1000:
            print(f"\n✓ Already have {img_count} images")
            print("  Use --force to re-download")
            return 0
    
    print("\n" + "="*70)
    print("Starting Download")
    print("="*70)
    print("\n  Dataset: Open Images V6 Validation Set")
    print("  Images: ~41,620")
    print("  Size: ~2 GB")
    print("  Estimated Time: 30-60 minutes")
    print("\n  This will download images to:")
    print(f"    {validation_dir}")
    print("\n  Starting download... (this may take a while)")
    
    try:
        # Download using FiftyOne
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="validation",
            dataset_dir=str(data_dir.parent),  # FiftyOne creates subdirectory
            label_types=["detections"],
            max_samples=None  # Download all validation images
        )
        
        print(f"\n✓ Downloaded {len(dataset)} images")
        
        # Reorganize to expected structure
        print("\n  Reorganizing files...")
        
        # Find where FiftyOne stored the images
        # FiftyOne typically stores in: dataset_dir/open-images-v6-validation/
        fo_dataset_paths = [
            data_dir.parent / "open-images-v6-validation",
            data_dir.parent / "open-images-v6" / "validation",
            data_dir / "validation"
        ]
        
        source_dir = None
        for path in fo_dataset_paths:
            if path.exists():
                img_count = len(list(path.rglob("*.jpg")))
                if img_count > 0:
                    source_dir = path
                    break
        
        if source_dir and source_dir != validation_dir:
            print(f"  Moving images from {source_dir} to {validation_dir}...")
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            # Move images preserving subdirectory structure
            for img_path in source_dir.rglob("*.jpg"):
                rel_path = img_path.relative_to(source_dir)
                dest_path = validation_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.rename(dest_path)
            
            print("  ✓ Files reorganized")
        
        # Download annotation CSV
        print("\n  Downloading annotations CSV...")
        csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
        csv_path = data_dir / "validation-annotations-bbox.csv"
        
        import requests
        from tqdm import tqdm
        
        response = requests.get(csv_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(csv_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading CSV") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("  ✓ Annotations downloaded")
        
        # Verify
        img_count = len(list(validation_dir.rglob("*.jpg")))
        csv_size = csv_path.stat().st_size if csv_path.exists() else 0
        
        print("\n" + "="*70)
        print("Download Complete!")
        print("="*70)
        print(f"\n  Images: {img_count}")
        print(f"  Annotations: {csv_size / 1024 / 1024:.1f} MB")
        print(f"  Location: {validation_dir}")
        
        if img_count > 1000 and csv_size > 100000:  # >100KB for CSV
            print("\n✅ Open Images V6 is ready for inference!")
            return 0
        else:
            print("\n⚠️  Download may be incomplete. Check files manually.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\n  Alternative: Use manual download")
        print("  See: OPEN_IMAGES_V6_DOWNLOAD_GUIDE.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
