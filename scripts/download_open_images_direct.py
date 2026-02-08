#!/usr/bin/env python3
"""Direct download script for Open Images V6 validation set.

Uses the CVDF GitHub repository downloader for reliable downloads."""

import sys
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def download_with_cvdf_repo(data_dir: Path) -> bool:
    """Download Open Images V6 using CVDF repository downloader."""
    print("\n" + "="*70)
    print("Downloading Open Images V6 using CVDF Repository")
    print("="*70)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = data_dir / "validation"
    
    # Checks if already downloaded.
    if validation_dir.exists():
        img_count = len(list(validation_dir.rglob("*.jpg")))
        if img_count > 1000:  # Reasonable threshold.
            print(f"  ✓ Open Images V6 already has {img_count} images")
            return True
    
    # Clone or use CVDF repository.
    temp_dir = ROOT / "temp_open_images_downloader"
    repo_url = "https://github.com/cvdfoundation/open-images-dataset.git"
    
    print("\n  Step 1: Setting up CVDF downloader...")
    
    if temp_dir.exists():
        print("  Using existing downloader directory...")
        downloader_dir = temp_dir
    else:
        print(f"  Cloning CVDF repository to {temp_dir}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(temp_dir)],
                check=True,
                capture_output=True
            )
            downloader_dir = temp_dir
            print("  ✓ Repository cloned")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to clone repository: {e}")
            print("\n  Alternative: Manual download required")
            print("  See: OPEN_IMAGES_V6_DOWNLOAD_GUIDE.md")
            return False
    
    # Finds downloader script.
    downloader_script = downloader_dir / "downloader.py"
    if not downloader_script.exists():
        # Uses alternative location.
        downloader_script = downloader_dir / "download.py"
    
    if not downloader_script.exists():
        print(f"  ✗ Downloader script not found in {downloader_dir}")
        print("  Trying alternative method...")
        return download_with_fiftyone(data_dir)
    
    # Run downloader for validation set.
    print("\n  Step 2: Downloading validation images...")
    print("  This will download ~41K images (~2GB)")
    print("  This may take 30-60 minutes...")
    
    try:
        # CVDF downloader command.
        cmd = [
            sys.executable,
            str(downloader_script),
            "--split", "validation",
            "--num_processes", "4",
            "--output_dir", str(validation_dir.parent)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(downloader_dir))
        
        if result.returncode == 0:
            print("  ✓ Download complete")
            
            # Download annotations CSV.
            print("\n  Step 3: Downloading annotations...")
            csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
            csv_path = data_dir / "validation-annotations-bbox.csv"
            
            import requests
            from tqdm import tqdm
            
            try:
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
                
                # Verify.
                img_count = len(list(validation_dir.rglob("*.jpg")))
                if img_count > 0 and csv_path.exists():
                    print(f"\n  ✅ Open Images V6 download complete!")
                    print(f"     Images: {img_count}")
                    print(f"     Annotations: ✓")
                    return True
                else:
                    print(f"  ⚠ Some files may be missing")
                    return False
                    
            except Exception as e:
                print(f"  ⚠ CSV download failed: {e}")
                print("  You can download it manually from:")
                print("  https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv")
                return False
        else:
            print("  ✗ Download failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Downloader execution failed: {e}")
        return download_with_fiftyone(data_dir)


def download_with_fiftyone(data_dir: Path) -> bool:
    """Fallback to FiftyOne method."""
    print("\n  Trying FiftyOne method...")
    try:
        import fiftyone as fo
        
        print("  Downloading with FiftyOne...")
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="validation",
            dataset_dir=str(data_dir.parent),
            label_types=["detections"],
            max_samples=None
        )
        
        print(f"  ✓ Downloaded {len(dataset)} images")
        
        # Reorganize files.
        validation_dir = data_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Download CSV.
        csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
        csv_path = data_dir / "validation-annotations-bbox.csv"
        
        import requests
        response = requests.get(csv_url)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
        
        print("  ✅ Download complete")
        return True
        
    except ImportError:
        print("  ✗ FiftyOne not installed. Install with: pip install fiftyone")
        return False
    except Exception as e:
        print(f"  ✗ FiftyOne download failed: {e}")
        return False


def main():
    data_dir = ROOT / "datasets" / "open_images_v6"
    
    print("="*70)
    print("Open Images V6 Direct Downloader")
    print("="*70)
    
    # Use CVDF method.
    if download_with_cvdf_repo(data_dir):
        print("\n✅ Success!")
        return 0
    else:
        print("\n⚠️  Download incomplete. See OPEN_IMAGES_V6_DOWNLOAD_GUIDE.md for manual options.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

