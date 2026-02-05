#!/usr/bin/env python3
"""Reorganize Open Images V6 from FiftyOne to datasets directory."""
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    print("="*70)
    print("Reorganizing Open Images V6")
    print("="*70)
    
    # Source: FiftyOne
    fo_source = Path.home() / "fiftyone" / "open-images-v6" / "validation"
    # Destination
    dest_dir = ROOT / "datasets" / "open_images_v6" / "validation"
    csv_dest = ROOT / "datasets" / "open_images_v6" / "validation-annotations-bbox.csv"
    
    if not fo_source.exists():
        print(f"\n✗ Source not found: {fo_source}")
        return 1
    
    img_count = len(list(fo_source.rglob("*.jpg")))
    print(f"\n✓ Found {img_count:,} images in {fo_source}")
    
    if dest_dir.exists():
        dest_count = len(list(dest_dir.rglob("*.jpg")))
        if dest_count > 1000:
            print(f"✓ Already reorganized ({dest_count:,} images)")
            if csv_dest.exists():
                print("✅ Ready!")
                return 0
    
    print(f"\n📦 Moving images to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    moved = 0
    for img_path in tqdm(fo_source.rglob("*.jpg"), desc="Moving", total=img_count):
        img_id = img_path.stem
        subdir = dest_dir / img_id[:2] if len(img_id) >= 2 else dest_dir
        subdir.mkdir(exist_ok=True)
        dest = subdir / img_path.name
        if not dest.exists():
            img_path.rename(dest)
            moved += 1
    
    print(f"✓ Moved {moved:,} images")
    
    # Download CSV
    if not csv_dest.exists():
        print("\n📥 Downloading annotations CSV...")
        import requests
        csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
        r = requests.get(csv_url, stream=True)
        r.raise_for_status()
        with open(csv_dest, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("✓ CSV downloaded")
    
    final_count = len(list(dest_dir.rglob("*.jpg")))
    print(f"\n✅ Complete! {final_count:,} images ready")
    return 0

if __name__ == "__main__":
    sys.exit(main())
