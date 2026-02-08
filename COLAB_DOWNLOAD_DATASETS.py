"""Single Colab cell to download all inference datasets for MaxSight.

Copy-paste this entire cell into Colab and run it."""

# CELL 1: Download Inference Datasets (Open Images V6, BDD100K, ADE20K)

import os
import sys
from pathlib import Path
import subprocess

# Setup paths.
if 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_IP' in os.environ:
    # Running in Colab.
    BASE_DIR = Path("/content/drive/MyDrive/MaxSight/datasets")
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[ok] Colab detected - using Drive: {BASE_DIR}")
else:
    # Running locally.
    BASE_DIR = Path("/content/datasets") if Path("/content").exists() else Path("./datasets")
    print(f"[ok] Local environment - using: {BASE_DIR}")

# Install FiftyOne for Open Images V6.
print("\n📦 Installing FiftyOne...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fiftyone"], check=False)

# Install requests and tqdm if needed.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests", "tqdm"], check=False)

import fiftyone as fo
import requests
from tqdm import tqdm

print("[ok] Dependencies installed\n")

# Download Open Images V6 Validation Set.

print("="*70)
print("Downloading Open Images V6 (Validation Set)")
print("="*70)

open_images_dir = BASE_DIR / "open_images_v6"
validation_dir = open_images_dir / "validation"
csv_path = open_images_dir / "validation-annotations-bbox.csv"

# Check if already downloaded.
if validation_dir.exists():
    img_count = len(list(validation_dir.rglob("*.jpg")))
    if img_count > 1000 and csv_path.exists():
        print(f"[ok] Open Images V6 already downloaded ({img_count:,} images)")
    else:
        print(f"WARNING Partial download found ({img_count} images) - continuing...")
else:
    print("\n📥 Downloading ~41,620 validation images (~2 GB)...")
    print("   This will take 30-60 minutes depending on connection...")
    
    try:
        # Download using FiftyOne.
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split="validation",
            dataset_dir=str(BASE_DIR.parent),
            label_types=["detections"],
            max_samples=None
        )
        
        print(f"[ok] Downloaded {len(dataset)} images")
        
        # Reorganize files to expected structure.
        print("  Reorganizing files...")
        fo_dataset_paths = [
            BASE_DIR.parent / "open-images-v6-validation",
            BASE_DIR.parent / "open-images-v6" / "validation",
        ]
        
        source_dir = None
        for path in fo_dataset_paths:
            if path.exists():
                img_count = len(list(path.rglob("*.jpg")))
                if img_count > 0:
                    source_dir = path
                    break
        
        if source_dir:
            validation_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for img_path in source_dir.rglob("*.jpg"):
                # Preserve subdirectory structure (by image ID prefix)
                rel_path = img_path.relative_to(source_dir)
                if "data" in str(rel_path):
                    parts = rel_path.parts
                    if len(parts) > 1:
                        subdir = validation_dir / parts[0]
                        subdir.mkdir(exist_ok=True)
                        dest = subdir / parts[-1]
                    else:
                        dest = validation_dir / rel_path.name
                else:
                    # Handle flat structure.
                    img_id = img_path.stem
                    subdir = validation_dir / img_id[:2] if len(img_id) >= 2 else validation_dir
                    subdir.mkdir(exist_ok=True)
                    dest = subdir / img_path.name
                
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    img_path.rename(dest)
                    moved += 1
            
            print(f"  [ok] Moved {moved} images to {validation_dir}")
        
        # Download annotation CSV.
        print("\n📥 Downloading annotations CSV...")
        csv_url = "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv"
        
        response = requests.get(csv_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(csv_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading CSV") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("  [ok] Annotations downloaded")
        
    except Exception as e:
        print(f"WARNING FiftyOne download failed: {e}")
        print("  You can download manually from:")
        print("  https://storage.googleapis.com/openimages/web/index.html")

# Verify Open Images V6.
img_count = len(list(validation_dir.rglob("*.jpg"))) if validation_dir.exists() else 0
if img_count > 0 and csv_path.exists():
    print(f"\nOK Open Images V6: {img_count:,} images ready")
else:
    print(f"\nWARNING  Open Images V6: Incomplete ({img_count} images)")

# Download BDD100K Validation Set.

print("\n" + "="*70)
print("Downloading BDD100K (Validation Set)")
print("="*70)

bdd100k_dir = BASE_DIR / "bdd100k"
images_dir = bdd100k_dir / "images" / "100k" / "val"
labels_dir = bdd100k_dir / "labels"
labels_path = labels_dir / "bdd100k_labels_images_val.json"

if images_dir.exists() and labels_path.exists():
    img_count = len(list(images_dir.glob("*.jpg")))
    print(f"[ok] BDD100K already downloaded ({img_count:,} images)")
else:
    print("\n📥 Downloading BDD100K validation set (~600 MB)...")
    
    try:
        # Try direct download from ETH Zurich.
        images_url = "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
        labels_url = "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"
        
        import zipfile
        import shutil
        
        # Download labels first (smaller)
        print("  Downloading labels (53 MB)...")
        labels_zip = bdd100k_dir / "det_20_labels.zip"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(labels_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(labels_zip, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading labels") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract labels.
        print("  Extracting labels...")
        temp_labels = bdd100k_dir / "temp_labels"
        with zipfile.ZipFile(labels_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_labels)
        
        # Find det_val.json.
        det_val_source = None
        for path in temp_labels.rglob("det_val.json"):
            det_val_source = path
            break
        
        if det_val_source:
            shutil.copy(det_val_source, labels_path)
            print(f"  [ok] Labels extracted to {labels_path}")
        
        # Cleanup.
        shutil.rmtree(temp_labels, ignore_errors=True)
        labels_zip.unlink(missing_ok=True)
        
        # Download images.
        print("\n  Downloading images (542 MB)...")
        images_zip = bdd100k_dir / "100k_images_val.zip"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(images_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(images_zip, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading images") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract images.
        print("  Extracting images...")
        temp_extract = bdd100k_dir / "temp_extract"
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        # Find val folder.
        val_source = None
        for path in temp_extract.rglob("val"):
            if path.is_dir() and len(list(path.glob("*.jpg"))) > 0:
                val_source = path
                break
        
        if val_source:
            if images_dir.exists():
                shutil.rmtree(images_dir)
            shutil.move(str(val_source), str(images_dir))
            print(f"  [ok] Images extracted to {images_dir}")
        
        # Cleanup.
        shutil.rmtree(temp_extract, ignore_errors=True)
        images_zip.unlink(missing_ok=True)
        
        print("OK BDD100K download complete")
        
    except Exception as e:
        print(f"WARNING BDD100K download failed: {e}")
        print("  Alternative: Use FiftyOne:")
        print("  dataset = fo.zoo.load_zoo_dataset('bdd100k', split='validation')")

# Verify BDD100K.
img_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
if img_count > 0 and labels_path.exists():
    print(f"OK BDD100K: {img_count:,} images ready")
else:
    print(f"WARNING  BDD100K: Incomplete ({img_count} images)")

# Download ADE20K Validation Set.

print("\n" + "="*70)
print("Downloading ADE20K (Validation Set)")
print("="*70)

ade20k_dir = BASE_DIR / "ade20k"
ade20k_val_images = ade20k_dir / "images" / "validation"
ade20k_val_annotations = ade20k_dir / "annotations" / "validation"

if ade20k_val_images.exists() and ade20k_val_annotations.exists():
    img_count = len(list(ade20k_val_images.glob("*.jpg")))
    print(f"[ok] ADE20K already downloaded ({img_count:,} images)")
else:
    print("\n📥 Downloading ADE20K validation set (~1 GB)...")
    
    try:
        ade20k_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        zip_path = ade20k_dir / "ADEChallengeData2016.zip"
        ade20k_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Downloading ADE20K dataset...")
        response = requests.get(ade20k_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract.
        print("  Extracting...")
        import zipfile
        temp_extract = ade20k_dir / "temp_extract"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        # Move to expected structure.
        ade_data = temp_extract / "ADEChallengeData2016"
        if ade_data.exists():
            if (ade_data / "images").exists():
                images_dest = ade20k_dir / "images"
                if images_dest.exists():
                    shutil.rmtree(images_dest)
                shutil.move(str(ade_data / "images"), str(images_dest))
            
            if (ade_data / "annotations").exists():
                ann_dest = ade20k_dir / "annotations"
                if ann_dest.exists():
                    shutil.rmtree(ann_dest)
                shutil.move(str(ade_data / "annotations"), str(ann_dest))
        
        # Cleanup.
        shutil.rmtree(temp_extract, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        
        print("OK ADE20K download complete")
        
    except Exception as e:
        print(f"WARNING ADE20K download failed: {e}")

# Verify ADE20K.
img_count = len(list(ade20k_val_images.glob("*.jpg"))) if ade20k_val_images.exists() else 0
ann_count = len(list(ade20k_val_annotations.glob("*.png"))) if ade20k_val_annotations.exists() else 0
if img_count > 0 and ann_count > 0:
    print(f"OK ADE20K: {img_count:,} images, {ann_count:,} annotations ready")
else:
    print(f"WARNING  ADE20K: Incomplete ({img_count} images, {ann_count} annotations)")

# Summary.

print("\n" + "="*70)
print("Download Summary")
print("="*70)

datasets_status = []

# Open Images V6.
oi6_count = len(list(validation_dir.rglob("*.jpg"))) if validation_dir.exists() else 0
oi6_csv = csv_path.exists()
datasets_status.append(("Open Images V6", oi6_count, oi6_csv, "41,620"))

# BDD100K.
bdd_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
bdd_labels = labels_path.exists()
datasets_status.append(("BDD100K", bdd_count, bdd_labels, "~10,000"))

# ADE20K.
ade_count = len(list(ade20k_val_images.glob("*.jpg"))) if ade20k_val_images.exists() else 0
ade_ann = ade20k_val_annotations.exists()
datasets_status.append(("ADE20K", ade_count, ade_ann, "~2,000"))

print(f"\n{'Dataset':<20} {'Images':<15} {'Annotations':<15} {'Status'}")
print("-" * 70)

for name, count, has_ann, target in datasets_status:
    status = "OK Complete" if (count > 100 and has_ann) else "WARNING  Incomplete"
    print(f"{name:<20} {count:>6,} / {target:<8} {'[ok]' if has_ann else 'no':<14} {status}")

print(f"\n All datasets saved to: {BASE_DIR}")
print("\nOK Ready for inference evaluation!")


