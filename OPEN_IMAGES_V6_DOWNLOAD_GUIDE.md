# Open Images V6 Download Guide

**Purpose**: Download validation set for inference evaluation (~41K images, ~2GB)

---

## Method 1: Using FiftyOne (Easiest - Recommended)

FiftyOne provides the easiest way to download Open Images V6 with selective options.

### Steps:

1. **Install FiftyOne**:
   ```bash
   pip install fiftyone
   ```

2. **Download Validation Set**:
   ```python
   import fiftyone as fo
   
   # Download validation set (~2GB)
   dataset = fo.zoo.load_zoo_dataset(
       "open-images-v6",
       split="validation",
       dataset_dir="datasets/open_images_v6",
       label_types=["detections"],  # Only download bounding box annotations
       max_samples=None  # Download all validation images
   )
   ```

3. **Export to Expected Structure**:
   ```python
   # Export images to validation/ folder
   dataset.export(
       export_dir="datasets/open_images_v6/validation",
       dataset_type=fo.types.ImageDirectory,
   )
   
   # Export annotations CSV
   dataset.export(
       export_dir="datasets/open_images_v6",
       dataset_type=fo.types.COCODetectionDataset,
   )
   ```

**Pros**: 
- Easy to use
- Can select specific classes/annotations
- Handles organization automatically

**Cons**: 
- Requires installing FiftyOne
- May need to reorganize files to match expected structure

---

## Method 2: Manual Download from Google Cloud Storage

### Steps:

1. **Visit Download Page**:
   - Go to: https://storage.googleapis.com/openimages/web/download.html
   - Or: https://storage.googleapis.com/openimages/web/index.html

2. **Download Validation Images**:
   - Look for "Validation Images" section
   - Download the validation images tar file(s)
   - **Note**: Images may be split into multiple tar files (e.g., `validation_000.tar`, `validation_001.tar`, etc.)
   - Total size: ~2GB

3. **Extract Images**:
   ```bash
   # Create directory
   mkdir -p datasets/open_images_v6/validation
   
   # Extract each tar file
   cd datasets/open_images_v6/validation
   tar -xf /path/to/validation_000.tar
   tar -xf /path/to/validation_001.tar
   # ... repeat for all tar files
   ```
   
   **Note**: Images are organized in subdirectories by image ID prefix (e.g., `0a1b2c3d/`, `0e1f2g3h/`)

4. **Download Validation Annotations**:
   - From the download page, find "Validation Annotations"
   - Download: `validation-annotations-bbox.csv` (or similar)
   - Place in: `datasets/open_images_v6/validation-annotations-bbox.csv`

5. **Verify Structure**:
   ```bash
   # Check images
   find datasets/open_images_v6/validation -name "*.jpg" | wc -l
   # Should show ~41,620 images
   
   # Check annotations
   ls -lh datasets/open_images_v6/validation-annotations-bbox.csv
   ```

**Pros**: 
- Direct download
- Full control over files

**Cons**: 
- May need to download multiple tar files
- Manual extraction required

---

## Method 3: Using CVDF GitHub Repository

### Steps:

1. **Clone/Visit Repository**:
   - Go to: https://github.com/cvdfoundation/open-images-dataset
   - Check the README for download scripts

2. **Use Download Script**:
   ```bash
   # Clone the repo (or download scripts)
   git clone https://github.com/cvdfoundation/open-images-dataset.git
   cd open-images-dataset
   
   # Use their download script for validation set
   # (Check their README for exact command)
   python downloader.py --split validation --num_processes 4
   ```

3. **Organize Files**:
   - Move downloaded images to: `datasets/open_images_v6/validation/`
   - Move annotations to: `datasets/open_images_v6/validation-annotations-bbox.csv`

**Pros**: 
- Official repository
- May have optimized download scripts

**Cons**: 
- Requires cloning repository
- May need to adapt scripts

---

## Expected Final Structure

After downloading, your directory should look like:

```
datasets/open_images_v6/
├── validation/
│   ├── 0a1b2c3d/
│   │   └── 0a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.jpg
│   ├── 0e1f2g3h/
│   │   └── 0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6.jpg
│   └── ... (more subdirectories)
└── validation-annotations-bbox.csv
```

**Key Points**:
- Images are in subdirectories organized by image ID prefix
- Each image filename is its full image ID
- CSV file contains bounding box annotations

---

## Verification

After downloading, verify everything is correct:

```bash
cd /Users/nani/2026-Prototype
PYTHONPATH=/Users/nani/2026-Prototype python scripts/download_inference_datasets.py --verify-only
```

Expected output:
```
open_images_v6:
    Images: 41620 found
    Annotations: ✓
  ✓ Complete
```

---

## Quick Command Reference

### Using Python Script (if we add direct download support):
```bash
python scripts/download_inference_datasets.py --skip-bdd100k --skip-ade20k
```

### Manual Verification:
```bash
# Count images
find datasets/open_images_v6/validation -name "*.jpg" | wc -l

# Check CSV exists
ls -lh datasets/open_images_v6/validation-annotations-bbox.csv

# Check CSV format
head -5 datasets/open_images_v6/validation-annotations-bbox.csv
```

---

## Troubleshooting

### Issue: Multiple tar files
**Solution**: Extract all tar files into the same `validation/` directory. The subdirectory structure will be preserved.

### Issue: Images not in expected subdirectories
**Solution**: The script `ml/data/inference_datasets.py` should handle both flat and nested structures. If issues persist, check the dataset loader code.

### Issue: Annotations CSV format
**Solution**: Open Images uses CSV format with columns like: `ImageID`, `Source`, `LabelName`, `Confidence`, `XMin`, `XMax`, `YMin`, `YMax`. The inference dataset loader should handle this format.

### Issue: Download is slow
**Solution**: 
- Use FiftyOne for better download management
- Download during off-peak hours
- Consider using a download manager for tar files

---

## Notes

- **Full Dataset**: Open Images V6 has 9M images (~500GB). For inference evaluation, the validation set (~41K images, ~2GB) is sufficient.
- **Storage**: Ensure you have at least 3-4GB free space (2GB images + extraction overhead + annotations)
- **Time**: Download time depends on your connection. Expect 30-60 minutes for ~2GB on a typical connection.

---

## Next Steps

After downloading:
1. Verify with the script: `python scripts/download_inference_datasets.py --verify-only`
2. Test loading: Use `ml/data/inference_datasets.py` to create a dataloader
3. Run inference: Use the dataloader with your trained model for evaluation
