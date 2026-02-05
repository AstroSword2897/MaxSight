# Inference Datasets Download Status

**Last Updated**: February 5, 2026

## Summary

| Dataset | Status | Images | Annotations | Notes |
|---------|--------|--------|-------------|-------|
| **ADE20K** | ✅ **COMPLETE** | 2,000 | 2,000 | Fully downloaded and ready |
| **BDD100K** | ❌ **DNS/BLOCKED** | 0 | 0 | Domain not accessible (dl.cv.ethz.ch) |
| **Open Images V6** | ❌ **MANUAL DOWNLOAD REQUIRED** | 0 | 0 | Requires manual download |

---

## ✅ ADE20K - COMPLETE

**Location**: `datasets/ade20k/`

**Status**: ✅ Fully downloaded and extracted
- Validation images: 2,000 images
- Validation annotations: 2,000 segmentation masks
- Size: ~1.0 GB

**Usage**: Ready for inference evaluation

---

## ❌ BDD100K - Manual Download Required

**Location**: `datasets/bdd100k/`

**Status**: ❌ Requires manual download

**⚠️ DNS Issue**: The official download server `dl.cv.ethz.ch` is not accessible from your network (DNS resolution fails). You'll need to use alternative methods.

### Option 1: Use VPN/Proxy (Recommended)

If you have access to a VPN or proxy that can resolve `dl.cv.ethz.ch`:

1. **Enable VPN/Proxy** that can access ETH Zurich domains
2. **Visit**: https://dl.cv.ethz.ch/bdd100k/data/
3. **Download Validation Images** (542MB):
   - Download: `100k_images_val.zip` (542MB)
   - Extract and copy the `val` folder to: `datasets/bdd100k/images/100k/val/`
4. **Download Validation Labels** (53MB):
   - Download: `bdd100k_det_20_labels_trainval.zip` (53MB)
   - Extract and find `det_val.json` inside
   - Copy to: `datasets/bdd100k/labels/bdd100k_labels_images_val.json`

**Direct Download Links** (requires VPN/proxy):
- Images: https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip
- Labels: https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip

### Option 2: Alternative Sources

**Hugging Face** (processed version):
- Dataset: https://huggingface.co/datasets/kd7/graid-bdd100k
- Note: This is a processed version with different format (parquet files)
- May require conversion to match expected structure

**FiftyOne Dataset Zoo**:
- Integration available through FiftyOne
- See: https://docs.voxel51.com/dataset_zoo/datasets/bdd100k.html

**GitHub Toolkit**:
- Official toolkit: https://github.com/bdd100k/bdd100k
- May have download scripts or alternative links

### Option 3: Skip BDD100K (Temporary)

If BDD100K is not accessible, you can:
- Use **ADE20K** for indoor scene evaluation (already downloaded ✅)
- Use **Open Images V6** for diverse scene evaluation (when downloaded)
- BDD100K is specifically for outdoor/driving scenarios - can be added later when network access is available

4. **Verify**:
   ```bash
   python scripts/download_inference_datasets.py --verify-only
   ```

**Expected Structure**:
```
datasets/bdd100k/
├── images/
│   └── 100k/
│       └── val/          # ~10K validation images
└── labels/
    └── bdd100k_labels_images_val.json
```

---

## ❌ Open Images V6 - Manual Download Required

**Location**: `datasets/open_images_v6/`

**Status**: ❌ Requires manual download

**Why**: Open Images V6 images are distributed across many tar files and require manual download from Google Cloud Storage.

**Manual Download Steps**:

1. **Visit**: https://storage.googleapis.com/openimages/web/index.html

2. **Download Validation Set** (~2GB):
   - Click: "Validation Images" 
   - This downloads a tar file or multiple tar files
   - Extract to: `datasets/open_images_v6/validation/`
   - Images are organized in subdirectories by image ID prefix

3. **Download Validation Annotations**:
   - Download: `validation-annotations-bbox.csv`
   - Place in: `datasets/open_images_v6/validation-annotations-bbox.csv`

4. **Verify**:
   ```bash
   python scripts/download_inference_datasets.py --verify-only
   ```

**Expected Structure**:
```
datasets/open_images_v6/
├── validation/
│   ├── [subdirectories by image ID prefix]/
│   │   └── *.jpg        # ~41K validation images
└── validation-annotations-bbox.csv
```

**Note**: The full Open Images V6 dataset has 9M images (~500GB). For inference evaluation, the validation set (~41K images, ~2GB) is sufficient.

---

## Verification

After downloading missing datasets, verify everything is complete:

```bash
python scripts/download_inference_datasets.py --verify-only
```

Expected output:
```
✅ ADE20K: Complete
✅ BDD100K: Complete  
✅ Open Images V6: Complete
```

---

## Usage

Once all datasets are downloaded, you can use them for inference evaluation:

```python
from ml.data.inference_datasets import create_inference_dataloader
from pathlib import Path

# ADE20K (indoor scenes)
ade20k_loader = create_inference_dataloader(
    dataset_name='ade20k',
    root=Path('datasets/ade20k'),
    split='validation',
    batch_size=32
)

# BDD100K (outdoor/driving scenes)
bdd100k_loader = create_inference_dataloader(
    dataset_name='bdd100k',
    root=Path('datasets/bdd100k'),
    split='val',
    batch_size=32
)

# Open Images V6 (diverse scenes)
open_images_loader = create_inference_dataloader(
    dataset_name='open_images_v6',
    root=Path('datasets/open_images_v6'),
    split='validation',
    batch_size=32
)
```

---

## Next Steps

1. ✅ **ADE20K**: Ready to use
2. ⏳ **BDD100K**: Register and download manually
3. ⏳ **Open Images V6**: Download validation set manually

The download script (`scripts/download_inference_datasets.py`) will attempt automatic downloads where possible, but these two datasets require manual steps due to authentication/registration requirements.
