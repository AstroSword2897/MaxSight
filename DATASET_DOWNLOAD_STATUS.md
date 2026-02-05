# Inference Dataset Download Status

**Started**: 2026-02-05  
**Status**: ⏳ **IN PROGRESS**

---

## Current Status

| Dataset | Status | Size | Notes |
|---------|--------|------|-------|
| **ADE20K** | ✅ **COMPLETE** | ~1.0 GB | Already downloaded (2,000 images) |
| **Open Images V6** | ⏳ **DOWNLOADING** | ~2 GB | Using FiftyOne (41K validation images) - Check progress below |
| **BDD100K** | ❌ **DNS BLOCKED** | ~600 MB | Cannot access dl.cv.ethz.ch - See alternatives below |

---

## Download Progress

### Open Images V6
- **Method**: FiftyOne (automatic download)
- **Progress**: Downloading validation set (~41,620 images)
- **Estimated Time**: 30-60 minutes depending on connection
- **Location**: `datasets/open_images_v6/validation/`
- **Annotations**: Will download CSV automatically

### BDD100K
- **Method**: Direct download from ETH Zurich
- **Progress**: Downloading validation images (542MB) + labels (53MB)
- **Estimated Time**: 10-20 minutes
- **Location**: `datasets/bdd100k/images/100k/val/`
- **Labels**: `datasets/bdd100k/labels/bdd100k_labels_images_val.json`

---

## Monitoring Progress

### Check Download Status
```bash
# View download log
tail -f /tmp/dataset_download.log

# Or check terminal output
# (The download is running in background)

# Verify datasets when complete
python scripts/download_inference_datasets.py --verify-only
```

### Expected Output When Complete
```
✅ ADE20K: Complete
✅ Open Images V6: Complete
✅ BDD100K: Complete
```

---

## What's Happening

1. **FiftyOne** is downloading Open Images V6 validation set
   - This handles the complex tar file structure automatically
   - Downloads images and organizes them properly
   - May take 30-60 minutes for ~2GB

2. **Direct Download** for BDD100K
   - Attempting to download from ETH Zurich servers
   - If DNS fails, will provide alternative options

---

## After Download Completes

Once downloads finish, you can:

1. **Verify Datasets**:
   ```bash
   python scripts/download_inference_datasets.py --verify-only
   ```

2. **Use for Inference**:
   ```python
   from ml.data.inference_datasets import create_inference_dataloader
   from pathlib import Path
   
   # Open Images V6
   loader = create_inference_dataloader(
       dataset_name='open_images_v6',
       root=Path('datasets/open_images_v6'),
       split='validation',
       batch_size=32
   )
   
   # BDD100K
   loader = create_inference_dataloader(
       dataset_name='bdd100k',
       root=Path('datasets/bdd100k'),
       split='val',
       batch_size=32
   )
   ```

---

## Troubleshooting

### If Open Images V6 Download Fails
**Current Method**: FiftyOne (automatic)

**If FiftyOne fails, try alternative**:
```bash
# Use CVDF repository downloader
python scripts/download_open_images_direct.py
```

**Manual Methods**: See `OPEN_IMAGES_V6_DOWNLOAD_GUIDE.md` for:
- Direct Google Cloud Storage download
- CVDF GitHub repository method
- FiftyOne manual setup

### If BDD100K Download Fails (DNS Issue) - CURRENT STATUS
**Problem**: `dl.cv.ethz.ch` DNS resolution fails from your network

**Solutions**:
1. **Use VPN/Proxy**: Enable VPN that can access ETH Zurich domains, then retry:
   ```bash
   python scripts/download_inference_datasets.py --skip-open-images --skip-ade20k
   ```

2. **Manual Download** (if VPN works):
   - Visit: https://dl.cv.ethz.ch/bdd100k/data/
   - Download: `100k_images_val.zip` (542MB) → Extract to `datasets/bdd100k/images/100k/val/`
   - Download: `bdd100k_det_20_labels_trainval.zip` (53MB) → Extract `det_val.json` to `datasets/bdd100k/labels/bdd100k_labels_images_val.json`

3. **Alternative Sources**:
   - **FiftyOne**: `fo.zoo.load_zoo_dataset("bdd100k", split="validation")`
   - **Hugging Face**: https://huggingface.co/datasets/kd7/graid-bdd100k (requires format conversion)
   - **GitHub**: https://github.com/bdd100k/bdd100k (may have download scripts)

4. **Skip for Now**: BDD100K is for outdoor/driving scenarios. You can use:
   - ✅ **ADE20K** for indoor scenes (already downloaded)
   - ⏳ **Open Images V6** for diverse scenes (downloading now)

### Check Disk Space
- Ensure you have at least **3-4 GB** free space
- Open Images V6: ~2GB
- BDD100K: ~600MB
- Extraction overhead: ~500MB

---

## Next Steps

1. ⏳ **Wait for downloads to complete** (check log file)
2. ✅ **Verify datasets** using verification script
3. 🚀 **Run inference** on downloaded datasets
4. 📊 **Evaluate model performance** across different scenarios

---

**Note**: Downloads are running in the background. Check `/tmp/dataset_download.log` for progress updates.
