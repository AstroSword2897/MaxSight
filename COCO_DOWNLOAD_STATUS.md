# COCO Dataset Download Status

**Last Updated**: 2025-01-30 21:40

## Current Status

✅ **Download in Progress**: `train2017.zip` is downloading via `curl` in background

### Download Progress

- **File**: `datasets/coco_raw/train2017.zip`
- **Expected Size**: ~18 GB
- **Current Size**: ~640 MB (3.5% complete)
- **Download Method**: `curl` with resume capability (`-C -`)

### Already Complete

✅ **val2017.zip**: Extracted (5000 images verified)  
✅ **annotations**: Extracted (all annotation files present)  
✅ **train_annotations**: Verified (118,287 images)  
✅ **val_annotations**: Verified (5,000 images)

### Pending

⏳ **train2017.zip**: Downloading (~18 GB, will take time depending on connection)

---

## Monitoring Download

### Option 1: Check File Size Manually
```bash
ls -lh datasets/coco_raw/train2017.zip
```

### Option 2: Use Monitoring Script
```bash
python scripts/monitor_coco_download.py
```

### Option 3: Check Download Process
```bash
ps aux | grep curl | grep train2017
```

---

## After Download Completes

### Step 1: Verify Download
```bash
python scripts/download_coco.py --verify-only
```

### Step 2: Extract Files
```bash
python scripts/extract_coco.py
```

This will:
- Extract `train2017.zip` → `datasets/coco_raw/train2017/` (~118K images)
- Verify all files are extracted correctly
- Check final dataset completeness

### Step 3: Final Verification
```bash
python scripts/download_coco.py --verify-only
```

Expected output:
```
✅ COCO dataset is complete and verified!
```

---

## Download Time Estimates

Based on typical download speeds:

| Speed | Estimated Time |
|-------|----------------|
| 10 Mbps | ~4 hours |
| 50 Mbps | ~48 minutes |
| 100 Mbps | ~24 minutes |
| 1 Gbps | ~2.4 minutes |

**Note**: Actual time depends on server load and network conditions.

---

## Troubleshooting

### Download Stuck or Slow

1. **Check if download is still running**:
   ```bash
   ps aux | grep curl | grep train2017
   ```

2. **If download stopped, resume it**:
   ```bash
   cd datasets/coco_raw
   curl -L -C - --progress-bar --retry 3 -o train2017.zip \
     "http://images.cocodataset.org/zips/train2017.zip"
   ```

3. **Check disk space**:
   ```bash
   df -h .
   ```
   Need at least 20GB free for extraction.

### Corrupted Download

If download completes but extraction fails:

1. **Remove corrupted file**:
   ```bash
   rm datasets/coco_raw/train2017.zip
   ```

2. **Re-download**:
   ```bash
   python scripts/download_coco.py --auto --data_dir datasets/coco_raw
   ```

### Manual Download

If automatic download fails, download manually:

1. Visit: http://images.cocodataset.org/zips/train2017.zip
2. Save to: `datasets/coco_raw/train2017.zip`
3. Run: `python scripts/extract_coco.py`

---

## Next Steps After Download

Once the dataset is complete:

1. **Create training splits**:
   ```bash
   python scripts/setup_training_data.py
   ```

2. **Test data pipeline**:
   ```bash
   python scripts/test_training_pipeline.py
   ```

3. **Start training**:
   ```bash
   python scripts/smoke_train.py --tier T0_BASELINE_CNN
   ```

---

**Status**: ⏳ Download in progress (background process)

