# COCO Image Patching Guide

## Problem

Your training dataset has **~20% missing images** (1,972 train / 406 val). The current loader falls back to random noise for missing files, which degrades training quality.

## Solution

Download missing images **while training continues** - no restart needed!

---

## Quick Start

### Option 1: Automatic (Recommended)

```bash
# Patch all splits (train + val)
./scripts/run_image_patcher.sh

# Or patch specific split
./scripts/run_image_patcher.sh train
./scripts/run_image_patcher.sh val
```

### Option 2: Direct Python

```bash
# All splits
python3 scripts/patch_missing_images.py --split all --workers 4

# Train only
python3 scripts/patch_missing_images.py --split train --workers 4

# Val only
python3 scripts/patch_missing_images.py --split val --workers 4
```

---

## How It Works

1. **Scans** your annotation files to find missing image paths
2. **Downloads** missing images from COCO servers (http://images.cocodataset.org)
3. **Places** them in the correct directories:
   - `datasets/coco_raw/train2017/`
   - `datasets/coco_raw/val2017/`

4. **Your training automatically picks up the real images** in the next batch (no code changes needed)

---

## Performance

- **4 parallel workers** (adjustable with `--workers`)
- Downloads ~50-100 images/minute (depends on network)
- **~1,972 train images** ≈ 20-40 minutes
- **~406 val images** ≈ 4-8 minutes

---

## Running in Background

To patch while training:

```bash
# Start in background
nohup ./scripts/run_image_patcher.sh all > image_patcher.log 2>&1 &

# Monitor progress
tail -f image_patcher.log

# Check if still running
ps aux | grep patch_missing_images
```

---

## Verification

Check completion status:

```python
python3 -c "
from pathlib import Path
import json

with open('datasets/cleaned_splits/maxsight_train.json') as f:
    train = json.load(f)

missing = sum(1 for s in train if not Path(s['image_path']).exists())
print(f'Missing train images: {missing}/{len(train)} ({100*missing/len(train):.1f}%)')
"
```

---

## Expected Output

```
================================================
COCO Image Patcher
================================================
Starting patch for train split...
Loaded 10000 samples from train split
Found 1972 missing images in train split
Downloading 1972 images with 4 workers...
Progress: 50/1972 (✓ 48, ✗ 2)
Progress: 100/1972 (✓ 97, ✗ 3)
...
Completed train split: 1950 success, 22 failed
✓ All train images are now available!
================================================
SUMMARY: Downloaded 1950 images, 22 failed
================================================
```

---

## Important Notes

### ✅ Safe During Training

- **No restart needed** - training continues unaffected
- **No code changes** - loader automatically uses new images
- **CPU-friendly** - downloads run independently
- **Gradual improvement** - data quality increases as images appear

### ⚠️ Some Images May Be Permanently Unavailable

- COCO servers occasionally remove images
- Script will retry 3x per image before giving up
- Failed downloads are logged for reference
- Training continues with the images that are available

### 📊 Impact on Training

**Before patching:**
- 80% real images, 20% random noise
- Model learns but with degraded signal

**After patching:**
- ~98% real images, ~2% noise (permanently missing)
- Significantly better convergence and final performance

---

## Troubleshooting

### "Connection refused" or timeouts

```bash
# Reduce workers to avoid rate limiting
python3 scripts/patch_missing_images.py --split all --workers 2
```

### Check disk space

```bash
# COCO images are ~15GB for train, ~800MB for val
df -h datasets/coco_raw/
```

### Verify downloaded images

```bash
# Check file count
find datasets/coco_raw/train2017 -name "*.jpg" | wc -l
find datasets/coco_raw/val2017 -name "*.jpg" | wc -l
```

---

## Manual Alternative

If automated download fails, you can manually download the full COCO datasets:

```bash
cd datasets/coco_raw

# Download full train2017 set (~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download full val2017 set (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

This ensures 100% coverage but takes longer and uses more bandwidth.

---

## Summary

Run this **now** while training continues:

```bash
./scripts/run_image_patcher.sh all
```

Your model will automatically benefit from real images as they become available!
