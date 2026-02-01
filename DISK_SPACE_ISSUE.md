# Disk Space Issue - COCO Extraction

## Problem

Extraction of `train2017.zip` failed with error:
```
[Errno 28] No space left on device
```

## Current Status

- ✅ **Download Complete**: `train2017.zip` (18 GB) downloaded successfully
- ✅ **Zip File Valid**: Contains 118,288 files
- ⚠️ **Extraction Failed**: Ran out of disk space at ~95,000 files (80% extracted)

## Disk Space Analysis

The COCO dataset requires:
- **train2017.zip**: 18 GB (compressed)
- **train2017/**: ~19 GB (extracted, ~118K images)
- **val2017/**: ~1 GB (already extracted)
- **annotations/**: ~250 MB (already extracted)

**Total Required**: ~38 GB

## Solutions

### Option 1: Free Up Space (Recommended)

1. **Check what's taking up space**:
   ```bash
   du -sh ~/* | sort -h | tail -10
   ```

2. **Common space hogs to check**:
   - Downloads folder: `~/Downloads`
   - Docker images: `docker system prune -a`
   - Old Python caches: `find ~ -name "__pycache__" -type d -exec rm -r {} +`
   - Conda/pip caches: `conda clean --all` or `pip cache purge`
   - Trash: Empty trash
   - Large log files: `find ~ -name "*.log" -size +100M`

3. **After freeing space, resume extraction**:
   ```bash
   python scripts/extract_coco.py
   ```
   (It will skip already-extracted files)

### Option 2: Extract to External Drive

If you have an external drive with space:

1. **Mount external drive** (if not already mounted)

2. **Create symlink or move dataset**:
   ```bash
   # Option A: Move entire coco_raw to external drive
   mv datasets/coco_raw /Volumes/ExternalDrive/coco_raw
   ln -s /Volumes/ExternalDrive/coco_raw datasets/coco_raw
   
   # Option B: Extract directly to external drive
   unzip -q datasets/coco_raw/train2017.zip -d /Volumes/ExternalDrive/coco_raw/
   ```

3. **Update paths in scripts** if needed

### Option 3: Use Smaller Dataset Subset

If you only need a subset for testing:

1. **Extract only first N images**:
   ```python
   import zipfile
   from pathlib import Path
   
   zip_path = Path('datasets/coco_raw/train2017.zip')
   extract_to = Path('datasets/coco_raw/train2017')
   extract_to.mkdir(exist_ok=True)
   
   with zipfile.ZipFile(zip_path, 'r') as z:
       files = z.namelist()
       # Extract first 10,000 images
       for f in files[:10001]:  # +1 for directory
           z.extract(f, extract_to.parent)
   ```

2. **Or use existing partial extraction**:
   - ~95,000 images were already extracted
   - This might be enough for initial testing

### Option 4: Clean Up Partial Extraction

If you want to start fresh after freeing space:

```bash
# Remove partial extraction
rm -rf datasets/coco_raw/train2017

# Then extract again
python scripts/extract_coco.py
```

## Check Current Disk Space

```bash
df -h .
```

You need at least **20 GB free** to complete the extraction.

## Verify Partial Extraction

If you want to use the partially extracted dataset:

```bash
# Count extracted images
find datasets/coco_raw/train2017 -name "*.jpg" | wc -l

# Check size
du -sh datasets/coco_raw/train2017
```

**Note**: ~95,000 images (80% of dataset) might be sufficient for initial training/testing.

## Next Steps

1. **Free up disk space** (recommended)
2. **Resume extraction**: `python scripts/extract_coco.py`
3. **Or use partial dataset** for now and complete later

---

**Status**: ⚠️ Extraction paused due to disk space (80% complete, ~95K images extracted)

