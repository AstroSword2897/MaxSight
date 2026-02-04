# Manual COCO Dataset Download Instructions

## Current Status

✅ **Already downloaded:**
- Val images (5000 images) - extracted
- Annotations (complete) - extracted
- Val images zip (778MB)

❌ **Missing:**
- Train images (~18GB) - **train2017.zip**

## Manual Download Steps

### 1. Download Train Images

**Direct download link:**
```
http://images.cocodataset.org/zips/train2017.zip
```

**File size:** ~18GB (19,336,861,798 bytes)

**Download options:**

#### Option A: Browser Download
1. Open your web browser
2. Navigate to: `http://images.cocodataset.org/zips/train2017.zip`
3. Save to: `datasets/coco_raw/train2017.zip`

#### Option B: Command Line (wget)
```bash
cd /Users/nani/2026-Prototype/datasets/coco_raw
wget http://images.cocodataset.org/zips/train2017.zip
```

#### Option C: Command Line (curl)
```bash
cd /Users/nani/2026-Prototype/datasets/coco_raw
curl -O http://images.cocodataset.org/zips/train2017.zip
```

#### Option D: Python requests (if browser/CLI don't work)
```bash
cd /Users/nani/2026-Prototype
source /Users/nani/2026/venv/bin/activate
python -c "
import requests
from pathlib import Path

url = 'http://images.cocodataset.org/zips/train2017.zip'
output_path = Path('datasets/coco_raw/train2017.zip')
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f'Downloading {url}...')
print(f'File size: ~18GB')
print(f'Saving to: {output_path}')

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
downloaded = 0

with open(output_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (100 * 1024 * 1024) == 0:  # Print every 100MB
                print(f'Downloaded: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB')

print('✅ Download complete!')
"
```

### 2. Verify Download

After downloading, verify the file:

```bash
cd /Users/nani/2026-Prototype
ls -lh datasets/coco_raw/train2017.zip
```

Expected size: ~18GB (19,336,861,798 bytes)

### 3. Extract Train Images

Once downloaded, extract the zip file:

```bash
cd /Users/nani/2026-Prototype
source /Users/nani/2026/venv/bin/activate
python scripts/setup_coco_data.py
```

Or manually:

```bash
cd /Users/nani/2026-Prototype/datasets/coco_raw
unzip train2017.zip
```

This will create `train2017/` directory with ~118K images.

### 4. Final Verification

Verify the complete dataset:

```bash
cd /Users/nani/2026-Prototype
source /Users/nani/2026/venv/bin/activate
python scripts/download_coco.py --verify-only
```

You should see:
```
✅ train_images
✅ val_images
✅ annotations
✅ train_annotations
✅ val_annotations
```

## Quick Reference

**Target directory:** `datasets/coco_raw/`

**Files needed:**
- `train2017.zip` (~18GB) - **Download this**
- `val2017.zip` (778MB) - ✅ Already have
- `annotations_trainval2017.zip` (241MB) - ✅ Already have

**After download:**
- Extract `train2017.zip` to get `train2017/` directory
- Run `python scripts/setup_coco_data.py` to verify

## Troubleshooting

### Download is slow
- COCO servers can be slow. Be patient, or try during off-peak hours.
- The download may take 30-60 minutes depending on your connection.

### Download fails partway through
- Resume with `wget -c` or `curl -C -`
- Or restart the download (the file is large, so partial downloads are common)

### Not enough disk space
- Train images need ~18GB for zip + ~18GB for extracted = ~36GB total
- Make sure you have at least 40GB free space

### Extraction takes a long time
- Extracting 118K images can take 10-20 minutes
- Be patient, it's normal

## Next Steps After Download

Once COCO is complete:

1. **Create training splits:**
   ```bash
   python scripts/setup_training_data.py \
     --train_samples 10000 \
     --val_samples 2000 \
     --test_samples 1000
   ```

2. **Test data pipeline:**
   ```bash
   python scripts/test_training_pipeline.py --num-batches 3
   ```

3. **Start training:**
   ```bash
   python scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml
   ```

