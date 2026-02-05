# Pre-Training Verification Checklist

**Run these checks before starting training to ensure everything is ready.**

---

## 1. Verify Inference Datasets (Optional - for evaluation)

If you plan to run inference evaluation after training:

**Local:**
```bash
python scripts/download_inference_datasets.py --verify-only
```

**Colab (after download completes):**
```python
from pathlib import Path
BASE_DIR = Path("/content/drive/MyDrive/MaxSight/datasets")

# Check Open Images V6
oi6_path = BASE_DIR / "open_images_v6/validation"
oi6_count = len(list(oi6_path.rglob("*.jpg"))) if oi6_path.exists() else 0
print(f"Open Images V6: {oi6_count:,} images")

# Check ADE20K
ade_path = BASE_DIR / "ade20k/images/validation"
ade_count = len(list(ade_path.glob("*.jpg"))) if ade_path.exists() else 0
print(f"ADE20K: {ade_count:,} images")

# Check BDD100K
bdd_path = BASE_DIR / "bdd100k/images/100k/val"
bdd_count = len(list(bdd_path.glob("*.jpg"))) if bdd_path.exists() else 0
print(f"BDD100K: {bdd_count:,} images")
```

**Expected:**
- Open Images V6: ~41,620 images
- ADE20K: ~2,000 images
- BDD100K: ~10,000 images (optional)

---

## 2. Verify Training Pipeline

**Critical - must pass before training:**

```bash
python scripts/validate_data_pipeline.py
```

**What it checks:**
- Training/validation splits exist (`datasets/cleaned_splits/maxsight_train.json`, `maxsight_val.json`)
- Images are accessible
- Data loaders work correctly
- No NaN/Inf values in batches
- Augmentations preserve annotation structure
- Class weights computed correctly

**If splits are missing:**
```bash
# Create splits from COCO data
python scripts/gather_training_data.py
```

**Success criteria:**
- No errors
- All batches load successfully
- No invalid values detected

---

## 3. Check Training Data Availability

**Verify training images exist:**
```bash
# Check COCO training images
ls -lh datasets/coco_raw/train2017/ | wc -l
# Should show ~95K+ images (80% complete is sufficient)

# Check validation images
ls -lh datasets/coco_raw/val2017/ | wc -l
# Should show ~5,000 images
```

**Minimum requirements:**
- Training: 50K+ images (you have 95K+ ✅)
- Validation: 5,000 images ✅
- Annotations: All JSON files present ✅

---

## 4. Verify Checkpoint Directory

**Ensure checkpoint directory exists and is writable:**
```bash
mkdir -p checkpoints
ls -lh checkpoints/
```

**If resuming training:**
- Check that `checkpoints/last_checkpoint.pt` exists
- Verify checkpoint size matches expected (~600MB-1GB)

---

## 5. Quick Sanity Check (Optional but Recommended)

**Run a tiny training test to catch errors early:**
```bash
python scripts/train_maxsight.py \
  --epochs 1 \
  --batch-size 4 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

**What to watch for:**
- No OOM (out of memory) errors
- No import errors
- Model loads correctly
- Forward pass works
- Loss computes without NaN

**If this passes, full training should work.**

---

## Quick Reference

**All checks pass?** → Proceed to training

**Any failures?** → Fix issues before training

**Training command (full):**
```bash
python scripts/train_maxsight.py \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

**Resume from checkpoint:**
```bash
python scripts/train_maxsight.py \
  --resume \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```
