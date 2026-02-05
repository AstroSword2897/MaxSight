# Colab Restart & System Verification Guide

## 🚨 Hit Colab Quota Limits? Start Here

**What's saved vs. lost:**
- ❌ **LOST**: Anything in `/content/` (repo, checkpoints, packages)
- ✅ **SAVED**: Anything in `/content/drive/MyDrive/...` (if you mounted Drive)

**Check what you have on Drive:**
```python
from google.colab import drive
drive.mount("/content/drive")

# Check for checkpoints
!ls -lh /content/drive/MyDrive/MaxSight/checkpoints/ 2>/dev/null || echo "No checkpoints found"

# Check for data
!ls -lh /content/drive/MyDrive/MaxSight/datasets/ 2>/dev/null || echo "No datasets found"
```

**If you have checkpoints on Drive:** → Jump to "Step 6: Resume Training"  
**If you don't have checkpoints:** → Start fresh from Cell 1 in COLAB_RUNBOOK.md

---

# Colab Restart & System Verification Guide

## 🧹 Step 1: Cleanup Old Checkpoints

After restarting the kernel, clean up old checkpoints to free space:

```python
# In Colab cell:
!python scripts/cleanup_cloud_checkpoints.py --checkpoint-dir checkpoints --execute
```

**Options:**
- `--keep-best`: Keep `best_model.pt` (default: True)
- `--keep-last`: Keep `last_checkpoint.pt` (default: True)
- `--keep-recent N`: Keep N most recent checkpoints
- `--clean-logs`: Also delete log files
- `--clean-temp`: Also delete `__pycache__` and `.pyc` files
- `--all`: Delete everything except best and last checkpoints
- `--execute`: Actually delete (default is dry-run)

**Example - Clean everything except best/last:**
```python
!python scripts/cleanup_cloud_checkpoints.py --all --execute
```

## ✅ Step 2: Verify All Systems

Run comprehensive tests to ensure everything works:

```python
# Test everything
!python scripts/test_systems_comprehensive.py --test all

# Or test individual components:
!python scripts/test_systems_comprehensive.py --test gradnorm
!python scripts/test_systems_comprehensive.py --test automl
!python scripts/test_systems_comprehensive.py --test false-positives
```

**Expected Output:**
```
🔬 Comprehensive System Test Suite
======================================================================
Device: cuda
PyTorch version: 2.x.x
CUDA available: True

🧪 Testing GradNorm Integration
======================================================================
✅ PASS: GradNorm

🧪 Testing AutoML Integration
======================================================================
✅ PASS: AutoML

🧪 Testing False Positive Detection
======================================================================
✅ PASS: False Positives

📊 Test Summary
======================================================================
✅ PASS: GradNorm
✅ PASS: AutoML
✅ PASS: False Positives
✅ PASS: Model Forward
✅ PASS: Validation Loss

Total: 5/5 tests passed
🎉 All systems operational!
```

## 📁 Annotation files (train/val JSON)

If you see **`--train-annotation and --val-annotation files must exist`**:

1. **Put JSONs under your data-dir** so the script can find them:
   - `{data-dir}/cleaned_splits/maxsight_train.json`
   - `{data-dir}/cleaned_splits/maxsight_val.json`  
   Example: if `--data-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw`, place the files in  
   `/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits/`.

2. **Or use absolute paths** to where the files already are:
   ```bash
   --train-annotation /content/drive/MyDrive/MaxSight/cleaned_splits/maxsight_train.json
   --val-annotation /content/drive/MyDrive/MaxSight/cleaned_splits/maxsight_val.json
   ```

3. **Create the splits** if you don’t have them yet:
   ```bash
   python scripts/gather_training_data.py --coco-dir <path-to-coco> --output-dir datasets/cleaned_splits
   ```
   Then upload the generated `maxsight_train.json` and `maxsight_val.json` to Drive (e.g. under `data-dir/cleaned_splits/`).

## 🧪 Step 3: Test AutoML

Test AutoML with a quick trial:

```python
# Quick AutoML test (1 trial) – use paths where your JSONs actually are
!python scripts/AutoMLType.py \
  --data-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --train-annotation maxsight_train.json \
  --val-annotation maxsight_val.json \
  --image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --checkpoint-dir checkpoints_automl \
  --n-trials 1 \
  --epochs-per-trial 1 \
  --device cuda
```

## 🎯 Step 4: Test GradNorm

Test GradNorm integration:

```python
# Run GradNorm integration test
!python tests/test_gradnorm_integration.py
```

**Expected:** All tests should pass without inplace operation errors.

## 🔍 Step 5: Test False Positives

Test false positive detection:

```python
# Test false positive detection in metrics
!python scripts/test_systems_comprehensive.py --test false-positives
```

## 🚀 Step 6: Start Training

Once all tests pass, start training:

```python
# Start training with fixes applied
!python scripts/train_maxsight.py \
  --data-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --epochs 5 \
  --batch-size 8 \
  --device cuda \
  --use-gradnorm \
  --checkpoint-interval 0
```

## 📋 Quick Checklist

- [ ] Kernel restarted
- [ ] Old checkpoints cleaned up
- [ ] All system tests passed
- [ ] GradNorm test passed (no inplace errors)
- [ ] AutoML test passed
- [ ] False positive detection working
- [ ] Validation loss is not NaN
- [ ] Training started successfully

## 🐛 Troubleshooting

### If validation loss is still NaN:
```python
# Enable anomaly detection to find the issue
import torch
torch.autograd.set_detect_anomaly(True)

# Then run training - it will show exactly where NaN occurs
```

### If GradNorm errors persist:
```python
# Check for inplace operations in model
!grep -r "\.clamp_\|\.fill_\|\.add_\|\.mul_" ml/models/
!grep -r "masked_fill_" ml/training/
```

### If AutoML fails:
```python
# Check Optuna installation
!pip show optuna

# Reinstall if needed
!pip install optuna --upgrade
```

## 📊 Monitoring Training

Watch for these in logs:
- ✅ Validation loss should be finite (not `nan`)
- ✅ GradNorm warnings should be gone or rare
- ✅ Validation metrics should show values (not all 0.0000)
- ✅ Training loss should decrease over epochs

## 🔗 Related Files

- `scripts/cleanup_cloud_checkpoints.py` - Cleanup script
- `scripts/test_systems_comprehensive.py` - System tests
- `tests/test_gradnorm_integration.py` - GradNorm tests
- `TRAINING_FIXES_SUMMARY.md` - Details of fixes applied
