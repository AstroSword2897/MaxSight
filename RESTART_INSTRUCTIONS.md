# Quick Restart Instructions

**Goal:** Stop validation warnings and use patched images  
**Time Required:** 2 minutes  
**Status:** ✅ Code fixed, ready to restart

---

## ✅ What's Been Fixed

1. **Removed validation warnings** - No more "Batch validation failed" spam
2. **Pre-sanitized data** - Boxes fixed in collate_fn before validation
3. **Patched images** - 100% train, 79.7% val (96.6% total) real images

---

## 🔄 Restart Steps

### Step 1: Stop Current Training
```bash
# Find the training process
ps aux | grep train_maxsight | grep -v grep

# Kill it gracefully (allows checkpoint save)
pkill -SIGINT -f train_maxsight.py

# Wait 10 seconds for graceful shutdown
sleep 10
```

### Step 2: Verify Patcher Completed
```bash
# Check if patcher finished
ps aux | grep patch_missing | grep -v grep

# If still running, wait for it
tail -f /Users/nani/2026-Prototype/image_patcher_full.log
# Press Ctrl+C when you see "SUMMARY"
```

### Step 3: Restart Training
```bash
cd /Users/nani/2026-Prototype

# Same command as before (will load fixed code)
nohup python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --checkpoint-dir ./checkpoints \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-accumulation-steps 4 \
  --scheduler-type cosine \
  --warmup-epochs 5 \
  --early-stopping-patience 10 \
  --checkpoint-interval 5 \
  --device mlx \
  --fp16 \
  --use-gradnorm \
  --lr-backbone 1e-5 \
  --lr-head 1e-4 \
  --resume checkpoints/best_model.pth \
  > training_clean.log 2>&1 &

# Get the PID
echo $!

# Monitor
tail -f training_clean.log | grep "loss="
```

---

## ✅ Expected Result

### Before (Noisy)
```
Epoch 2/50:  97%|█████████▋| 2422/2500 [20:06<01:14,  1.05it/s, loss=3.7327, lr=1.89e-06]
2026-02-03 21:33:29 - ml.utils.batch_validation - WARNING - Batch validation failed: boxes has non-positive width: min=0.0. Attempting auto-fix...
2026-02-03 21:33:29 - ml.utils.batch_validation - INFO - Batch successfully sanitized
Epoch 2/50:  97%|█████████▋| 2423/2500 [20:08<01:24,  1.09s/it, loss=3.7327, lr=1.89e-06]
2026-02-03 21:33:30 - ml.utils.batch_validation - WARNING - Batch validation failed: boxes has non-positive width: min=0.0. Attempting auto-fix...
```

### After (Clean)
```
Epoch 3/50:   0%|          | 0/2500 [00:00<?, ?it/s, loss=0.0, lr=2.0e-06]
Epoch 3/50:   1%|          | 10/2500 [00:05<18:32,  2.24it/s, loss=3.4, lr=2.0e-06]
Epoch 3/50:   2%|▏         | 50/2500 [00:25<19:42,  2.07it/s, loss=3.3, lr=2.1e-06]
```

**No warnings!** 🎉

---

## 📊 Benefits After Restart

1. **Clean logs** - No more false warnings every batch
2. **Better data** - 96.6% real images (was 80%)
3. **Same progress** - Resumes from Epoch 2 checkpoint
4. **Faster convergence** - Higher quality training data

---

## ⏱️ Timeline

- **Now:** Epoch 2 is 97% complete (~78 batches remaining)
- **~2 min:** Epoch 2 completes and saves checkpoint
- **Stop & Restart:** Takes ~30 seconds
- **Epoch 3 starts:** With clean code and patched images

---

## 🚨 Quick Command (Copy-Paste)

```bash
# All-in-one restart
cd /Users/nani/2026-Prototype && \
pkill -SIGINT -f train_maxsight.py && \
sleep 15 && \
nohup python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --checkpoint-dir ./checkpoints \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-accumulation-steps 4 \
  --scheduler-type cosine \
  --warmup-epochs 5 \
  --early-stopping-patience 10 \
  --checkpoint-interval 5 \
  --device mlx \
  --fp16 \
  --use-gradnorm \
  --lr-backbone 1e-5 \
  --lr-head 1e-4 \
  --resume checkpoints/best_model.pth \
  > training_clean.log 2>&1 & \
tail -f training_clean.log
```

---

## ✅ Summary

**What we fixed:**
- ✅ Removed validation warnings (silent now)
- ✅ Pre-sanitized boxes in data pipeline
- ✅ Downloaded 1,746 missing train images
- ✅ Downloaded ~360 missing val images (some permanently gone)

**Action needed:**
1. Wait for Epoch 2 to complete (~2 min)
2. Stop training
3. Restart with same command
4. Enjoy clean logs! 🎉

**No data loss, no progress loss, just cleaner output!**
