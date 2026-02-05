# MaxSight Training Runbook

**Quick reference for running and monitoring training.**

---

## Sanity Run (Test Before Full Training)

**Run 1 epoch with small batch to catch errors early:**

```bash
python scripts/train_maxsight.py \
  --epochs 1 \
  --batch-size 4 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

**What to check:**
- ✅ No OOM errors
- ✅ No import errors
- ✅ Model loads
- ✅ Forward pass works
- ✅ Loss computes (no NaN)

**If this passes, proceed to full training.**

---

## Full Training Run

**Standard production training:**

```bash
python scripts/train_maxsight.py \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto \
  --grad-clip 1.0
```

**Colab (with GPU):**
```bash
python scripts/train_maxsight.py \
  --epochs 20 \
  --batch-size 16 \
  --data-dir /content/drive/MyDrive/MaxSight/datasets \
  --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
  --device cuda \
  --grad-clip 1.0
```

**Key parameters:**
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 16, adjust based on GPU memory)
- `--device auto`: Auto-select best device (cuda > cpu)
- `--grad-clip`: Gradient clipping threshold (default: 1.0)
- `--checkpoint-dir`: Where to save checkpoints (default: `./checkpoints`)

---

## Resume Training

**If training was interrupted, resume from last checkpoint:**

```bash
python scripts/train_maxsight.py \
  --resume \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

**Resume from specific checkpoint:**
```bash
python scripts/train_maxsight.py \
  --resume-from checkpoints/final_model.pt \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

**Resume model only (use new optimizer/LR):**
```bash
python scripts/train_maxsight.py \
  --resume-from checkpoints/final_model.pt \
  --resume-model-only \
  --epochs 20 \
  --batch-size 16 \
  --data-dir datasets \
  --checkpoint-dir checkpoints \
  --device auto
```

---

## Monitoring Training

### Checkpoints

**Location:** `checkpoints/`

**Files:**
- `last_checkpoint.pt` - Latest checkpoint (updated every epoch)
- `best_model.pt` - Best validation loss checkpoint
- `final_model.pt` - Final checkpoint (saved at end)
- `training_history.json` - Training metrics history

**Check checkpoint size:**
```bash
ls -lh checkpoints/*.pt
```

### Logs

**Location:** `logs/` (if logging configured)

**Watch training progress:**
```bash
tail -f logs/training_*.log
```

**Or check console output** - training script prints:
- Epoch progress
- Train/val loss
- Validation mAP scores
- Checkpoint saves

### Key Metrics to Watch

- **Train Loss**: Should decrease over time
- **Val Loss**: Should decrease (may fluctuate)
- **Val mAP**: Should increase (target: 30-50%+)
- **Val mAP@0.5**: IoU threshold 0.5 (most important)
- **Val mAP@0.75**: IoU threshold 0.75 (higher precision)

**Warning signs:**
- Loss increases → Learning rate too high
- Loss NaN → Gradient explosion (reduce LR or increase grad-clip)
- OOM errors → Reduce batch size
- Val loss much higher than train → Overfitting (add regularization)

---

## Expected Training Time

**On GPU (CUDA):**
- 1 epoch: ~30-60 minutes (depends on GPU)
- Full 20 epochs: ~10-20 hours

**On CPU:**
- 1 epoch: ~5-10 hours
- Full 20 epochs: ~100-200 hours (not recommended)

**Recommendation:** Use Colab GPU or local GPU for training.

---

## Troubleshooting

### Out of Memory (OOM)

**Reduce batch size:**
```bash
--batch-size 8  # or 4, or 2
```

**Use gradient accumulation:**
```bash
--batch-size 4 --grad-accumulation-steps 4  # Effective batch size = 16
```

### Training Too Slow

**Use GPU:**
```bash
--device cuda  # Explicitly use CUDA
```

**Increase batch size (if memory allows):**
```bash
--batch-size 32  # Faster training, more memory
```

### Loss Not Decreasing

**Check learning rate:**
- Default LR should work, but can adjust with `--lr` flag
- Try lower LR: `--lr 0.0001`

**Check data:**
- Verify data pipeline: `python scripts/validate_data_pipeline.py`
- Ensure images are loading correctly

### Checkpoint Issues

**If checkpoint won't load:**
- Check file exists: `ls -lh checkpoints/last_checkpoint.pt`
- Check file size (should be ~600MB-1GB)
- Try `--resume-from` with explicit path

---

## After Training Completes

**Check final results:**
```bash
# View training history
cat checkpoints/training_history.json | python -m json.tool

# Check best model
ls -lh checkpoints/best_model.pt
ls -lh checkpoints/final_model.pt
```

**Next steps:**
1. Export model for Xcode (see export documentation)
2. Run inference evaluation on test datasets
3. Deploy to Xcode app

---

## Quick Commands Reference

```bash
# Sanity test
python scripts/train_maxsight.py --epochs 1 --batch-size 4 --device auto

# Full training
python scripts/train_maxsight.py --epochs 20 --batch-size 16 --device auto

# Resume
python scripts/train_maxsight.py --resume --epochs 20 --batch-size 16 --device auto

# Check progress
ls -lh checkpoints/
tail -f logs/training_*.log
```
