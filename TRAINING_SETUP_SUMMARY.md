# Training Setup Summary

## ✅ Completed Tasks

### 1. COCO Download Troubleshooting ✅
- **Status**: Link verified, setup scripts created
- **Files Created**:
  - `scripts/setup_coco_data.py` - Extract zips and verify dataset
  - `scripts/download_coco.py` - Download with multiple fallback methods
- **Current Status**: 
  - ✅ Annotations complete (118K train, 5K val)
  - ✅ Val images extracted (5000 images)
  - ❌ Train images missing (need train2017.zip ~18GB)

**Next Step**: Download train2017.zip from http://images.cocodataset.org/zips/train2017.zip

### 2. Data Pipeline Setup ✅
- **Status**: Complete
- **Files Created**:
  - `ml/data/data_pipeline.py` - Data loader creation, collation, class weights
  - `scripts/setup_training_data.py` - Create train/val/test splits from COCO
- **Features**:
  - Custom collate function for variable-length sequences
  - Support for multi-modal data (images + audio)
  - Class weight computation for imbalanced datasets
  - Auto-detection of image directories

### 3. Training Configuration Files ✅
- **Status**: Complete
- **Files Created**:
  - `ml/training/configs/t0_baseline.yaml` - T0 baseline config
  - `ml/training/configs/t1_attention.yaml` - T1 attention config
  - `ml/training/configs/t2_hybrid_vit.yaml` - T2 hybrid ViT config
  - `ml/training/configs/t3_cross_task.yaml` - T3 cross-task config
  - `ml/training/configs/t4_cross_modal.yaml` - T4 cross-modal config
  - `ml/training/configs/t5_temporal.yaml` - T5 temporal config
  - `ml/training/configs/README.md` - Configuration documentation

### 4. Training Pipeline Test Script ✅
- **Status**: Created and ready
- **File**: `scripts/test_training_pipeline.py`
- **Features**:
  - Tests data loaders
  - Tests model creation
  - Tests forward pass
  - Tests loss computation
  - Tests training steps
  - Supports YAML config files

## 🚀 Production run (one-shot)

Full training is started via:

```bash
# Default: DEVICE=auto (Apple GPU on Mac, CUDA on Linux if available, else CPU)
./scripts/run_production_training.sh --no-export

# Explicit CPU (MPS disabled due to backward errors)
DEVICE=cpu EPOCHS=100 ./scripts/run_production_training.sh --no-export

# Background (log to file) — process continues if you close the terminal
nohup ./scripts/run_production_training.sh --no-export > training_run.log 2>&1 &
tail -f training_run.log   # monitor; Ctrl+C only stops tail, not training

# Or use tmux so you can detach and reattach from another session
tmux new -s maxsight
DEVICE=mps BATCH_SIZE=1 NUM_WORKERS=2 EPOCHS=28 ./scripts/run_production_training.sh --no-export
# Detach: Ctrl+B then D. Reattach later: tmux attach -t maxsight
```

- **Checkpoints**: `./checkpoints/` (e.g. `best.pt`, `last.pt`).
- **Device**: default is `auto` (CUDA when available, else CPU). MPS is **disabled** due to backward pass errors. Override with `DEVICE=cuda` or `DEVICE=cpu`.
- **Export after training**: run without `--no-export`, or run `python -m ml.training.export --checkpoint checkpoints/best.pt --format jit --output exports/maxsight_jit.pt`.

### Stable CPU run (lower memory)

For CPU training with lower memory usage:

```bash
DEVICE=mps BATCH_SIZE=1 NUM_WORKERS=2 EPOCHS=100 ./scripts/run_production_training.sh --no-export
```

If crashes continue, try single-threaded loading:

```bash
DEVICE=mps BATCH_SIZE=1 NUM_WORKERS=0 ./scripts/run_production_training.sh --no-export
```

Override any of: `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `DEVICE`, `LR`, `WEIGHT_DECAY`.

### 12–18 hour run (fewer epochs)

To get a finished run in about 12–18 hours on CPU with BATCH_SIZE=1, use **25–30 epochs** instead of 100:

```bash
DEVICE=mps BATCH_SIZE=1 NUM_WORKERS=2 EPOCHS=28 ./scripts/run_production_training.sh --no-export
```

Rough timing: ~35–45 min per epoch × 28 epochs ≈ 16–21 hours. For a bit faster (≈12–15 h), use `EPOCHS=20`. Checkpoints still go to `./checkpoints/` (e.g. `best.pt`, `last.pt`).

### MLX-style / T5-style full convergence (Mac)

**Goal:** Stable, full convergence on Mac with gradient accumulation, separate backbone/head LRs, cosine scheduler, and early stopping. `DEVICE=mlx` uses **CPU** (MPS disabled due to backward errors).

| Parameter | Value | Notes |
|-----------|--------|--------|
| **DEVICE** | `mlx` or `cpu` | CPU only; MPS disabled due to errors |
| **BATCH_SIZE** | 4 | Per-step; safe for Mac RAM |
| **GRAD_ACC** | 4 | Effective batch = 16 |
| **EPOCHS** | 50 | Extend to 60–70 if loss hasn’t plateaued |
| **LR_BACKBONE** | 1e-5 | Backbone (encoder) LR |
| **LR_HEAD** | 1e-4 | Head layers |
| **SCHEDULER** | cosine | With warmup |
| **WARMUP_EPOCHS** | 5 | ~10% of 50 epochs |
| **CHECKPOINT_INTERVAL** | 5 | Save snapshot every 5 epochs |
| **EARLY_STOPPING_PATIENCE** | 10 | Stop if no improvement for 10 epochs |
| **NUM_WORKERS** | 2 | Reduce to 0 if memory issues |

**One command (plug-and-play):**

```bash
./scripts/run_mlx_style_training.sh --no-export
```

**Or with env overrides:**

```bash
DEVICE=mlx BATCH_SIZE=4 GRAD_ACC=4 EPOCHS=50 NUM_WORKERS=2 \
LR_BACKBONE=1e-5 LR_HEAD=1e-4 SCHEDULER=cosine WARMUP_EPOCHS=5 \
CHECKPOINT_INTERVAL=5 EARLY_STOPPING_PATIENCE=10 \
./scripts/run_production_training.sh --no-export
```

**Resume from last checkpoint:**

```bash
RESUME_FROM=checkpoints/last_checkpoint.pt ./scripts/run_mlx_style_training.sh --no-export
```

**Resume from previous checkpoint with MLX-style (new LRs, batch 4, grad acc 4):**

**Quick way (after you’ve stopped the run):** run the all-in-one script — it backs up checkpoints if needed and starts MLX-style with model-only resume:

```bash
./scripts/resume_mlx_from_first3_mps.sh --no-export
```

**Step-by-step (explicit):**

1. **Stop the current run safely**  
   Find PID: `ps aux | grep run_production_training`  
   Then: `kill -SIGINT <PID>` (or Ctrl+C in the training terminal). Do not just close the terminal or you may lose the last epoch.

2. **Back up checkpoints and logs**  
   Checkpoint files in this repo are **`last_checkpoint.pt`** and **`best_model.pt`** (not `last.pt` / `best.pt`):
   ```bash
   mkdir -p backups/first3_epochs
   cp checkpoints/last_checkpoint.pt backups/first3_epochs/
   [ -f checkpoints/best_model.pt ] && cp checkpoints/best_model.pt backups/first3_epochs/
   ```
   Save training log up to end of epoch 3 (if log has reached "Epoch 4"):
   ```bash
   grep -n "Epoch 4" training_run.log | head -1 | cut -d: -f1 | xargs -I {} head -n {} training_run.log > backups/first3_epochs/epoch_losses.txt
   ```
   Or save all loss-related lines:  
   `grep -E "Epoch|Loss|Train Loss|Val Loss" training_run.log > backups/first3_epochs/epoch_losses.txt`

3. **Run MLX fine-tuning (resume from first 3 epochs with new config)**  
   Use **RESUME_MODEL_ONLY=1** so the new optimizer, scheduler, batch size, and LRs apply (model weights come from the backup):
   ```bash
   DEVICE=mlx BATCH_SIZE=4 GRAD_ACC=4 EPOCHS=50 NUM_WORKERS=2 \
   LR_BACKBONE=1e-5 LR_HEAD=1e-4 SCHEDULER=cosine WARMUP_EPOCHS=5 \
   CHECKPOINT_INTERVAL=5 RESUME_FROM=backups/first3_epochs/last_checkpoint.pt RESUME_MODEL_ONLY=1 \
   ./scripts/run_production_training.sh --no-export
   ```
   Or use the MLX-style script with resume env vars:
   ```bash
   RESUME_FROM=backups/first3_epochs/last_checkpoint.pt RESUME_MODEL_ONLY=1 ./scripts/run_mlx_style_training.sh --no-export
   ```

4. **Monitor:** `tail -f training_run.log`  
   Checkpoints every 5 epochs in `./checkpoints/`. If val loss &lt; 1.0 for 5 consecutive epochs, consider early stopping.

5. **Optional — run in background:**
   ```bash
   nohup env DEVICE=mlx BATCH_SIZE=4 GRAD_ACC=4 EPOCHS=50 NUM_WORKERS=2 LR_BACKBONE=1e-5 LR_HEAD=1e-4 SCHEDULER=cosine WARMUP_EPOCHS=5 CHECKPOINT_INTERVAL=5 RESUME_FROM=backups/first3_epochs/last_checkpoint.pt RESUME_MODEL_ONLY=1 ./scripts/run_production_training.sh --no-export > training_run.log 2>&1 &
   tail -f training_run.log
   ```

6. **Early stopping:** Val loss &lt; 1.0 for 5 epochs → stop early. Gradients exploding / loss NaN → kill, reduce batch or LR.

**Safe stop and backup:**

1. Stop with **SIGINT** (Ctrl+C) so the trainer can save a checkpoint:
   ```bash
   kill -SIGINT <PID>
   ```
2. Backup checkpoints and losses:
   ```bash
   mkdir -p backups/run_$(date +%Y%m%d)
   cp checkpoints/last_checkpoint.pt checkpoints/best_model.pt backups/run_$(date +%Y%m%d)/
   grep -E "Epoch|Loss|Train Loss|Val Loss" training_run.log > backups/run_$(date +%Y%m%d)/epoch_losses.txt
   ```
3. Resume later (same or another machine) with `RESUME_FROM=...`.

**Rough expected loss (COCO-style, effective batch 16):**

| Epoch | Train loss (approx) | Val loss (approx) |
|-------|----------------------|--------------------|
| 1 | 5.0–4.2 | 4.8–4.0 |
| 5 | 2.4–2.0 | 2.5–2.2 |
| 10 | 1.6–1.3 | 1.8–1.5 |
| 20 | 1.1–0.9 | 1.2–1.0 |
| 40–50 | 0.8–0.6 | 0.9–0.7 |

Stop early if val loss stays &lt;~1.0 for 5–10 consecutive epochs. Logs: train/val loss each epoch in console and in `training_run.log` if you use `nohup ... > training_run.log 2>&1 &`.

**Timing:** ~10–15 min/epoch on Mac (MLX→CPU, batch 4 + grad accum 4). 50 epochs ≈ 8–12 h.

### Cloud GPU (finish training in hours)

To get results in **hours instead of days**, run the same pipeline on a cloud GPU (CUDA). Much faster than CPU; larger batch size; same checkpoints.

| Where | Best for | Doc |
|-------|----------|-----|
| **Google Colab** | Free T4 GPU, 5 min setup | [QUICK_START_CLOUD.md](QUICK_START_CLOUD.md) |
| **RunPod** | Pay-per-second A100, ~\$0.20/h | [QUICK_START_CLOUD.md](QUICK_START_CLOUD.md#-runpod--lambda) |
| **Lambda Labs** | A100 / H100, simple SSH | [QUICK_START_CLOUD.md](QUICK_START_CLOUD.md#-runpod--lambda) |
| **AWS EC2** | g4dn (T4) or p3 (V100) | [QUICK_START_CLOUD.md](QUICK_START_CLOUD.md#-option-2-aws-ec2-30-minutes) |

**On any cloud (after clone + `pip install -r requirements.txt`):**

```bash
# One-shot: env check, data check, training, optional export
DEVICE=cuda EPOCHS=28 BATCH_SIZE=16 NUM_WORKERS=4 ./scripts/run_production_training.sh --no-export
```

Or run the Python script directly with your data paths (see `QUICK_START_CLOUD.md`). Download checkpoints from the instance (e.g. `scp`, or sync to Google Drive / S3).

### Continue training on another GPU (resume)

You can stop training locally (or on one cloud instance), copy the checkpoint, then **continue on another GPU** (e.g. cloud) from the same loss/epoch.

1. **Copy the checkpoint** from the first run to the second machine:
   - Use `checkpoints/last_checkpoint.pt` (has optimizer, scheduler, epoch, so training continues exactly).
   - Or `checkpoints/best_model.pt` (best validation so far).
   - Example: `scp -r user@local:2026-Prototype/checkpoints ./` or upload `last_checkpoint.pt` to the cloud instance.

2. **On the second machine** (e.g. cloud GPU), run with the same data paths and **resume**:
   ```bash
   # Resume from the file you copied (e.g. into ./checkpoints/last_checkpoint.pt)
   DEVICE=cuda EPOCHS=28 BATCH_SIZE=16 RESUME_FROM=checkpoints/last_checkpoint.pt ./scripts/run_production_training.sh --no-export
   ```
   Or with the Python script:
   ```bash
   python scripts/train_maxsight.py ... --resume-from checkpoints/last_checkpoint.pt --device cuda
   ```

3. **Same machine, next run**: use `RESUME=1` to pick the latest checkpoint in `CHECKPOINT_DIR`:
   ```bash
   RESUME=1 DEVICE=cuda EPOCHS=100 ./scripts/run_production_training.sh --no-export
   ```

The checkpoint stores model, optimizer, scheduler, epoch, and best val loss, so the new run continues from that point (and can use a different batch size or device).

## 📋 Next Steps

### Immediate (Required for Training)

1. **Download COCO Train Images**
   ```bash
   # Option 1: Manual download
   # Visit: http://images.cocodataset.org/zips/train2017.zip
   # Download to: datasets/coco_raw/
   
   # Option 2: Use download script (if link works)
   python scripts/download_coco.py --auto
   
   # Extract
   python scripts/setup_coco_data.py
   ```

2. **Create Training Splits**
   ```bash
   python scripts/setup_training_data.py \
     --train_samples 10000 \
     --val_samples 2000 \
     --test_samples 1000
   ```

3. **Test Training Pipeline**
   ```bash
   # Test with default T0 config
   python scripts/test_training_pipeline.py --num-batches 3
   
   # Test with data dir (if train/val exist)
   python scripts/archive/test_training_pipeline.py --num-batches 3
   ```

### Future Enhancements

1. **Optional YAML config** (not required): `train_maxsight.py` uses explicit args (--data-dir, --train-annotation, --val-annotation, --image-dir). YAML config loading can be added later if desired.

2. **Training Monitoring**
   - TensorBoard integration
   - WandB support (optional)
   - Training metrics dashboard

3. **Distributed Training**
   - Multi-GPU support
   - DDP (Distributed Data Parallel)
   - Gradient synchronization

## 📁 File Structure

```
ml/
├── data/
│   ├── data_pipeline.py          # Data loader creation
│   └── __init__.py               # Updated exports
├── training/
│   └── configs/
│       ├── t0_baseline.yaml
│       ├── t1_attention.yaml
│       ├── t2_hybrid_vit.yaml
│       ├── t3_cross_task.yaml
│       ├── t4_cross_modal.yaml
│       ├── t5_temporal.yaml
│       └── README.md

scripts/
├── setup_coco_data.py            # Extract and verify COCO
├── download_coco.py              # Download COCO dataset
├── setup_training_data.py        # Create train/val/test splits
└── test_training_pipeline.py     # Test training pipeline
```

## 🎯 Training Ready Checklist

- [x] COCO dataset verification script
- [x] COCO download script (with fallbacks)
- [x] Data pipeline module
- [x] Training configuration files (all tiers)
- [x] Training pipeline test script
- [ ] COCO train images downloaded
- [ ] Training splits created
- [ ] Training pipeline tested
- [ ] Full training run (T0 baseline)

## 📝 Usage Examples

### Setup Data
```bash
# 1. Download and extract COCO
python scripts/setup_coco_data.py

# 2. Create training splits
python scripts/setup_training_data.py \
  --train_samples 10000 \
  --val_samples 2000 \
  --test_samples 1000

# 3. Verify setup
python scripts/setup_training_data.py --verify-only --test-loaders
```

### Test Pipeline
```bash
# Test with default settings
python scripts/test_training_pipeline.py

# Test pipeline (script in scripts/archive/)
python scripts/archive/test_training_pipeline.py --num-batches 5
```

### Train Model
```bash
# Use data paths from scripts/gather_training_data.py
python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --epochs 100 --device cuda
```

