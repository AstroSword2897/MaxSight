# Colab Training Commands - Ready to Run

**Your code is pushed to GitHub.** Use these commands in Colab after pulling the latest code.

---

## 🚀 Quick Start (Recommended)

### 1. Mount Drive & Set Paths
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/2026-Prototype')

# Set your data paths
DATA_DIR = "/content/drive/MyDrive/MaxSight_Training"
IMAGE_DIR = DATA_DIR  # Or separate if images are elsewhere
```

### 2. Install Dependencies (if needed)
```python
!pip install -q -r requirements_colab.txt
```

### 3. Quick Training (T5 with all fixes)
```python
!python scripts/train_maxsight.py \
    --data-dir {DATA_DIR} \
    --image-dir {IMAGE_DIR} \
    --batch-size 8 \
    --epochs 10 \
    --num-workers 2 \
    --use-gradnorm \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --device cuda
```

**Note**: EMA and mixed precision are enabled by default in the training loop (hardcoded for stability).

---

## 🎯 Full Training Options

### Standard Training (with all T5 fixes)
```python
!python scripts/train_maxsight.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 8 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 2 \
    --grad-accumulation-steps 4 \
    --use-gradnorm \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --checkpoint-interval 5 \
    --early-stopping-patience 10 \
    --device cuda \
    --seed 42
```

**Note**: 
- EMA is enabled by default (decay=0.9999)
- Mixed precision is disabled for stability (FP32 only)
- GradNorm alpha defaults to 1.5

### Fast Testing (smoke test)
```python
!python scripts/smoke_train.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 4 \
    --epochs 2 \
    --num-samples 50 \
    --device cuda
```

### Resume from Checkpoint
```python
!python scripts/train_maxsight.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 8 \
    --epochs 30 \
    --resume-from /content/drive/MyDrive/MaxSight/checkpoints/last_checkpoint.pt \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --use-gradnorm \
    --device cuda
```

### Resume Model Only (new optimizer/scheduler)
```python
!python scripts/train_maxsight.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 8 \
    --epochs 30 \
    --learning-rate 5e-5 \
    --resume-from /content/drive/MyDrive/MaxSight/checkpoints/last_checkpoint.pt \
    --resume-model-only \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --use-gradnorm \
    --device cuda
```

---

## 🧪 AutoML (Hyperparameter Search)

```python
!python scripts/check_and_train_colab.py

# Or set environment variables:
import os
os.environ['MODE'] = 'automl'
os.environ['N_TRIALS'] = '10'
os.environ['EPOCHS_PER_TRIAL'] = '3'
os.environ['BATCH_SIZE'] = '8'

!python scripts/check_and_train_colab.py
```

---

## 📊 Monitor Training

### Check Training Progress
```python
# View training history
import pandas as pd
history = pd.read_csv('/content/drive/MyDrive/MaxSight/checkpoints/training_history.csv')
print(history.tail())

# Plot losses
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Losses')

plt.subplot(1, 2, 2)
plt.plot(history['val_map'], label='mAP')
plt.plot(history['val_precision'], label='Precision')
plt.plot(history['val_recall'], label='Recall')
plt.legend()
plt.title('Metrics')
plt.show()
```

### Check GradNorm Task Weights
```python
# View logs to see GradNorm weights (logged every 500 steps)
!tail -100 /content/drive/MyDrive/MaxSight/checkpoints/training.log | grep "GradNorm"
```

### Check GPU Usage
```python
!nvidia-smi
```

---

## 🔧 Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 8 | Batch size per GPU |
| `--epochs` | 20 | Number of training epochs |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--grad-accumulation-steps` | 1 | Accumulate gradients over N steps |
| `--use-gradnorm` | False | Enable GradNorm (RECOMMENDED for T5) |
| `--grad-clip` | 5.0 | Max gradient norm for clipping |
| `--checkpoint-interval` | 0 | Save checkpoint every N epochs (0=only best) |
| `--early-stopping-patience` | 10 | Stop if no improvement for N epochs |
| `--freeze-backbone-epochs` | 0 | Freeze backbone for first N epochs |
| `--warmup-epochs` | 5 | LR warmup epochs |
| `--scheduler-type` | cosine | LR scheduler (cosine/onecycle/cosine_restarts) |

---

## ✅ What's Fixed (You Have All These)

All **8 T5 training loop fixes** are applied:
1. ✅ `epoch_losses` tracking for stability manager
2. ✅ GradNorm validation mode (no gradient corruption)
3. ✅ Device type handling (CPU/CUDA/MPS compatible)
4. ✅ Early stopping with proper baseline
5. ✅ Fast NaN checking with `math.isnan()`
6. ✅ T5 batch validation (multi-modal)
7. ✅ Config validation (catches bad hyperparameters)
8. ✅ GradNorm weight logging (every 500 steps)

Plus:
- ✅ `SpatialMemorySystem` integrated
- ✅ `ReadinessMonitor` integrated
- ✅ All 17 architecture components active

---

## 🎓 Recommended Settings for T5

### For First Training Run
```python
!python scripts/train_maxsight.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 8 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --grad-accumulation-steps 4 \
    --use-gradnorm \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --checkpoint-interval 5 \
    --device cuda
```

### For Fine-tuning (after initial training)
```python
!python scripts/train_maxsight.py \
    --data-dir /content/drive/MyDrive/MaxSight_Training \
    --image-dir /content/drive/MyDrive/MaxSight_Training \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --resume-from /content/drive/MyDrive/MaxSight/checkpoints/best_model.pt \
    --freeze-backbone-epochs 0 \
    --use-gradnorm \
    --checkpoint-dir /content/drive/MyDrive/MaxSight/checkpoints \
    --device cuda
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size and increase gradient accumulation
--batch-size 4 --grad-accumulation-steps 8

# Or reduce number of workers
--num-workers 0
```

### Training Too Slow
```python
# Increase workers
--num-workers 4

# Increase batch size (if you have memory)
--batch-size 16

# Use torch.compile (CUDA only)
--compile
```

### NaN Losses
```python
# Lower learning rate
--learning-rate 5e-5

# Lower gradient clip norm (default is 5.0)
--grad-clip 1.0

# Check logs for gradient norms
```

### GradNorm Not Working
```python
# Make sure --use-gradnorm is set
# Check logs for "GradNorm enabled" message
# View task weights: grep "GradNorm weights" training.log
```

---

## 📝 Notes

- **Save to Drive**: Always use `/content/drive/MyDrive/...` for checkpoints so they persist
- **GradNorm Recommended**: For T5's 15 heads, GradNorm prevents gradient warfare (use `--use-gradnorm`)
- **EMA Active**: EMA is enabled by default (decay=0.9999) for better model quality
- **FP32 Only**: Mixed precision disabled for stability (hardcoded in script)
- **Checkpoint Interval**: Use `--checkpoint-interval 5` to save every 5 epochs

---

## 🆘 If Training Fails

1. Check data paths exist
2. Check annotation JSONs are valid
3. Check GPU available: `!nvidia-smi`
4. Check disk space: `!df -h`
5. Try smoke test first: `scripts/smoke_train.py`

**Your code is ready!** All T5 fixes applied and pushed to GitHub. 🚀
