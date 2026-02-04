# Training Speed Optimization Guide

**Current Status:**
- Device: CPU (M3 Pro)
- Speed: ~1.5-2.0 it/s
- Batch size: 4
- Workers: 2
- Model: 98.71M parameters
- Time per epoch: ~20-25 minutes

---

## 🚀 Quick Wins (Easy to Implement)

### 1. Increase Data Loading Workers ⭐ **BEST**
**Current:** `--num-workers 2`  
**Recommended:** `--num-workers 4` or `6`

```bash
# Your M3 Pro has multiple cores
--num-workers 6
```

**Expected gain:** 20-30% faster (data loading becomes parallel)  
**Risk:** Low  
**Effort:** 1 minute

### 2. Increase Batch Size ⭐
**Current:** `--batch-size 4`  
**Try:** `--batch-size 8` or `12`

```bash
--batch-size 8  # Start here
# If no memory issues, try:
--batch-size 12
```

**Expected gain:** 30-50% faster (better GPU/CPU utilization)  
**Risk:** May run out of RAM (monitor with Activity Monitor)  
**Effort:** 1 minute

**Note:** If you increase batch size, reduce gradient accumulation to maintain effective batch size:
- Current effective batch: 4 × 4 = 16
- New: batch 8 × accum 2 = 16 (same effective size)
- Or: batch 16 × accum 1 = 16

### 3. Enable Model Compilation
**Add:** `--compile`

```bash
--compile  # Uses torch.compile for optimization
```

**Expected gain:** 10-20% faster (fuses operations)  
**Risk:** Low (may take longer to start)  
**Effort:** 1 minute

### 4. Reduce Checkpoint Frequency
**Current:** `--checkpoint-interval 5` (every 5 epochs)  
**Try:** `--checkpoint-interval 10`

```bash
--checkpoint-interval 10
```

**Expected gain:** Saves I/O time during checkpointing  
**Risk:** None (can still resume)  
**Effort:** 1 minute

---

## 🎯 Medium Wins (Moderate Effort)

### 5. Optimize Model Forward Pass
**Disable expensive heads during training:**

Edit `ml/config.py` to disable some heads you don't need:

```python
# Disable computationally expensive heads
enabled_heads = [
    'classification',
    'box_regression',
    'objectness',
    'urgency',
    'distance',
    # Comment out expensive ones:
    # 'contrast',
    # 'glare',
    # 'findability',
    # 'navigation_difficulty',
]
```

**Expected gain:** 20-40% faster (fewer computations)  
**Risk:** Medium (affects what model learns)  
**Effort:** 5 minutes

### 6. Use Smaller Backbone
**Current:** ResNet-50  
**Alternative:** ResNet-18 or ResNet-34

Modify model creation to use smaller backbone:

```python
# In ml/models/maxsight_cnn.py
backbone = models.resnet18(pretrained=True)  # Instead of resnet50
```

**Expected gain:** 50-70% faster (much smaller model)  
**Risk:** Lower accuracy  
**Effort:** 10 minutes

### 7. Profile and Optimize Bottlenecks

Find slow parts:

```bash
python -m cProfile -o profile.stats scripts/train_maxsight.py [args]

# Analyze
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

**Expected gain:** Varies (find specific bottlenecks)  
**Effort:** 20 minutes

---

## 💰 Big Wins (Hardware/Infrastructure)

### 8. Use Cloud GPU ⭐⭐⭐ **BIGGEST**
**Options:**
- Google Colab (Free T4 GPU): **10-20x faster**
- Kaggle Notebooks (Free GPU): **10-20x faster**  
- Lambda Labs ($0.50/hr): **20-50x faster**
- Vast.ai (Cheap spot GPUs): **20-50x faster**

**Setup Colab:**
```python
# Upload your code to Google Drive
# Mount drive in Colab:
from google.colab import drive
drive.mount('/content/drive')

# Run training with CUDA
!python scripts/train_maxsight.py \
  --device cuda \
  --batch-size 16 \
  [other args]
```

**Expected gain:** **10-50x faster** depending on GPU  
**Cost:** Free (Colab/Kaggle) to $0.50/hr  
**Effort:** 30-60 minutes setup

### 9. Use Apple's MLX Framework
**True Apple Silicon acceleration** (not PyTorch's buggy MPS)

Convert model to MLX:
```bash
# Install MLX
pip install mlx

# Would need to rewrite model in MLX
# Effort: Several days
```

**Expected gain:** 5-10x faster on M3 Pro  
**Risk:** High (major rewrite)  
**Effort:** 2-3 days

### 10. Distributed Training
**Multiple machines** (if available)

```bash
# PyTorch DDP
torchrun --nproc_per_node=2 scripts/train_maxsight.py [args]
```

**Expected gain:** Near-linear with # of GPUs  
**Effort:** Moderate (if you have multiple machines)

---

## 📊 Recommended Quick Command

**Try this now** (20-30% faster, minimal risk):

```bash
cd /Users/nani/2026-Prototype

# Stop current training
pkill -f train_maxsight

# Restart with optimizations
python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --checkpoint-dir ./checkpoints \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 6 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-accumulation-steps 2 \
  --scheduler-type cosine \
  --warmup-epochs 5 \
  --early-stopping-patience 10 \
  --checkpoint-interval 10 \
  --device mlx \
  --fp16 \
  --use-gradnorm \
  --lr-backbone 1e-5 \
  --lr-head 1e-4 \
  --compile \
  > training_fast.log 2>&1 &

tail -f training_fast.log | grep "loss="
```

**Changes:**
- ✅ `--batch-size 8` (was 4)
- ✅ `--num-workers 6` (was 2)
- ✅ `--grad-accumulation-steps 2` (was 4, maintains effective batch 16)
- ✅ `--checkpoint-interval 10` (was 5)
- ✅ `--compile` (new, enables torch.compile)

**Expected:** ~2.5-3.5 it/s (was ~1.5-2.0 it/s) = **50-75% faster**

---

## 🎯 Best Long-Term Solution

**Use Google Colab (Free GPU):**

1. **Upload to Drive:**
   - Zip your code: `tar -czf maxsight.tar.gz ml/ scripts/ datasets/cleaned_splits/`
   - Upload to Google Drive

2. **Create Colab Notebook:**
   ```python
   # Mount drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Extract
   !tar -xzf /content/drive/MyDrive/maxsight.tar.gz
   
   # Install dependencies
   !pip install -r requirements.txt
   
   # Train with GPU
   !python scripts/train_maxsight.py \
     --device cuda \
     --batch-size 32 \
     --num-workers 4 \
     [other args]
   ```

3. **Get results:**
   - **Free T4 GPU:** ~25-35 it/s (15-20x faster!)
   - **Epoch time:** ~2-3 minutes (vs 20-25 min on CPU)
   - **Total training:** ~2-3 hours (vs 35-40 hours on CPU)

---

## 📈 Speed Comparison

| Setup | Speed (it/s) | Epoch Time | Total Time (50 epochs) | Cost |
|-------|-------------|------------|------------------------|------|
| **Current (M3 CPU)** | 1.5-2.0 | 20-25 min | 35-40 hrs | $0 |
| **Optimized (M3 CPU)** | 2.5-3.5 | 12-15 min | 20-25 hrs | $0 |
| **Colab Free (T4 GPU)** | 25-35 | 2-3 min | 2-3 hrs | $0 |
| **Colab Pro (V100)** | 50-80 | 1-2 min | 1-2 hrs | $10/mo |
| **Cloud GPU (A100)** | 100-150 | 30-60 sec | 30-60 min | ~$15 |

---

## 🚦 Priority Actions

### Immediate (Do Now):
1. ✅ Restart with `--num-workers 6 --batch-size 8 --compile`
2. ✅ Monitor speed improvement

### Short-Term (This Week):
1. Set up Google Colab for GPU training
2. Profile code to find bottlenecks
3. Consider disabling non-essential heads

### Long-Term (Future):
1. Convert to MLX for native Apple Silicon support
2. Consider cloud GPU for production training
3. Optimize model architecture

---

## ⚠️ Important Notes

### Memory Monitoring
Watch RAM usage when increasing batch size:
```bash
# Monitor in terminal
watch -n 1 'ps aux | grep train_maxsight | grep -v grep'

# Or use Activity Monitor app
```

If OOM (out of memory):
- Reduce batch size: `--batch-size 6`
- Reduce workers: `--num-workers 4`
- Increase grad accumulation: `--grad-accumulation-steps 3`

### Validation
After changing settings:
- Check loss curves still look normal
- Verify validation metrics don't degrade
- Ensure model still converges

---

## 🎉 Bottom Line

**Quick win (5 min):** Restart with optimized params → **50-75% faster**  
**Best solution (1 hr):** Set up Colab GPU → **15-20x faster, FREE**  
**Ultimate (if budget):** Cloud A100 GPU → **50-100x faster**

**Recommended:** Try optimized CPU params now, then move to Colab if you want serious speed!
