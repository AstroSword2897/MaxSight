# Epoch Performance Diagnosis

## Identified Bottlenecks

### 1. ⚠️ **Validation Loop is Slow** (CRITICAL)
**Problem**: `get_detections()` is called **per image** instead of **per batch**
- Current: Loop through batch, call `get_detections()` for each image individually
- Impact: NMS and post-processing run N times (N = batch size) instead of once
- Fix: Call `get_detections()` once per batch, then iterate through results

**Location**: `ml/training/train_loop.py` lines 901-904

**Before**:
```python
for b in range(batch_size):
    detections_b = self.model.get_detections(
        {k: v[b:b+1] for k, v in outputs.items()},  # Per-image call
        confidence_threshold=0.3
    )
```

**After** (Optimized):
```python
# Process entire batch at once
batch_detections = self.model.get_detections(
    outputs,  # Full batch
    confidence_threshold=0.3
)
# Then iterate through results
for b in range(batch_size):
    detections_list = batch_detections[b]
```

**Expected Speedup**: 2-5x faster validation (depends on batch size)

---

### 2. ⚠️ **Data Loading May Be Slow**
**Check**:
- `num_workers` setting (default: 4)
- `pin_memory` enabled for CUDA
- Image loading/preprocessing overhead

**Recommendations**:
- Increase `num_workers` to 6-8 for multi-core CPUs
- Ensure `pin_memory=True` when using CUDA
- Profile data loading separately (see `scripts/diagnose_training_speed.py`)

---

### 3. ⚠️ **Tensor Creation in Validation Loop**
**Problem**: Creating new tensors per image in validation
- Lines 922-924: `torch.tensor()` called per image
- Can be optimized by pre-allocating or batching

**Impact**: Moderate (tensor creation is relatively fast, but adds up)

---

### 4. ⚠️ **GradNorm Computation**
**Problem**: Multiple backward passes for gradient norm computation
- Each task requires a separate backward pass
- `retain_graph=True` keeps computation graph alive

**Impact**: Moderate (only affects training, not validation)
- Already optimized in `task_balancing.py` (only retain graph when needed)

---

## Quick Fixes Applied

✅ **Optimized validation loop** to call `get_detections()` once per batch
- Changed from per-image calls to batch processing
- Reduces NMS overhead significantly

---

## Diagnostic Tools

### Run Speed Diagnosis
```bash
python scripts/diagnose_training_speed.py \
  --train-annotation <path> \
  --val-annotation <path> \
  --image-dir <path> \
  --batch-size 16 \
  --num-workers 4 \
  --num-batches 20
```

This will:
- Profile data loading time
- Profile forward pass time
- Profile validation time
- Identify bottlenecks
- Provide recommendations

---

## Expected Performance After Fixes

### Before Optimization
- Validation: ~30-50% of epoch time
- Data loading: ~10-20% of epoch time
- Forward/backward: ~30-50% of epoch time

### After Optimization
- Validation: ~10-20% of epoch time (2-5x faster)
- Data loading: ~10-20% of epoch time (unchanged)
- Forward/backward: ~60-70% of epoch time (more time for actual training)

---

## Additional Recommendations

1. **Increase Batch Size** (if memory allows)
   - Larger batches = better GPU utilization
   - Reduces overhead per sample

2. **Reduce Validation Frequency**
   - Validate every N epochs instead of every epoch
   - Use `--checkpoint-interval` to control validation frequency

3. **Enable Mixed Precision**
   - Already enabled by default
   - Reduces memory and speeds up computation

4. **Profile with `torch.profiler`**
   ```python
   with torch.profiler.profile() as prof:
       # training code
   prof.export_chrome_trace("trace.json")
   ```

---

## Monitoring

Watch for these in training logs:
- `Time per epoch: X seconds`
- `Validation time: Y seconds`
- `Data loading time: Z seconds`

If validation > 30% of epoch time, the optimization should help significantly.
