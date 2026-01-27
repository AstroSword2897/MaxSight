# Testing Guide for Integration Features

**GradNorm + Timing Enforcement**

---

## 🎯 Purpose

This guide ensures that:

- **GradNorm** is correctly wired into training, not just imported
- **Timing enforcement** measures real latency and safely degrades behavior when needed
- **Structural correctness** and **runtime behavior** are validated separately
- **Failures** are easy to diagnose and fix

**If all tests pass, the system is safe to integrate into real training and inference pipelines.**

---

## ✅ Prerequisites (Clearer + Stricter)

### Install Dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Core Environment (DO NOT SKIP)

```bash
python - << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")

from ml.training.task_balancing import GradNormMultiHeadLoss
print("GradNorm available: OK")
EOF
```

**🚨 If this fails, stop here. Nothing else will work correctly.**

---

## 🧪 Test Suite Overview

| Test Type | Requires PyTorch | Purpose |
|-----------|------------------|---------|
| **Structural** | ❌ No | Verifies integration points in code |
| **GradNorm** | ✅ Yes | Confirms loss balancing works |
| **Timing** | ✅ Yes | Confirms latency-based behavior |

---

## 1️⃣ Structural Tests (No PyTorch)

```bash
python tests/test_integration_structure.py
```

### What This Actually Guarantees

**If these pass, you did not mess up the wiring of your system.**

Specifically, it confirms:

- GradNorm is correctly imported (no broken paths)
- Training loop accepts the right parameters
- Timing module is properly included
- Stage A latency is measured correctly
- Model outputs always contain:
  - `stage_a_completed`
  - `stage_b_completed`
  - `skip_stage_b_reason`
  - `stage_a_latency_ms`

### Expected Output

```
✅ PASS: GradNorm Import
✅ PASS: GradNorm Parameters
✅ PASS: GradNorm Integration
✅ PASS: Timing Import
✅ PASS: Timing Code
✅ PASS: Timing Threshold
✅ PASS: Output Structure

Total: 7/7 tests passed
🎉 All structural tests passed!
```

**🚨 If any fail: your integration is broken. Fix imports, constructor args, or output dicts before proceeding.**

---

## 2️⃣ GradNorm Integration Tests (Requires PyTorch)

```bash
python tests/test_gradnorm_integration.py
```

### What This Proves

These tests confirm GradNorm is **not just present, but functional**:

- The class actually initializes
- It integrates into your training loop
- Loss computation works with real tensors

### Expected Output

```
✅ GradNormMultiHeadLoss imported successfully
✅ GradNorm initialized successfully
✅ Training loop created with GradNorm enabled
✅ GradNorm loss computation successful
```

### What a Failure Likely Means (Critical)

If this fails, expect one of these issues:

- **GradNorm constructor mismatch** - Check parameter names/types
- **Loss format incompatibility** - Loss function must return dict or compatible format
- **Training loop calling loss incorrectly** - Check `compute_multihead_loss()` implementation
- **Missing GradNorm availability check** - Verify `GRADNORM_AVAILABLE` flag

**Fix those before moving forward.**

### Failure Examples

**If GradNorm is broken, you might see:**

- All task weights remain constant during training (weights not updating)
- NaN losses appearing after a few steps (unstable balancing)
- Training loop crashing when `use_gradnorm=True` (integration bug)
- Import error: `ModuleNotFoundError: No module named 'ml.training.task_balancing'` (path issue)

---

## 3️⃣ Timing Enforcement Tests (Requires PyTorch)

```bash
python tests/test_timing_enforcement.py
```

### What This Validates

These tests ensure your model **actually behaves differently** based on latency, not just records it.

Specifically, they check:

- `time` is properly imported
- `_enable_timing` actually changes behavior
- Stage A latency is measured correctly
- Stage B is skipped when latency > 200ms
- Timing is truly disabled when `_enable_timing=False`

### Expected Output

```
✅ Time module available
✅ _enable_timing flag can be set
✅ Timing tracking test completed
✅ Timing enforcement test completed
✅ Timing disabled mode works correctly
✅ Latency measurement successful
   - Average latency: XX.XXms
   - Target: <150ms, Hard limit: <200ms
```

### Interpreting Results

- **< 150ms average** → Your model is comfortably within target
- **150–200ms** → You're flirting with the limit. Expect occasional Stage B skips
- **> 200ms** → Stage B will often be skipped. You need optimization

**If average latency is consistently above 150ms, expect frequent Stage B skips.**

### Failure Examples

**If timing is broken, you might see:**

- `stage_a_latency_ms` always missing from outputs (timing not enabled or not measured)
- Stage B running even when latency > 200ms (gating logic broken)
- `_enable_timing=True` having no effect (flag not checked)
- Timing overhead even when disabled (timing code not properly gated)

---

## 🔬 Manual Testing (Realistic Use Cases)

### A) GradNorm in Real Training

```python
from ml.training.train_loop import ProductionTrainLoop
from ml.models.maxsight_cnn import create_model
from ml.training.losses import DetectionLoss
from torch.utils.data import DataLoader

model = create_model(num_classes=80)
train_loader = DataLoader(...)  # YOUR real dataset
loss_fn = DetectionLoss(num_classes=80)

trainer = ProductionTrainLoop(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
    device='cpu',  # or 'mps' or 'cuda'
    use_gradnorm=True,
    gradnorm_alpha=1.5,
    gradnorm_update_interval=100
)

trainer.train()
```

### What to Watch For

**During training, you should see:**

- ✅ Task weights changing over time
- ✅ More balanced per-head losses
- ✅ No exploding/NaN losses

**If training becomes unstable:**

- Try `gradnorm_alpha = 1.0` (lower restoring force)
- Or increase `gradnorm_update_interval = 200` (update less frequently)

### Understanding GradNorm

**GradNorm dynamically adjusts task weights so that tasks with slower learning progress get higher weighting over time.**

This prevents gradient warfare where dominant tasks (like detection) overwhelm rare tasks (like fatigue detection).

---

### B) Timing Enforcement in Inference

```python
from ml.models.maxsight_cnn import create_model
import torch

model = create_model(num_classes=80)
model.eval()
model._enable_timing = True

images = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(images)

print(f"Stage A latency: {outputs.get('stage_a_latency_ms', 'N/A')}ms")
print(f"Stage B completed: {outputs.get('stage_b_completed', False)}")
print(f"Skip reason: {outputs.get('skip_stage_b_reason', 'N/A')}")

# Verify correctness
if outputs.get('stage_a_latency_ms', 0) > 200:
    assert not outputs.get('stage_b_completed', True), "Stage B should be skipped!"
    print("✅ Timing enforcement working: Stage B skipped due to high latency")
```

### What Correct Behavior Looks Like

- **If latency ≤ 200ms** → Stage B likely runs
- **If latency > 200ms** → Stage B should be skipped with reason `'high_latency'`

**If that doesn't happen, your gating logic is wrong.**

---

## ⚙️ Expected System Behavior

### GradNorm

#### Enabled (`use_gradnorm=True`)

- ✅ Training loop accepts the flag
- ✅ Task weights adjust dynamically
- ✅ Multi-head losses are balanced
- ✅ Training remains stable

#### Disabled

- ✅ Training behaves exactly as before
- ✅ Zero performance penalty
- ✅ No breaking changes

### Timing Enforcement

#### Enabled (`_enable_timing=True`)

- ✅ Stage A latency is always recorded
- ✅ Stage B is skipped if Stage A > 200ms
- ✅ Output dict contains:
  - `stage_a_latency_ms`
  - `stage_b_completed`
  - `skip_stage_b_reason`

#### Disabled

- ✅ Model behaves normally
- ✅ No timing overhead
- ✅ Backward compatible

---

## 🛠 Troubleshooting

### PyTorch Not Installed

```bash
pip install torch torchvision torchaudio
```

**Apple Silicon (MPS) should work automatically.**

### Import Errors

```bash
cd /path/to/2026-Prototype
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**If that still fails, your module structure is broken.**

### GradNorm Not Found

```bash
ls ml/training/task_balancing.py
python -c "from ml.training.task_balancing import GradNormMultiHeadLoss; print('OK')"
```

**If this fails, your file path or class name is wrong.**

### Timing Not Working

```python
# Check if timing is enabled
model._enable_timing = True
assert hasattr(model, '_enable_timing'), "Timing flag not found"

# Check if time module is imported
import ml.models.maxsight_cnn as m
assert hasattr(m, 'time'), "Time module not imported"
```

---

## 📊 Test Results Interpretation

### Structural Tests

- **All Pass** → Integration is wired correctly
- **Any Fail** → Your code is broken at the integration level

### Functional Tests

- **All Pass** → Features work in practice
- **Any Fail** → You likely have:
  - A training loop bug, or
  - A timing logic bug

**Fix those before deploying.**

---

## 🚀 Next Steps (More Realistic)

After passing all tests:

1. **Train on a small real dataset**
2. **Measure:**
   - Training stability with GradNorm
   - Average inference latency
3. **If needed:**
   - Tune `gradnorm_alpha`
   - Adjust 200ms threshold
4. **Only then move to production**

---

## 📈 Performance Expectations

### GradNorm

**Expected behavior:**
- Task weights should change during training (not stay constant)
- Losses should be more balanced across heads
- Training should remain stable (no NaN, no explosions)

**If unstable:**
- Lower `gradnorm_alpha` (1.5 → 1.0 or 0.8)
- Increase `gradnorm_update_interval` (100 → 200 or 500)

### Timing Enforcement

**Expected latency:**
- **Target:** <150ms for Stage A
- **Hard limit:** 200ms (Stage B skipped if exceeded)

**If average latency is consistently above 150ms, expect frequent Stage B skips.**

**If skipping too often:**
- Optimize Stage A (reduce model complexity)
- Consider slightly higher threshold (e.g., 220ms instead of 200ms)
- Profile bottleneck (use `torch.profiler`)

---

## 🎓 Key Concepts

### GradNorm Explained

**GradNorm dynamically adjusts task weights so that tasks with slower learning progress get higher weighting over time.**

This prevents gradient warfare where:
- Dominant tasks (detection) overwhelm rare tasks (fatigue)
- Some heads stop learning (gradient starvation)
- Training becomes unstable (conflicting gradients)

**How it works:**
1. Computes gradient norms for each task
2. Compares to average gradient norm
3. Adjusts task weights to balance learning rates
4. Updates weights every N iterations

### Timing Enforcement Explained

**Two-stage inference with latency safeguards:**

- **Stage A (Safety Pass):** <150ms target, <200ms hard limit
  - Tier 1 heads only (safety-critical)
  - Always runs, never skipped
  - Answers: "Is the user safe now?"

- **Stage B (Context Pass):** Opportunistic
  - Tier 2/3 heads (navigation, enhancement)
  - Skipped if Stage A > 200ms or uncertainty > 0.7
  - Provides rich context (can be skipped safely)

**Why this matters:**
- Safety-first: Tier 1 always runs
- Graceful degradation: System continues if Stage B skipped
- Predictable behavior: Users know safety features always work

---

**Last Updated**: January 2026  
**Status**: Production-Ready Testing Guide
