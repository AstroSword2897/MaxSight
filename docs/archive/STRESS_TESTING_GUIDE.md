# MaxSight Stress Testing Guide

**Production-Grade Stress Testing Implementation**

This guide explains how to use the stress testing infrastructure to validate system stability and safety.

---

## Overview

The stress testing infrastructure implements the **MaxSight Stress Testing Playbook** with the following tests:

1. **Head Isolation Stress Tests** - Detect gradient interference
2. **Loss Scaling Stress Tests** - Ensure no loss term dominates
3. **Input Corruption Stress Tests** - Test robustness to real-world conditions
4. **Temporal Stress Tests** - Validate stability across time
5. **Head Dropout Stress Tests** - Ensure graceful degradation

---

## Quick Start

### Running Stress Tests

```bash
# Run full stress test suite
python scripts/run_stress_tests.py --checkpoint checkpoints/best_model.pt

# Run quick tests (skip expensive tests)
python scripts/run_stress_tests.py --checkpoint checkpoints/best_model.pt --quick

# Save report to custom location
python scripts/run_stress_tests.py --checkpoint checkpoints/best_model.pt --output reports/stress_test.json
```

### Using GradNorm in Training

```python
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.head_losses import (
    ContrastLoss, DepthLoss, MotionLoss, FatigueLoss
)

# Create head losses
head_losses = {
    'detection': DetectionLoss(num_classes=80),
    'depth': DepthLoss(),
    'contrast': ContrastLoss(),
    'motion': MotionLoss(),
    'fatigue': FatigueLoss()
}

# Create GradNorm combiner
gradnorm_loss = GradNormMultiHeadLoss(
    head_losses=head_losses,
    alpha=1.5,  # Restoring force
    update_interval=100  # Update weights every 100 iterations
)

# Use in training loop
for batch in train_loader:
    outputs = model(images)
    total_loss, metrics = gradnorm_loss(outputs, targets, model=model)
    
    # Log metrics
    if 'task_weights' in metrics:
        print(f"Task weights: {metrics['task_weights']}")
```

### Using Runtime Head Kill Switches

```python
from ml.utils.error_handling import HeadKillSwitchManager, wrap_heads_with_killswitch

# Create kill switch manager
kill_switch = HeadKillSwitchManager()

# Wrap model heads
model = wrap_heads_with_killswitch(model, kill_switch)

# Disable heads at runtime
kill_switch.disable_head('motion')
kill_switch.disable_head('fatigue')

# Or disable by category
kill_switch.disable_heads_by_category('optional')  # Disable all optional heads

# Run inference (disabled heads return safe defaults)
outputs = model(images)
```

### Using Ethical Safeguards

```python
from ml.utils.error_handling import EthicalGuard

# Create ethical guard
guard = EthicalGuard(
    uncertainty_threshold=0.7,
    suppression_mode='soft',  # 'soft', 'hard', 'graded'
    enable_safety_checks=True
)

# Apply safeguards to outputs
guarded = guard.guard_outputs(outputs, uncertainty=uncertainty)

# Check if outputs are safe
if guarded['safety_info']['safe']:
    # Use outputs
    pass
else:
    # Handle unsafe outputs
    print(f"Unsafe outputs: {guarded['safety_info']['reasons']}")
```

---

## Stress Test Details

### 1. Head Isolation Stress Tests

**Purpose:** Detect gradient interference and silent head collapse.

**What it tests:**
- Trains 5 variants from the same checkpoint:
  - A: Detection only
  - B: Detection + Depth
  - C: Detection + Accessibility
  - D: Detection + Navigation
  - E: All heads

**Red flags:**
- Detection mAP drops only in variant E
- Depth MAE oscillates wildly in E but not B
- Any head's gradient norm → near zero

**Usage:**
```python
from ml.training.stress_tests import HeadIsolationStressTest, StressTestConfig

config = StressTestConfig()
test = HeadIsolationStressTest(config)
results = test.run(model, train_loader, val_loader, device='cuda', epochs_per_variant=5)
```

### 2. Loss Scaling Stress Tests

**Purpose:** Ensure no loss term dominates training.

**What it tests:**
- Artificially scales each loss independently (λ ∈ {0.1, 0.5, 1, 2, 5})
- Observes if training diverges or other losses collapse

**Red flags:**
- Single loss scaling breaks everything
- Uncertainty head saturates (always high or low)

**Usage:**
```python
from ml.training.stress_tests import LossScalingStressTest

test = LossScalingStressTest(config)
results = test.run(model, loss_fn, train_loader, device='cuda', head_name='depth')
```

### 3. Input Corruption Stress Tests

**Purpose:** Simulate bad cameras, motion blur, low light, occlusion.

**What it tests:**
- Gaussian blur
- Motion blur
- Random occlusions (CutOut)
- Contrast reduction
- JPEG compression

**Red flags:**
- Uncertainty doesn't rise when confidence drops
- High urgency false positive rate
- System doesn't degrade gracefully

**Usage:**
```python
from ml.training.stress_tests import InputCorruptionStressTest

test = InputCorruptionStressTest(config)
results = test.run(model, val_loader, device='cuda')
```

### 4. Temporal Stress Tests

**Purpose:** Test stability across time.

**What it tests:**
- Static scene for 100 frames
- Slowly changing scene
- Sudden change (door opens, object appears)

**Red flags:**
- Frame-to-frame urgency flipping
- Distance jumping >30% frame-to-frame
- Repeated identical TTS messages

**Usage:**
```python
from ml.training.stress_tests import TemporalStressTest

test = TemporalStressTest(config)
result = test.run(model, device='cuda')
```

### 5. Head Dropout Stress Tests

**Purpose:** Ensure graceful degradation.

**What it tests:**
- Disable Depth head
- Disable OCR
- Disable Motion
- Inject NaNs into one head output

**Red flags:**
- Missing head crashes the pipeline
- System doesn't adapt
- Uncertainty doesn't increase

**Usage:**
```python
from ml.training.stress_tests import HeadDropoutStressTest

test = HeadDropoutStressTest(config)
results = test.run(model, val_loader, device='cuda', kill_switch_manager=kill_switch)
```

---

## Stress Test Dashboard

The stress test suite generates a dashboard with:

- **Summary:** Total tests, passed, failed, warnings
- **Test Details:** Per-test results with metrics and red flags
- **Recommendations:** Actions to take based on failures

**Example dashboard:**
```json
{
  "summary": {
    "total_tests": 15,
    "passed": 12,
    "failed": 3,
    "warnings": 5
  },
  "tests": [
    {
      "category": "head_isolation",
      "test": "Head Isolation: variant_E",
      "status": "❌",
      "red_flags": [
        "Detection mAP dropped significantly in variant E",
        "Depth gradient norm → near zero"
      ],
      "metrics": {
        "detection_map": 0.25,
        "depth_mae": 0.45
      }
    }
  ]
}
```

---

## Integration with Training

### Adding GradNorm to Training Loop

```python
from ml.training.train_loop import ProductionTrainLoop
from ml.training.task_balancing import GradNormMultiHeadLoss

# Create GradNorm loss combiner
head_losses = {
    'detection': DetectionLoss(num_classes=80),
    'depth': DepthLoss(),
    # ... other heads
}
gradnorm_loss = GradNormMultiHeadLoss(head_losses=head_losses)

# Create training loop
trainer = ProductionTrainLoop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=gradnorm_loss,  # Use GradNorm loss
    device='cuda'
)

# Train
trainer.train()
```

### Monitoring Per-Head Losses

```python
from ml.training.task_balancing import PerHeadLossMonitor

# Create monitor
monitor = PerHeadLossMonitor(window_size=100)

# In training loop
for batch in train_loader:
    outputs = model(images)
    loss_dict = loss_fn(outputs, targets)
    
    # Update monitor
    head_losses = {
        'detection': loss_dict.get('detection_loss', 0.0),
        'depth': loss_dict.get('depth_loss', 0.0),
        # ... other heads
    }
    monitor.update(head_losses)
    
    # Check for issues
    if iteration % 100 == 0:
        issues = monitor.detect_issues()
        if issues['dominant']:
            print(f"⚠️ Dominant heads: {issues['dominant']}")
        if issues['oscillating']:
            print(f"⚠️ Oscillating heads: {issues['oscillating']}")
```

---

## Best Practices

### 1. Run Stress Tests Regularly

- Before each release
- After major architecture changes
- When adding new heads
- When changing loss functions

### 2. Monitor During Training

- Track per-head losses
- Monitor gradient norms
- Watch for red flags
- Log GradNorm metrics

### 3. Use Kill Switches for Debugging

- Isolate problematic heads
- Test graceful degradation
- Validate fallback behavior

### 4. Apply Ethical Safeguards

- Always enable uncertainty suppression
- Use safety checks in production
- Monitor safety info in logs

---

## Troubleshooting

### Stress Tests Fail

**Problem:** Tests fail with errors

**Solutions:**
- Check model checkpoint path
- Verify data loaders are valid
- Ensure device is available
- Check model architecture matches expected heads

### GradNorm Not Working

**Problem:** Task weights not updating

**Solutions:**
- Verify shared parameters are set
- Check update_interval is reasonable
- Ensure model is passed to forward()
- Check gradient norms are computed correctly

### Kill Switches Not Working

**Problem:** Heads still execute when disabled

**Solutions:**
- Verify heads are wrapped with KillSwitchWrapper
- Check kill switch manager is passed correctly
- Ensure head names match exactly

### Ethical Safeguards Too Aggressive

**Problem:** System suppresses too many outputs

**Solutions:**
- Adjust uncertainty_threshold
- Use 'soft' suppression mode instead of 'hard'
- Review safety_checker thresholds
- Validate uncertainty calibration

---

## References

- **GradNorm Paper:** Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
- **Stress Testing Playbook:** See `docs/STRESS_TESTING_PLAYBOOK.md` (if exists)
- **System Limitations:** See `docs/SYSTEM_LIMITATIONS.md`

---

**Last Updated:** 2024
**Status:** Active

