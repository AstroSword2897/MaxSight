# Hyperparameter Rationale & Trade-offs

## Core Principles Applied

### 1. Learning Rate Scaling by Model Size
- **T0 (29M)**: 1.5e-3 - Can tolerate higher LR
- **T1 (50M)**: 1.2e-4 - Moderate for attention
- **T2 (210M)**: 1.0e-4 - Hybrid architecture
- **T3 (250M)**: 9.0e-5 - Cross-task learning
- **T4 (280M)**: 8.0e-5 - Cross-modal fusion
- **T5 (320M)**: 7.5e-5 - **Sweet spot** for 300-400M params at batch 32

**Why 7.5e-5 for T5?**
- 1e-4 is slightly hot for:
  - Stacked attention layers
  - Temporal gradients (backprop through time)
  - Dynamic convolution updates
- 7.5e-5 balances:
  - Fast enough convergence
  - Stable gradient flow
  - Prevents attention collapse

### 2. Weight Decay: 0.05 (Not 0.0001)

**Problem with 0.0001:**
- Too low for 300M+ parameter models
- High overfitting risk
- Model is too expressive without regularization

**Why 0.05 works:**
- Strong enough to prevent overfitting
- Not so strong it kills learning
- Standard for large transformer-like models

### 3. Loss Weight Rebalancing

**Previous Problem:**
```yaml
box_regression: 5.0  # Tyrannical
scene_description: 0.1  # Muted
scene_graph: 0.1  # Muted
```

**Why this fails:**
- Box regression dominates early training
- Semantic tasks never get enough signal
- GradNorm can't fully rescue extreme imbalance

**Rebalanced Solution:**
```yaml
box_regression: 3.0  # Still important, not dominant
scene_description: 0.3  # Above activation threshold
scene_graph: 0.3  # Above activation threshold
```

**Activation Threshold (0.3):**
- Below 0.3: Task effectively doesn't learn
- Above 0.3: Task gets real gradient signal
- GradNorm can fine-tune from here

### 4. Data Loader: num_workers = 8

**Why increase from 4?**
- Model is **compute-bound** (GPU waits for data)
- Starving GPU murders throughput
- 8 workers keeps GPU fed during forward/backward

**Trade-off:**
- More memory usage
- Worth it for 2-3x throughput improvement

### 5. Warmup: 20 epochs (T5)

**Why longer warmup?**
- Gives GradNorm time to stabilize
- Temporal models need gradual ramp-up
- Prevents early collapse of attention mechanisms

### 6. min_lr: 1e-6

**Why add minimum LR?**
- Prevents late-stage collapse
- Temporal heads can overfit late in training
- Keeps model learning even at end

### 7. 2-Phase Loss Scaling (Optional)

**When to use:**
- If training is unstable
- If semantic tasks aren't learning
- If you want safer fallback

**How it works:**
- **Phase 1 (0-40 epochs)**: Core detection only
- **Phase 2 (40+)**: All tasks enabled
- GradNorm rebalances in phase 2

**Saves weeks** of dead training by:
- Establishing strong detection baseline first
- Then adding complexity gradually
- Avoiding early collapse

## Hardware Considerations

### A100 (80GB)
- Can handle batch_size=4 with accumulate=8
- num_workers=8 is safe
- Mixed precision recommended

### H100 (80GB)
- Same as A100, but faster
- Can potentially increase batch_size slightly
- Same hyperparameters apply

### V100 (32GB)
- May need to reduce batch_size to 2
- Increase accumulate_grad_batches to 16
- Same learning rates apply

## Validation Strategy

**First 25 epochs:**
- Ignore absolute validation loss
- Watch **loss smoothness**
- Temporal models lie early

**After 25 epochs:**
- Monitor validation loss
- Check task-specific metrics
- GradNorm should have stabilized

## Gradient Clipping

**Default: 1.0**
- Works for most cases
- Prevents gradient explosion

**If seeing loss spikes after warmup:**
- Lower to 0.8
- Indicates unstable gradients
- May need to reduce LR slightly

## Future Optimizations

1. **LR from parameter count formally**
   - Derive from model size + batch size
   - Use learning rate finder

2. **Loss-unlock schedule**
   - More granular than 2-phase
   - Unlock tasks based on detection quality

3. **Hardware-specific tuning**
   - A100 vs H100 differences
   - Memory bandwidth considerations
