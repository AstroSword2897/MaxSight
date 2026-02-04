# Device Selection Policy for MaxSight Training

## Automatic Device Selection Rules

The smoke training script (`scripts/smoke_train.py`) automatically selects the appropriate device based on model size:

### **Rule: Parameter-Based Device Selection**

- **Models < 10,000 parameters**: Automatically use **CPU**
  - Suitable for: Small experiments, smoke tests, tiny models
  - No GPU required
  
- **Models >= 10,000 parameters**: Require **Cloud GPU (CUDA)**
  - Suitable for: Production training, large models
  - Must use cloud GPU services (Colab, AWS, Paperspace, Lambda Labs)

## Current MaxSight Model Sizes

| Tier | Parameters | Device Required |
|------|------------|-----------------|
| T0_BASELINE_CNN | ~29M | Cloud GPU |
| T1_EDGE | ~50M | Cloud GPU |
| T2_HYBRID_VIT | ~210M | Cloud GPU |
| T3_CROSS_MODAL | ~250M | Cloud GPU |
| T4_CROSS_MODAL | ~280M | Cloud GPU |
| T5_TEMPORAL | ~320M | Cloud GPU |

**All MaxSight tiers require cloud GPU for training.**

## Usage Examples

### Automatic Selection (Recommended)

```bash
# Script automatically detects model size and requires cloud GPU
python scripts/smoke_train.py --tier T2_HYBRID_VIT
```

**Output:**
```
❌ ERROR: Model has 210,857,837 parameters (>= 10,000). 
Cloud GPU (CUDA) required for training.
```

### Force CPU (Override - Not Recommended)

```bash
# Override to use CPU (very slow for large models)
python scripts/smoke_train.py --tier T2_HYBRID_VIT --force-cpu
```

**Warning:** Training will be extremely slow (hours/days per epoch).

### Custom Threshold

```bash
# Change threshold (e.g., for testing)
python scripts/smoke_train.py --tier T2_HYBRID_VIT --param-threshold 1000000
```

## Cloud GPU Options

When CUDA is required, use one of these services:

### 1. Google Colab (Free/Paid)
- Free tier: T4 GPU (limited hours)
- Paid: A100, V100
- Setup: Upload code, run in notebook

### 2. AWS EC2
- Instance: `g4dn.xlarge` or larger
- Cost: ~$0.50-2.00/hour
- Setup: Launch instance, SSH, install dependencies

### 3. Paperspace Gradient
- Free tier: M4000 (limited)
- Paid: A100, V100
- Setup: Upload code, run job

### 4. Lambda Labs
- Cost: ~$0.50-1.00/hour
- Setup: SSH access, pre-configured PyTorch

## Why This Policy?

1. **CPU is too slow** for large models (210M+ parameters)
   - Training would take days/weeks per epoch
   - Not practical for development

2. **MPS has backward pass bugs** (Apple Silicon)
   - Forward pass works fine
   - Backward pass crashes on complex models
   - PyTorch limitation, not code bug

3. **Cloud GPU is practical** for large models
   - Fast training (hours, not days)
   - Cost-effective for development
   - Full gradient support

## Recommendations

### For Local Development
- Use CPU for **small models** (< 10k params)
- Use MPS for **forward pass testing** only
- Use CPU with `--force-cpu` for **smoke tests** (slow but works)

### For Production Training
- Use **cloud GPU (CUDA)** for all models >= 10k params
- Use **cloud GPU** for all MaxSight tiers
- Use **cloud GPU** for full training runs

## Summary

| Model Size | Device | Use Case |
|------------|--------|----------|
| < 10k params | CPU | Local experiments |
| >= 10k params | Cloud GPU (CUDA) | Production training |
| Any size (override) | CPU (--force-cpu) | Smoke tests only |

**MaxSight models are all >= 10k params → Cloud GPU required for training.**

