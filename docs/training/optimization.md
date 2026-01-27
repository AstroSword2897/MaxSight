# Model Optimization Guide

## Overview

MaxSight provides several optimization utilities for model pruning, quantization, and mobile deployment.

## Pruning

Model pruning reduces model size by removing less important weights.

### Usage

```python
from ml.optimization.mobile_optimizations import MobileOptimizer

optimizer = MobileOptimizer(model)
pruned_model = optimizer.prune_model(amount=0.3)  # Remove 30% of weights
```

### Available Methods

- `prune_model(amount)`: Magnitude-based pruning
- `prune_heads(heads_to_disable)`: Disable specific heads for mobile

## Quantization

Quantization reduces model precision from FP32 to INT8, reducing model size by ~4x.

### Usage

```python
from ml.training.quantization import quantize_model_int8

quantized_model = quantize_model_int8(
    model,
    calibration_data=dataloader,
    num_calibration_batches=10
)
```

### Trade-offs

- **Size**: ~4x reduction
- **Speed**: 2-4x faster inference
- **Accuracy**: Typically <3% mAP drop

## Mobile Optimization

The `MobileOptimizer` class provides utilities for mobile deployment:

- Model pruning
- Head disabling
- Memory estimation
- Edge-cloud hybrid routing

See `ml/optimization/mobile_optimizations.py` for details.

## Best Practices

1. **Prune before quantize**: Pruning reduces model size, quantization reduces precision
2. **Calibrate quantization**: Use representative calibration data
3. **Test accuracy**: Always verify accuracy after optimization
4. **Profile memory**: Use `ml/tools/memory_profile.py` to measure memory usage

