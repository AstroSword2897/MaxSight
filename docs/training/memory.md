# Memory Management Guide

## Overview

Memory profiling and management utilities for MaxSight models.

## Memory Profiling

### Usage

```python
from ml.tools.memory_profile import report_memory, print_memory_summary

# Report current memory usage
stats = report_memory(device='cuda')
print_memory_summary(device='cuda')

# Reset peak statistics
reset_peak_stats(device='cuda')
```

### Memory Usage Patterns

**T0 (Baseline)**: ~200-300 MB inference, ~500-700 MB training
**T2 (Hybrid)**: ~1.5-2.0 GB inference, ~3.0-4.0 GB training
**T5 (Temporal)**: ~3.0-4.0 GB inference, ~6.0-8.0 GB training

## Memory Optimization Strategies

1. **Gradient Checkpointing**: Reduces memory at cost of compute
2. **Mixed Precision**: FP16 reduces memory by ~50%
3. **Batch Size Reduction**: Linear memory reduction
4. **Head Disabling**: Disable non-critical heads for mobile

## Memory Leak Detection

Use memory profiling during training to detect leaks:

```python
from ml.tools.memory_profile import report_memory

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        ...
        
        # Check memory every 100 batches
        if batch_idx % 100 == 0:
            stats = report_memory()
            print(f"Memory: {stats['allocated']:.2f} MB")
```

## Troubleshooting

- **OOM Errors**: Reduce batch size, enable gradient checkpointing
- **Memory Growth**: Check for tensor accumulation, use `.detach()` where appropriate
- **MPS Issues**: Use MPS-stable mode, ensure tensor contiguity

