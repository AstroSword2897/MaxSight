"""Inference latency benchmarking for MaxSight model."""

import time
import torch
from typing import Dict, Optional
import statistics


def benchmark_inference(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    batch_sizes: Optional[list] = None
) -> Dict[str, float]:
    """
    Benchmark model inference latency.
    
    Args:
        model: Model to benchmark
        input_size: Input tensor shape (batch, channels, height, width)
        device: Device to run on (default: model's device)
        num_warmup: Number of warmup runs
        num_runs: Number of timing runs
        batch_sizes: List of batch sizes to test (default: [1, 4, 8])
    
    Returns:
        Dictionary with latency stats (mean, median, min, max, std) in milliseconds
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    
    results = {}
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *input_size[1:], device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                times.append(elapsed)
        
        # Compute statistics
        results[f'batch_{batch_size}'] = {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
        }
    
    # Overall stats (batch_size=1)
    if 'batch_1' in results:
        results['overall'] = results['batch_1'].copy()
    
    return results


def print_benchmark_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print benchmark results in readable format."""
    print("\n" + "=" * 70)
    print("Inference Latency Benchmark Results")
    print("=" * 70)
    
    for key, stats in results.items():
        if key == 'overall':
            continue
        
        print(f"\nBatch Size {key.replace('batch_', '')}:")
        print(f"  Mean:   {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  Min:    {stats['min_ms']:.2f} ms")
        print(f"  Max:    {stats['max_ms']:.2f} ms")
        print(f"  Std:    {stats['std_ms']:.2f} ms")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOverall (batch_size=1):")
        print(f"  Mean:   {overall['mean_ms']:.2f} ms")
        print(f"  Target: <500 ms")
        print(f"  Status: {'✓ PASS' if overall['mean_ms'] < 500 else '✗ FAIL'}")
    
    print("=" * 70 + "\n")

