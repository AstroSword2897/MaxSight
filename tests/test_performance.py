"""Performance Benchmark Tests for MaxSight Model
Tests latency, throughput, and memory usage for production deployment."""

import torch
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.benchmark import benchmark_inference


def test_inference_latency():
    """Test inference latency meets <500ms target for mobile deployment."""
    print("Performance Test 1: Inference Latency")
    
    model = create_model()
    device = torch.device('cpu')  # Test on CPU (mobile deployment)
    model = model.to(device)
    model.eval()
    
    # Test with batch size 1 (typical mobile use case)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup.
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_image)
    
    # Measure latency.
    latencies = []
    num_runs = 50
    
    with torch.no_grad():
        for _ in range(num_runs):
            # CRITICAL: Synchronize GPU before timing for accurate latency measurements.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_image)
            # CRITICAL: Synchronize GPU after inference to ensure completion.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms.
    
    mean_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"  Mean latency: {mean_latency:.2f} ms")
    print(f"  P95 latency: {p95_latency:.2f} ms")
    print(f"  P99 latency: {p99_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Target: <500ms")
    
    # Assert latency meets target.
    assert mean_latency < 500, f"Mean latency {mean_latency:.2f}ms exceeds 500ms target"
    assert p95_latency < 600, f"P95 latency {p95_latency:.2f}ms exceeds 600ms target"
    
    print("  ✅ PASSED: Latency within target")


def test_throughput():
    """Test throughput (FPS) for real-time processing."""
    print("\nPerformance Test 2: Throughput (FPS)")
    
    model = create_model()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup.
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_image)
    
    # Measure throughput.
    num_frames = 100
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_frames):
            _ = model(dummy_image)
    
    end = time.perf_counter()
    total_time = end - start
    fps = num_frames / total_time
    
    print(f"  Processed {num_frames} frames in {total_time:.2f}s")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  Target: >2 FPS (real-time)")
    
    assert fps > 2.0, f"Throughput {fps:.2f} FPS below 2 FPS target"
    
    print("  ✅ PASSED: Throughput meets target")


def test_memory_usage():
    """Test memory usage for mobile deployment constraints."""
    print("\nPerformance Test 3: Memory Usage")
    
    model = create_model()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Estimate model size.
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 * 1024)  # FP32: 4 bytes per param.
    
    # Estimate INT8 quantized size.
    int8_size_mb = (total_params * 1) / (1024 * 1024)  # INT8: 1 byte per param.
    
    print(f"  Model parameters: {total_params:,}")
    print(f"  FP32 size: {model_size_mb:.2f} MB")
    print(f"  INT8 size (estimated): {int8_size_mb:.2f} MB")
    print(f"  Target: <300 MB (quantized)")
    
    assert int8_size_mb < 300, f"INT8 model size {int8_size_mb:.2f}MB exceeds 300MB target"
    
    print("  ✅ PASSED: Memory usage within target")


def test_batch_processing():
    """Test performance with different batch sizes."""
    print("\nPerformance Test 4: Batch Processing Performance")
    
    model = create_model()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    results = {}
    
    for batch_size in batch_sizes:
        dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Warmup.
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_image)
        
        # Measure.
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_image)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        mean_latency = sum(latencies) / len(latencies)
        results[batch_size] = mean_latency
        
        print(f"  Batch size {batch_size}: {mean_latency:.2f} ms")
    
    # Batch size 1 is fastest for mobile.
    assert results[1] < 500, f"Batch size 1 latency {results[1]:.2f}ms exceeds target"
    
    print("  ✅ PASSED: Batch processing performance acceptable")


def test_benchmark_integration():
    """Test integration with ml.training.benchmark module."""
    print("\nPerformance Test 5: Benchmark Module Integration")
    
    model = create_model()
    device = torch.device('cpu')
    model = model.to(device)
    
    # Use the benchmark function from ml.training.benchmark.
    results = benchmark_inference(
        model,
        input_size=(1, 3, 224, 224),
        device=device,
        num_warmup=5,
        num_runs=30,
        batch_sizes=[1]
    )
    
    # Results are nested by batch size, check 'overall' or first batch.
    overall: Dict[str, float]
    if 'overall' in results:
        overall = results['overall']  # type: ignore[assignment]
    elif 'batch_1' in results:
        overall = results['batch_1']  # type: ignore[assignment]
    else:
        # Get first batch result (skip non-dict values like final_peak_memory_mb)
        for value in results.values():
            if isinstance(value, dict):
                overall = value  # type: ignore[assignment]
                break
        else:
            raise ValueError("No batch results found in benchmark output")
    
    assert 'mean_ms' in overall, "Benchmark results missing mean latency"
    assert 'median_ms' in overall, "Benchmark results missing median latency"
    
    print(f"  Mean latency: {overall['mean_ms']:.2f} ms")
    print(f"  Median latency: {overall['median_ms']:.2f} ms")
    print(f"  Min latency: {overall['min_ms']:.2f} ms")
    print(f"  Max latency: {overall['max_ms']:.2f} ms")
    
    assert overall['mean_ms'] < 500, "Benchmark mean latency exceeds target"
    
    print("  ✅ PASSED: Benchmark integration working")


if __name__ == "__main__":
    print("Running Performance Benchmark Tests")
    print("=" * 50)
    
    test_inference_latency()
    test_throughput()
    test_memory_usage()
    test_batch_processing()
    test_benchmark_integration()
    
    print("\n" + "=" * 50)
    print("All performance tests passed!")


