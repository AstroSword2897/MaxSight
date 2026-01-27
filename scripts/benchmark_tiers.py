#!/usr/bin/env python3
"""
Benchmark Before Scaling (1 day)

Measure now:
- Stage A latency (target <150ms)
- Memory per tier
- Parameter counts
- Mobile export size

This tells you:
- Which tiers are actually viable
- Where pruning / distillation will matter
- Whether mobile is realistic before long training
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
import json
from typing import Dict, List
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, CapabilityTier
from ml.training.export import export_to_coreml, export_to_onnx, export_to_jit


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def benchmark_model(
    tier: CapabilityTier,
    device: torch.device,
    num_runs: int = 50,
    batch_size: int = 1
) -> Dict:
    """Benchmark a single tier."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {tier.name}")
    print(f"{'='*60}")
    
    results = {
        'tier': tier.name,
        'success': False,
    }
    
    try:
        # Create model
        model = create_model(
            tier=tier,
            num_classes=91,
            use_audio=(tier.value >= 4),
            device=device
        )
        model.eval()
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['parameters'] = {
            'total': total_params,
            'trainable': trainable_params,
            'total_mb': total_params * 4 / 1024**2,  # Assume float32
        }
        
        # Create input
        images = torch.randn(batch_size, 3, 224, 224, device=device)
        audio = torch.randn(batch_size, 16000, device=device) if tier.value >= 4 else None
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(images, audio=audio)
        
        # Measure memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**2
        elif device.type == 'mps':
            memory_before = torch.mps.current_allocated_memory() / 1024**2
        else:
            memory_before = 0
        
        # Benchmark latency
        latencies = []
        stage_a_latencies = []
        stage_b_latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                outputs = model(images, audio=audio)
                elapsed = (time.time() - start) * 1000  # ms
                latencies.append(elapsed)
                
                # Try to get stage timings if available
                if hasattr(model, '_last_stage_a_time'):
                    stage_a_latencies.append(model._last_stage_a_time * 1000)
                if hasattr(model, '_last_stage_b_time'):
                    stage_b_latencies.append(model._last_stage_b_time * 1000)
        
        # Measure peak memory
        if device.type == 'cuda':
            memory_peak = torch.cuda.max_memory_allocated() / 1024**2
        elif device.type == 'mps':
            memory_peak = torch.mps.driver_allocated_memory() / 1024**2
        else:
            memory_peak = 0
        
        results['latency_ms'] = {
            'mean': sum(latencies) / len(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'p50': sorted(latencies)[len(latencies)//2],
            'p95': sorted(latencies)[int(len(latencies)*0.95)],
            'p99': sorted(latencies)[int(len(latencies)*0.99)],
        }
        
        if stage_a_latencies:
            results['stage_a_latency_ms'] = {
                'mean': sum(stage_a_latencies) / len(stage_a_latencies),
                'min': min(stage_a_latencies),
                'max': max(stage_a_latencies),
            }
        
        if stage_b_latencies:
            results['stage_b_latency_ms'] = {
                'mean': sum(stage_b_latencies) / len(stage_b_latencies),
                'min': min(stage_b_latencies),
                'max': max(stage_b_latencies),
            }
        
        results['memory_mb'] = {
            'peak': memory_peak,
            'inference': memory_peak - memory_before,
        }
        
        results['throughput'] = {
            'samples_per_sec': 1000.0 / results['latency_ms']['mean'],
            'fps': 1000.0 / results['latency_ms']['mean'],
        }
        
        # Check Stage A latency target
        if results.get('stage_a_latency_ms'):
            stage_a_mean = results['stage_a_latency_ms']['mean']
            results['stage_a_meets_target'] = stage_a_mean < 150
        else:
            results['stage_a_meets_target'] = None
        
        results['success'] = True
        
        # Print results
        print(f"  ✅ Parameters: {total_params:,} ({results['parameters']['total_mb']:.2f} MB)")
        print(f"  ✅ Latency: {results['latency_ms']['mean']:.2f}ms (p50: {results['latency_ms']['p50']:.2f}ms, p95: {results['latency_ms']['p95']:.2f}ms)")
        if results.get('stage_a_latency_ms'):
            print(f"  ✅ Stage A: {results['stage_a_latency_ms']['mean']:.2f}ms {'✅' if results['stage_a_meets_target'] else '❌'}")
        if results.get('stage_b_latency_ms'):
            print(f"  ✅ Stage B: {results['stage_b_latency_ms']['mean']:.2f}ms")
        print(f"  ✅ Memory: {results['memory_mb']['peak']:.2f} MB peak")
        print(f"  ✅ Throughput: {results['throughput']['fps']:.2f} FPS")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results['error'] = str(e)
        results['success'] = False
    
    return results


def benchmark_export(tier: CapabilityTier, device: torch.device) -> Dict:
    """Benchmark model export sizes."""
    print(f"\n{'='*60}")
    print(f"Export Benchmark: {tier.name}")
    print(f"{'='*60}")
    
    results = {
        'tier': tier.name,
        'exports': {},
    }
    
    try:
        # Create model
        model = create_model(
            tier=tier,
            num_classes=91,
            use_audio=(tier.value >= 4),
            device=device
        )
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        if tier.value >= 4:
            dummy_audio = torch.randn(1, 16000)
        else:
            dummy_audio = None
        
        # Test JIT export
        try:
            print("  Testing JIT export...")
            jit_path = Path(f"/tmp/maxsight_{tier.name.lower()}_jit.pt")
            export_to_jit(model, str(jit_path), device=str(device))
            jit_size = jit_path.stat().st_size / 1024**2  # MB
            results['exports']['jit'] = {'size_mb': jit_size, 'success': True}
            print(f"    ✅ JIT: {jit_size:.2f} MB")
            jit_path.unlink()  # Cleanup
        except Exception as e:
            print(f"    ❌ JIT export failed: {e}")
            results['exports']['jit'] = {'success': False, 'error': str(e)}
        
        # Test ONNX export
        try:
            print("  Testing ONNX export...")
            onnx_path = Path(f"/tmp/maxsight_{tier.name.lower()}_onnx.onnx")
            export_to_onnx(model, str(onnx_path), dummy_input, device=str(device))
            onnx_size = onnx_path.stat().st_size / 1024**2  # MB
            results['exports']['onnx'] = {'size_mb': onnx_size, 'success': True}
            print(f"    ✅ ONNX: {onnx_size:.2f} MB")
            onnx_path.unlink()  # Cleanup
        except Exception as e:
            print(f"    ❌ ONNX export failed: {e}")
            results['exports']['onnx'] = {'success': False, 'error': str(e)}
        
        # Test CoreML export (if on macOS)
        try:
            print("  Testing CoreML export...")
            coreml_path = Path(f"/tmp/maxsight_{tier.name.lower()}_coreml.mlpackage")
            export_to_coreml(model, str(coreml_path), dummy_input, device=str(device))
            if coreml_path.is_dir():
                # .mlpackage is a directory
                coreml_size = sum(f.stat().st_size for f in coreml_path.rglob('*') if f.is_file()) / 1024**2
            else:
                coreml_size = coreml_path.stat().st_size / 1024**2
            results['exports']['coreml'] = {'size_mb': coreml_size, 'success': True}
            print(f"    ✅ CoreML: {coreml_size:.2f} MB")
            # Cleanup
            import shutil
            if coreml_path.exists():
                if coreml_path.is_dir():
                    shutil.rmtree(coreml_path)
                else:
                    coreml_path.unlink()
        except Exception as e:
            print(f"    ❌ CoreML export failed: {e}")
            results['exports']['coreml'] = {'success': False, 'error': str(e)}
        
    except Exception as e:
        print(f"  ❌ Export benchmark failed: {e}")
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark all tiers")
    parser.add_argument("--tier", type=str, default=None,
                       choices=["T0_MOBILE", "T1_EDGE", "T2_HYBRID_VIT", "T3_CROSS_MODAL", "T4_CROSS_MODAL", "T5_TEMPORAL"],
                       help="Specific tier to benchmark (default: all)")
    parser.add_argument("--runs", type=int, default=50, help="Number of runs per tier")
    parser.add_argument("--export", action="store_true", help="Also benchmark exports")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    device = get_device()
    print("="*60)
    print("BENCHMARK BEFORE SCALING")
    print("="*60)
    print(f"\nDevice: {device}")
    print(f"Runs per tier: {args.runs}")
    
    # Select tiers
    if args.tier:
        tiers = [CapabilityTier[args.tier]]
    else:
        tiers = [
            CapabilityTier.T0_MOBILE,
            CapabilityTier.T1_EDGE,
            CapabilityTier.T2_HYBRID_VIT,
            CapabilityTier.T3_CROSS_MODAL,
            CapabilityTier.T4_CROSS_MODAL,
            CapabilityTier.T5_TEMPORAL,
        ]
    
    all_results = []
    
    for tier in tiers:
        # Benchmark inference
        inference_results = benchmark_model(tier, device, num_runs=args.runs)
        all_results.append(inference_results)
        
        # Benchmark exports if requested
        if args.export:
            export_results = benchmark_export(tier, device)
            all_results.append(export_results)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\n{'Tier':<20} | {'Params (M)':>12} | {'Latency (ms)':>15} | {'Memory (MB)':>12} | {'FPS':>8}")
    print("-" * 80)
    
    for result in all_results:
        if result.get('success') and 'parameters' in result:
            tier = result['tier']
            params_m = result['parameters']['total'] / 1e6
            latency = result['latency_ms']['mean']
            memory = result['memory_mb']['peak']
            fps = result['throughput']['fps']
            
            # Check Stage A target
            stage_a_ok = result.get('stage_a_meets_target')
            marker = "✅" if stage_a_ok else "⚠️ " if stage_a_ok is False else "  "
            
            print(f"{tier:<20} | {params_m:>12.2f} | {latency:>15.2f} | {memory:>12.2f} | {fps:>8.2f} {marker}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    viable_tiers = []
    for result in all_results:
        if result.get('success') and 'parameters' in result:
            stage_a_ok = result.get('stage_a_meets_target')
            if stage_a_ok:
                viable_tiers.append(result['tier'])
    
    if viable_tiers:
        print(f"\n✅ Viable tiers (Stage A <150ms): {', '.join(viable_tiers)}")
    else:
        print("\n⚠️  No tiers meet Stage A <150ms target - consider optimization")
    
    return 0


if __name__ == "__main__":
    exit(main())

