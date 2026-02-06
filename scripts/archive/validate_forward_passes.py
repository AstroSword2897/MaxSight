#!/usr/bin/env python3
"""Hard Validation Sprint: Forward-Pass Sanity Check for All Tiers (T0-T5)..."""

import argparse
import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, CapabilityTier, TierConfig
from ml.utils.preprocessing import ImagePreprocessor


def get_device(requested: str = None):
    """Get device. If requested is given, use it (with fallback). Else pick best available."""
    if requested and requested != "auto":
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cpu")
        if requested == "mps":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_test_input(batch_size: int = 2, device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Create test input batch with shapes matching real data...."""
    if device is None:
        device = get_device()
    
    # Image input: [B, 3, H, W] - matches ImagePreprocessor output
    # Note: Real data would be [0, 255] uint8, but model expects normalized float
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    # Normalize to [0, 1] range (simulating preprocessing)
    images = torch.clamp((images + 1) / 2, 0, 1)
    
    # Audio input: [B, samples] - raw waveform
    audio = torch.randn(batch_size, 16000, device=device)  # 1 second at 16kHz
    # Normalize audio to reasonable range
    audio = torch.clamp(audio, -1, 1)
    
    return {
        'images': images,
        'audio': audio,
    }


def test_tier_forward_pass(
    tier: CapabilityTier,
    device: torch.device,
    batch_size: int = 2,
    num_runs: int = 5
) -> Dict[str, any]:
    """Test forward pass for a single tier.
    
    Returns:
        Dictionary with results: success, latency_ms, memory_mb, shapes, errors"""
    print(f"\n{'='*60}")
    print(f"Testing Tier: {tier.name}")
    print(f"{'='*60}")
    
    results = {
        'tier': tier.name,
        'success': False,
        'latency_ms': None,
        'memory_mb': None,
        'shapes': {},
        'errors': [],
        'stage_a_latency_ms': None,
        'stage_b_latency_ms': None,
    }
    
    try:
        # Create model (exactly as training will)
        print(f"Creating model for {tier.name}...")
        model = create_model(
            num_classes=91,  # COCO classes
            use_audio=(tier.value >= 4),  # T4+ use audio
            tier_config=TierConfig.for_tier(tier)
        )
        # CRITICAL: Set to eval mode (no dropout, batchnorm uses running stats)
        model.eval()
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create test input
        inputs = create_test_input(batch_size=batch_size, device=device)
        
        # Warmup (critical for accurate timing)
        print("  Warming up...")
        with torch.no_grad():
            try:
                # Audio is passed as audio_features (not audio keyword)
                if model.use_audio and inputs.get('audio') is not None:
                    _ = model(inputs['images'], audio_features=inputs['audio'])
                else:
                    _ = model(inputs['images'])
                # GPU sync after warmup (CUDA or MPS)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
            except Exception as e:
                print(f"  ⚠️  Warmup failed: {e}")
                raise
        
        # Clear cache and measure memory before
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / 1024**2
        elif device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
            memory_before = torch.mps.current_allocated_memory() / 1024**2
        else:
            memory_before = 0
        
        # Time forward passes
        latencies = []
        stage_a_latencies = []
        stage_b_latencies = []
        
        print("  Running forward passes...")
        with torch.no_grad():
            for i in range(num_runs):
                # GPU sync before timing (CUDA or MPS)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                start_time = time.time()
                
                # Forward pass
                # Audio is passed as audio_features (not audio keyword)
                if model.use_audio and inputs.get('audio') is not None:
                    outputs = model(inputs['images'], audio_features=inputs['audio'])
                else:
                    outputs = model(inputs['images'])
                
                # GPU sync after forward pass (critical for accurate timing)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                # Measure Stage A latency (if available)
                if hasattr(model, '_last_stage_a_time'):
                    stage_a_time = model._last_stage_a_time
                    stage_a_latencies.append(stage_a_time * 1000)
                
                # Measure Stage B latency (if available)
                if hasattr(model, '_last_stage_b_time'):
                    stage_b_time = model._last_stage_b_time
                    stage_b_latencies.append(stage_b_time * 1000)
                
                elapsed = time.time() - start_time
                latencies.append(elapsed * 1000)  # Convert to ms
                
                # Capture output shapes on first run
                if i == 0:
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            results['shapes'][key] = list(value.shape)
                        elif isinstance(value, dict):
                            results['shapes'][key] = {
                                k: list(v.shape) if isinstance(v, torch.Tensor) else str(v)
                                for k, v in value.items()
                            }
        
        # Measure memory after
        if device.type == 'cuda':
            memory_after = torch.cuda.max_memory_allocated() / 1024**2
        elif device.type == 'mps':
            memory_after = torch.mps.driver_allocated_memory() / 1024**2
        else:
            memory_after = 0
        
        # Calculate statistics
        results['latency_ms'] = {
            'mean': sum(latencies) / len(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies))**0.5
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
        
        results['memory_mb'] = memory_after - memory_before
        results['parameters'] = {
            'total': total_params,
            'trainable': trainable_params,
        }
        results['success'] = True
        
        # Print results
        print(f"\n  ✅ SUCCESS")
        print(f"  Latency: {results['latency_ms']['mean']:.2f}ms (min: {results['latency_ms']['min']:.2f}ms, max: {results['latency_ms']['max']:.2f}ms)")
        if results['stage_a_latency_ms']:
            print(f"  Stage A: {results['stage_a_latency_ms']['mean']:.2f}ms")
        if results['stage_b_latency_ms']:
            print(f"  Stage B: {results['stage_b_latency_ms']['mean']:.2f}ms")
        print(f"  Memory: {results['memory_mb']:.2f} MB")
        print(f"  Parameters: {total_params:,}")
        
        # Check Stage A latency target (<150ms)
        if results['stage_a_latency_ms']:
            stage_a_mean = results['stage_a_latency_ms']['mean']
            if stage_a_mean > 150:
                print(f"  ⚠️  WARNING: Stage A latency ({stage_a_mean:.2f}ms) exceeds 150ms target!")
                results['errors'].append(f"Stage A latency {stage_a_mean:.2f}ms > 150ms target")
            else:
                print(f"  ✅ Stage A latency ({stage_a_mean:.2f}ms) meets <150ms target")
        
    except Exception as e:
        error_msg = f"Forward pass failed: {str(e)}"
        print(f"\n  ❌ FAILED: {error_msg}")
        print(f"  Traceback:")
        traceback.print_exc()
        results['errors'].append(error_msg)
        results['success'] = False
    
    return results


def main():
    """Run forward-pass validation for all tiers."""
    parser = argparse.ArgumentParser(description="Forward-pass validation for all tiers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device (default: auto)")
    args = parser.parse_args()
    
    print("="*60)
    print("HARD VALIDATION SPRINT: Forward-Pass Sanity Check")
    print("="*60)
    
    device = get_device(args.device)
    print(f"\nDevice: {device}")
    
    # Test all tiers (using correct enum names)
    tiers = [
        CapabilityTier.T0_BASELINE_CNN,
        CapabilityTier.T1_ATTENTION,
        CapabilityTier.T2_HYBRID_VIT,
        CapabilityTier.T3_CROSS_TASK,
        CapabilityTier.T4_CROSS_MODAL,
        CapabilityTier.T5_TEMPORAL,
    ]
    
    all_results = []
    failed_tiers = []
    
    for tier in tiers:
        try:
            results = test_tier_forward_pass(tier, device, batch_size=2, num_runs=5)
            all_results.append(results)
            
            if not results['success']:
                failed_tiers.append(tier.name)
            
            # Clear cache between tiers to avoid memory accumulation
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR testing {tier.name}: {e}")
            traceback.print_exc()
            failed_tiers.append(tier.name)
            
            # Clear cache even on error
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nTiers Tested: {len(tiers)}")
    print(f"Tiers Passed: {len(tiers) - len(failed_tiers)}")
    print(f"Tiers Failed: {len(failed_tiers)}")
    
    if failed_tiers:
        print(f"\n❌ FAILED TIERS: {', '.join(failed_tiers)}")
        print("\n⚠️  STOP: Fix failures before proceeding to training!")
        return 1
    
    print("\n✅ ALL TIERS PASSED!")
    
    # Print latency summary
    print("\n" + "-"*60)
    print("LATENCY SUMMARY")
    print("-"*60)
    for result in all_results:
        if result['success']:
            tier = result['tier']
            latency = result['latency_ms']['mean']
            stage_a_data = result.get('stage_a_latency_ms')
            stage_a = stage_a_data.get('mean', 'N/A') if stage_a_data else 'N/A'
            print(f"{tier:20s} | Total: {latency:7.2f}ms | Stage A: {stage_a if isinstance(stage_a, float) else str(stage_a):>7}")
    
    # Print memory summary
    print("\n" + "-"*60)
    print("MEMORY SUMMARY")
    print("-"*60)
    for result in all_results:
        if result['success']:
            tier = result['tier']
            memory = result['memory_mb']
            params = result['parameters']['total']
            print(f"{tier:20s} | Memory: {memory:7.2f} MB | Params: {params:>12,}")
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE - PROCEED TO SMOKE TRAINING")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

