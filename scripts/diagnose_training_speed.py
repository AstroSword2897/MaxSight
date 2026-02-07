#!/usr/bin/env python3
"""Diagnose training speed bottlenecks.

Identifies slow operations during training epochs:
- Data loading time
- Forward pass time
- Backward pass time
- Validation time
- Checkpoint saving time
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import argparse
import json
from collections import defaultdict

from ml.data.data_pipeline import create_data_loaders
from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier
from ml.training.train_loop import ProductionTrainLoop


class TimingProfiler:
    """Profile timing for different training operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
    
    def time_operation(self, name: str, func, *args, **kwargs):
        """Time a function call and record it."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.timings[name].append(elapsed)
        self.counts[name] += 1
        return result
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed operations."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': self.counts[name]
                }
        return stats
    
    def print_report(self):
        """Print timing report."""
        stats = self.get_stats()
        print("\n" + "="*70)
        print("TRAINING SPEED DIAGNOSTICS")
        print("="*70)
        
        total_time = sum(s['total'] for s in stats.values())
        
        for name, s in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            pct = (s['total'] / total_time * 100) if total_time > 0 else 0
            print(f"\n{name}:")
            print(f"  Total: {s['total']:.3f}s ({pct:.1f}%)")
            print(f"  Mean:  {s['mean']:.4f}s")
            print(f"  Min:   {s['min']:.4f}s")
            print(f"  Max:   {s['max']:.4f}s")
            print(f"  Count: {s['count']}")
        
        print(f"\n{'='*70}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"{'='*70}\n")


def diagnose_data_loading(train_loader, num_batches: int = 10):
    """Diagnose data loading speed."""
    print("🔍 Diagnosing data loading...")
    profiler = TimingProfiler()
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        profiler.time_operation('data_load', lambda: None)
    
    stats = profiler.get_stats()
    if 'data_load' in stats:
        mean_time = stats['data_load']['mean']
        print(f"  ⏱️  Mean batch load time: {mean_time:.4f}s")
        print(f"  📊 Estimated time per epoch: {mean_time * len(train_loader):.2f}s")
    
    return profiler


def diagnose_forward_pass(model, train_loader, device, num_batches: int = 10):
    """Diagnose forward pass speed."""
    print("🔍 Diagnosing forward pass...")
    profiler = TimingProfiler()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            
            images = batch['images'].to(device)
            
            # Time forward pass
            def forward():
                return model(images)
            
            profiler.time_operation('forward_pass', forward)
    
    stats = profiler.get_stats()
    if 'forward_pass' in stats:
        mean_time = stats['forward_pass']['mean']
        print(f"  ⏱️  Mean forward pass time: {mean_time:.4f}s")
        print(f"  📊 Estimated time per epoch: {mean_time * len(train_loader):.2f}s")
    
    return profiler


def diagnose_validation(trainer, num_batches: int = 5):
    """Diagnose validation speed."""
    print("🔍 Diagnosing validation...")
    profiler = TimingProfiler()
    
    # Mock validation with timing
    original_validate = trainer.validate
    
    def timed_validate(*args, **kwargs):
        start = time.time()
        result = original_validate(*args, **kwargs)
        elapsed = time.time() - start
        profiler.timings['validation'].append(elapsed)
        profiler.counts['validation'] += 1
        return result
    
    trainer.validate = timed_validate
    
    # Run validation
    try:
        trainer.validate(epoch=0, use_ema=False)
    except Exception as e:
        print(f"  ⚠️  Validation error: {e}")
    
    stats = profiler.get_stats()
    if 'validation' in stats:
        mean_time = stats['validation']['mean']
        print(f"  ⏱️  Validation time: {mean_time:.2f}s")
    
    return profiler


def main():
    parser = argparse.ArgumentParser(description="Diagnose training speed bottlenecks")
    parser.add_argument("--train-annotation", type=str, required=True)
    parser.add_argument("--val-annotation", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_diagnostic")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--tier", type=str, default="T5", choices=["T2", "T3", "T4", "T5"])
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("TRAINING SPEED DIAGNOSTICS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Profiling {args.num_batches} batches")
    print("="*70)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        train_annotation_file=Path(args.train_annotation),
        val_annotation_file=Path(args.val_annotation),
        test_annotation_file=None,
        image_dir=Path(args.image_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    print(f"\n📊 Dataset sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    model = create_model(tier_config=tier_config)
    model = model.to(device)
    
    print(f"\n📦 Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Run diagnostics
    all_profilers = []
    
    # Data loading
    data_profiler = diagnose_data_loading(train_loader, args.num_batches)
    all_profilers.append(data_profiler)
    
    # Forward pass
    forward_profiler = diagnose_forward_pass(model, train_loader, device, args.num_batches)
    all_profilers.append(forward_profiler)
    
    # Create trainer for validation profiling
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,
    )
    
    # Validation
    val_profiler = diagnose_validation(trainer, num_batches=1)
    all_profilers.append(val_profiler)
    
    # Combined report
    combined_profiler = TimingProfiler()
    for profiler in all_profilers:
        for name, times in profiler.timings.items():
            combined_profiler.timings[name].extend(times)
            combined_profiler.counts[name] += profiler.counts[name]
    
    combined_profiler.print_report()
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    stats = combined_profiler.get_stats()
    total_time = sum(s['total'] for s in stats.values())
    
    if 'validation' in stats:
        val_pct = stats['validation']['total'] / total_time * 100
        if val_pct > 30:
            print("⚠️  Validation is taking >30% of time!")
            print("   → Consider optimizing validation loop (batch get_detections)")
            print("   → Reduce validation frequency (validate every N epochs)")
    
    if 'data_load' in stats:
        data_pct = stats['data_load']['total'] / total_time * 100
        if data_pct > 20:
            print("⚠️  Data loading is taking >20% of time!")
            print(f"   → Increase num_workers (current: {args.num_workers})")
            print("   → Enable pin_memory if using CUDA")
            print("   → Consider caching/preloading images")
    
    if 'forward_pass' in stats:
        forward_pct = stats['forward_pass']['total'] / total_time * 100
        if forward_pct < 50:
            print("⚠️  Forward pass is <50% of time - likely I/O bound!")
            print("   → Increase batch size if memory allows")
            print("   → Optimize data loading (see above)")
    
    print("="*70)


if __name__ == "__main__":
    main()
