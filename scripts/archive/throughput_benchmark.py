"""Throughput Benchmark for MaxSight 3.0

Measures throughput for different forward pass scenarios
to understand performance characteristics before training."""

import torch
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier


class ThroughputBenchmark:
    """Benchmarks throughput for different scenarios."""
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
    
    def benchmark_scenario(
        self,
        name: str,
        tier: CapabilityTier,
        batch_size: int = 1,
        temporal: bool = False,
        audio: bool = False,
        num_warmup: int = 10,
        num_runs: int = 50
    ) -> Dict:
        """Benchmark a specific scenario."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"  Tier: {tier.name}")
        print(f"  Batch size: {batch_size}")
        print(f"  Temporal: {temporal}")
        print(f"  Audio: {audio}")
        print(f"{'='*60}")
        
        try:
            # Create model.
            tier_config = TierConfig.for_tier(tier)
            model = create_model(
                num_classes=91,
                use_audio=audio,
                tier_config=tier_config
            )
            model.eval()
            model = model.to(self.device)
            
            # Create inputs.
            if temporal:
                images = torch.randn(batch_size, 8, 3, 224, 224, device=self.device)
            else:
                images = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            audio_features = None
            if audio:
                audio_features = torch.randn(batch_size, 128, device=self.device)
            
            # Warmup.
            print("  Warming up...")
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(images, audio_features=audio_features, use_temporal=temporal)
            
            # Synchronize.
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark.
            print("  Running benchmark...")
            times = []
            memory_peaks = []
            
            for i in range(num_runs):
                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                start = time.time()
                with torch.no_grad():
                    outputs = model(images, audio_features=audio_features, use_temporal=temporal)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    memory_peaks.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB.
                
                elapsed = (time.time() - start) * 1000  # Ms.
                times.append(elapsed)
                
                if (i + 1) % 10 == 0:
                    print(f"    Run {i+1}/{num_runs}: {elapsed:.2f}ms")
            
            # Calculate statistics.
            times_sorted = sorted(times)
            stats = {
                'mean_ms': sum(times) / len(times),
                'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
                'min_ms': min(times),
                'max_ms': max(times),
                'p50_ms': times_sorted[len(times_sorted)//2],
                'p95_ms': times_sorted[int(len(times_sorted)*0.95)],
                'p99_ms': times_sorted[int(len(times_sorted)*0.99)],
                'throughput_fps': 1000.0 / (sum(times) / len(times)),
                'mean_memory_mb': sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                'max_memory_mb': max(memory_peaks) if memory_peaks else 0,
            }
            
            # Analyze outputs.
            output_info = {
                'num_outputs': len(outputs),
                'output_keys': list(outputs.keys()),
                'has_stage_a': any('objectness' in k or 'classification' in k for k in outputs.keys()),
                'has_stage_b': any('motion' in k or 'scene' in k or 'ocr' in k for k in outputs.keys()),
            }
            
            print(f"\n  Results:")
            print(f"    Mean latency: {stats['mean_ms']:.2f}ms")
            print(f"    Throughput: {stats['throughput_fps']:.2f} FPS")
            print(f"    P95 latency: {stats['p95_ms']:.2f}ms")
            print(f"    Outputs: {output_info['num_outputs']} keys")
            print(f"    Stage A: {output_info['has_stage_a']}")
            print(f"    Stage B: {output_info['has_stage_b']}")
            
            return {
                'name': name,
                'tier': tier.name,
                'batch_size': batch_size,
                'temporal': temporal,
                'audio': audio,
                'stats': stats,
                'output_info': output_info,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"  FAIL Failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'name': name,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark across all scenarios."""
        results = []
        
        # Key scenarios to test.
        scenarios = [
            # T0 Baseline.
            {'name': 'T0_Single', 'tier': CapabilityTier.T0_BASELINE_CNN, 'batch': 1, 'temporal': False, 'audio': False},
            {'name': 'T0_Batch4', 'tier': CapabilityTier.T0_BASELINE_CNN, 'batch': 4, 'temporal': False, 'audio': False},
            
            # T1 Attention.
            {'name': 'T1_Single', 'tier': CapabilityTier.T1_ATTENTION, 'batch': 1, 'temporal': False, 'audio': False},
            {'name': 'T1_WithAudio', 'tier': CapabilityTier.T1_ATTENTION, 'batch': 1, 'temporal': False, 'audio': True},
            
            # T2 Hybrid ViT.
            {'name': 'T2_Hybrid', 'tier': CapabilityTier.T2_HYBRID_VIT, 'batch': 1, 'temporal': False, 'audio': True},
            {'name': 'T2_Temporal', 'tier': CapabilityTier.T2_HYBRID_VIT, 'batch': 1, 'temporal': True, 'audio': True},
            
            # T3 Cross-Task.
            {'name': 'T3_CrossTask', 'tier': CapabilityTier.T3_CROSS_TASK, 'batch': 1, 'temporal': False, 'audio': True},
            
            # T5 Temporal (most comprehensive)
            {'name': 'T5_Temporal', 'tier': CapabilityTier.T5_TEMPORAL, 'batch': 1, 'temporal': True, 'audio': True},
        ]
        
        for scenario in scenarios:
            result = self.benchmark_scenario(
                name=scenario['name'],
                tier=scenario['tier'],
                batch_size=scenario['batch'],
                temporal=scenario['temporal'],
                audio=scenario['audio']
            )
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict], output_path: str = "throughput_benchmark.json"):
        """Generate benchmark report."""
        report = {
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'summary': {}
        }
        
        # Summary statistics.
        successful = [r for r in results if r.get('status') == 'success']
        
        if successful:
            # By tier.
            by_tier = defaultdict(list)
            for r in successful:
                by_tier[r['tier']].append(r['stats']['mean_ms'])
            
            summary = {}
            for tier, times in by_tier.items():
                summary[tier] = {
                    'count': len(times),
                    'mean_latency_ms': sum(times) / len(times),
                    'min_latency_ms': min(times),
                    'max_latency_ms': max(times),
                    'mean_fps': 1000.0 / (sum(times) / len(times)),
                }
            
            report['summary'] = {
                'total_scenarios': len(results),
                'successful': len(successful),
                'failed': len(results) - len(successful),
                'by_tier': summary
            }
        
        # Save report.
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary.
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Total scenarios: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        
        if successful:
            print(f"\nLatency by Tier:")
            for tier, stats in report['summary']['by_tier'].items():
                print(f"  {tier}:")
                print(f"    Mean: {stats['mean_latency_ms']:.2f}ms ({stats['mean_fps']:.2f} FPS)")
                print(f"    Range: [{stats['min_latency_ms']:.2f}, {stats['max_latency_ms']:.2f}]ms")
        
        print(f"\nDetailed results saved to: {output_path}")
        
        return report


def main():
    """Run throughput benchmark."""
    benchmark = ThroughputBenchmark()
    
    print("\n" + "="*80)
    print("MaxSight 3.0 Throughput Benchmark")
    print("="*80)
    
    results = benchmark.run_comprehensive_benchmark()
    report = benchmark.generate_report(results)
    
    return report


if __name__ == "__main__":
    main()


