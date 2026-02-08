"""Comprehensive Forward Pass Analysis for MaxSight 3.0

Analyzes all possible forward pass scenarios and measures throughput
to understand computational flow before integration and training."""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from collections import defaultdict

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import MaxSightCNN, create_model, TierConfig, CapabilityTier


class ForwardPassAnalyzer:
    """Analyzes forward pass scenarios and throughput."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = []
        self.scenarios = []
        
    def get_device(self):
        """Get the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def create_scenarios(self) -> List[Dict]:
        """Define all forward pass scenarios to test."""
        scenarios = []
        
        # Base scenarios.
        base_configs = [
            {'name': 'T0_Baseline', 'tier': CapabilityTier.T0_BASELINE_CNN},
            {'name': 'T1_Attention', 'tier': CapabilityTier.T1_ATTENTION},
            {'name': 'T2_Hybrid', 'tier': CapabilityTier.T2_HYBRID_VIT},
            {'name': 'T3_CrossTask', 'tier': CapabilityTier.T3_CROSS_TASK},
            {'name': 'T4_CrossModal', 'tier': CapabilityTier.T4_CROSS_MODAL},
            {'name': 'T5_Temporal', 'tier': CapabilityTier.T5_TEMPORAL},
        ]
        
        # Input variations.
        input_variations = [
            {'batch_size': 1, 'temporal': False, 'audio': False, 'name_suffix': '_single_image'},
            {'batch_size': 4, 'temporal': False, 'audio': False, 'name_suffix': '_batch4'},
            {'batch_size': 1, 'temporal': True, 'audio': False, 'name_suffix': '_temporal_seq8'},
            {'batch_size': 1, 'temporal': False, 'audio': True, 'name_suffix': '_with_audio'},
            {'batch_size': 1, 'temporal': True, 'audio': True, 'name_suffix': '_temporal_audio'},
        ]
        
        # Condition modes.
        condition_modes = [None, 'glaucoma', 'amd', 'cataracts', 'cvi']
        
        # Generate all combinations.
        for base_config in base_configs:
            for input_var in input_variations:
                for condition in condition_modes:
                    scenario = {
                        'name': f"{base_config['name']}{input_var['name_suffix']}_{condition or 'normal'}",
                        'tier': base_config['tier'],
                        'batch_size': input_var['batch_size'],
                        'temporal': input_var['temporal'],
                        'audio': input_var['audio'],
                        'condition_mode': condition,
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def create_model_for_scenario(self, scenario: Dict) -> MaxSightCNN:
        """Create model configured for a scenario."""
        tier_config = TierConfig.for_tier(scenario['tier'])
        
        model = create_model(
            num_classes=91,  # COCO classes.
            use_audio=scenario['audio'],
            condition_mode=scenario['condition_mode'],
            tier_config=tier_config
        )
        
        model.eval()
        model = model.to(self.device)
        
        return model
    
    def create_inputs_for_scenario(self, scenario: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create input tensors for a scenario."""
        batch_size = scenario['batch_size']
        
        if scenario['temporal']:
            # Temporal: [B, T, C, H, W].
            images = torch.randn(batch_size, 8, 3, 224, 224, device=self.device)
        else:
            # Single frame: [B, C, H, W].
            images = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        audio_features = None
        if scenario['audio']:
            audio_features = torch.randn(batch_size, 128, device=self.device)
        
        return images, audio_features
    
    def measure_forward_pass(
        self,
        model: MaxSightCNN,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor],
        scenario: Dict,
        num_warmup: int = 5,
        num_runs: int = 20
    ) -> Dict:
        """Measure forward pass performance."""
        # Warmup.
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(images, audio_features=audio_features, use_temporal=scenario['temporal'])
        
        # Synchronize if using GPU.
        if self.device != 'cpu':
            torch.cuda.synchronize() if self.device == 'cuda' else None
        
        # Measure.
        times = []
        memory_used = []
        
        for _ in range(num_runs):
            if self.device != 'cpu':
                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    outputs = model(images, audio_features=audio_features, use_temporal=scenario['temporal'])
                torch.cuda.synchronize()
                elapsed = time.time() - start
                memory_used.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB.
            else:
                start = time.time()
                with torch.no_grad():
                    outputs = model(images, audio_features=audio_features, use_temporal=scenario['temporal'])
                elapsed = time.time() - start
                memory_used.append(0)  # CPU doesn't track memory easily.
            
            times.append(elapsed * 1000)  # Convert to ms.
        
        # Analyze outputs.
        output_keys = list(outputs.keys())
        output_shapes = {k: list(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) 
                        for k, v in outputs.items()}
        
        return {
            'mean_time_ms': sum(times) / len(times),
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'p50_time_ms': sorted(times)[len(times)//2],
            'p95_time_ms': sorted(times)[int(len(times)*0.95)],
            'p99_time_ms': sorted(times)[int(len(times)*0.99)],
            'mean_memory_mb': sum(memory_used) / len(memory_used) if memory_used else 0,
            'max_memory_mb': max(memory_used) if memory_used else 0,
            'output_keys': output_keys,
            'output_shapes': output_shapes,
            'num_outputs': len(output_keys),
        }
    
    def analyze_scenario(self, scenario: Dict) -> Dict:
        """Analyze a single scenario."""
        print(f"\n{'='*80}")
        print(f"Analyzing: {scenario['name']}")
        print(f"{'='*80}")
        
        try:
            # Create model.
            print("  Creating model...")
            model = self.create_model_for_scenario(scenario)
            
            # Create inputs.
            print("  Creating inputs...")
            images, audio_features = self.create_inputs_for_scenario(scenario)
            
            # Measure.
            print("  Measuring forward pass...")
            metrics = self.measure_forward_pass(model, images, audio_features, scenario)
            
            result = {
                'scenario': scenario,
                'metrics': metrics,
                'status': 'success'
            }
            
            print(f"  OK Mean time: {metrics['mean_time_ms']:.2f}ms")
            print(f"  OK Output keys: {len(metrics['output_keys'])}")
            
            return result
            
        except Exception as e:
            print(f"  FAIL Failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'scenario': scenario,
                'status': 'failed',
                'error': str(e)
            }
    
    def analyze_all_scenarios(self, limit: Optional[int] = None):
        """Analyze all scenarios."""
        scenarios = self.create_scenarios()
        
        if limit:
            scenarios = scenarios[:limit]
        
        print(f"\n{'='*80}")
        print(f"Forward Pass Analysis: {len(scenarios)} scenarios")
        print(f"Device: {self.device}")
        print(f"{'='*80}")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] Processing scenario...")
            result = self.analyze_scenario(scenario)
            self.results.append(result)
            self.scenarios.append(scenario)
    
    def generate_report(self, output_path: str = "forward_pass_analysis.json"):
        """Generate comprehensive analysis report."""
        report = {
            'device': self.device,
            'total_scenarios': len(self.results),
            'successful': sum(1 for r in self.results if r['status'] == 'success'),
            'failed': sum(1 for r in self.results if r['status'] == 'failed'),
            'scenarios': []
        }
        
        # Organize by tier.
        by_tier = defaultdict(list)
        for result in self.results:
            if result['status'] == 'success':
                tier_name = result['scenario']['tier'].name
                by_tier[tier_name].append(result)
        
        # Summary statistics.
        summary = {}
        for tier, results in by_tier.items():
            times = [r['metrics']['mean_time_ms'] for r in results]
            summary[tier] = {
                'count': len(results),
                'mean_time_ms': sum(times) / len(times) if times else 0,
                'min_time_ms': min(times) if times else 0,
                'max_time_ms': max(times) if times else 0,
            }
        
        report['summary_by_tier'] = summary
        report['detailed_results'] = self.results
        
        # Save report.
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"Analysis Report Generated: {output_path}")
        print(f"{'='*80}")
        print(f"Total scenarios: {report['total_scenarios']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"\nSummary by Tier:")
        for tier, stats in summary.items():
            print(f"  {tier}: {stats['count']} scenarios, "
                  f"mean={stats['mean_time_ms']:.2f}ms, "
                  f"range=[{stats['min_time_ms']:.2f}, {stats['max_time_ms']:.2f}]ms")
        
        return report


def main():
    """Run forward pass analysis."""
    analyzer = ForwardPassAnalyzer()
    device = analyzer.get_device()
    analyzer.device = device
    
    print(f"Using device: {device}")
    
    # Analyze a subset first (can expand later)
    print("\nStarting forward pass analysis...")
    analyzer.analyze_all_scenarios(limit=30)  # Start with 30 scenarios.
    
    # Generate report.
    report = analyzer.generate_report("forward_pass_analysis.json")
    
    return report


if __name__ == "__main__":
    main()


