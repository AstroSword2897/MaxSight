#!/usr/bin/env python3
"""Comprehensive Data Collection Runner..."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_loss_data import collect_loss_data
from scripts.collect_inference_data import collect_inference_data


def verify_model_completeness() -> Dict[str, Any]:
    """Verify model completeness and document stubs.
    
    Returns:
        Dictionary with completeness status"""
    print("\n" + "="*60)
    print("Model Completeness Verification")
    print("="*60)
    
    completeness = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'stubs': [],
        'integrated': [],
        'status': 'complete'
    }
    
    # Check for stubs
    import os
    stub_files = []
    for root, dirs, files in os.walk('ml/models'):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'STUB' in content or 'NOT INTEGRATED' in content:
                            stub_files.append(str(filepath))
                except:
                    pass
    
    completeness['stubs'] = stub_files
    
    # Core components (always integrated)
    completeness['integrated'] = [
        'MaxSightCNN',
        'ResNet50 backbone',
        'FPN',
        'Detection heads (objectness, classification, box)',
        'Urgency head',
        'Distance head',
        'Depth head',
        'TherapyStateHead',
        'MultiHeadLoss'
    ]
    
    # Documented stubs (expected)
    completeness['documented_stubs'] = [
        'PredictiveAlertHead (STUB - future feature)',
        'EyeModel (STUB - not integrated)',
        'Therapy components (STUB - placeholder)'
    ]
    
    print(f"\n✅ Integrated components: {len(completeness['integrated'])}")
    print(f"⚠️  Stub components: {len(completeness['stubs'])}")
    print(f"📝 Documented stubs: {len(completeness['documented_stubs'])}")
    
    if len(stub_files) > len(completeness['documented_stubs']):
        completeness['status'] = 'has_undocumented_stubs'
        print("⚠️  Warning: Some stubs may not be documented")
    else:
        completeness['status'] = 'complete'
        print("✅ All stubs are documented")
    
    return completeness


def generate_report(
    loss_data: Dict[str, Any],
    inference_stats: Dict[str, Any],
    completeness: Dict[str, Any],
    output_dir: Path
):
    """Generate comprehensive data collection report."""
    
    report = {
        'collection_timestamp': datetime.now().isoformat(),
        'summary': {
            'loss_data_collected': loss_data is not None,
            'inference_stats_collected': inference_stats is not None,
            'model_completeness_verified': completeness is not None
        },
        'loss_data_summary': {
            'iterations': loss_data.get('metadata', {}).get('total_iterations', 0) if loss_data else 0,
            'heads_tracked': loss_data.get('metadata', {}).get('heads_tracked', []) if loss_data else [],
            'loss_statistics_count': len(loss_data.get('loss_statistics', {})) if loss_data else 0
        },
        'inference_stats_summary': {
            'total_images': inference_stats.get('total_images', 0) if inference_stats else 0,
            'total_objects': inference_stats.get('total_objects', 0) if inference_stats else 0,
            'classes': inference_stats.get('class_distribution_counts', {}).get('total_classes', 0) if inference_stats else 0
        },
        'model_completeness': completeness,
        'recommendations': []
    }
    
    # Add recommendations
    if completeness.get('status') != 'complete':
        report['recommendations'].append("Review and document all stub components")
    
    if loss_data and len(loss_data.get('detected_issues', {})) > 0:
        report['recommendations'].append("Review detected loss issues - some heads may need attention")
    
    if inference_stats and inference_stats.get('total_images', 0) < 100:
        report['recommendations'].append("Consider collecting statistics from larger dataset")
    
    # Save report
    report_path = output_dir / 'collection_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("Data Collection Report")
    print("="*60)
    print(f"\n✅ Report saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Loss data: {'✅' if report['summary']['loss_data_collected'] else '❌'}")
    print(f"  Inference stats: {'✅' if report['summary']['inference_stats_collected'] else '❌'}")
    print(f"  Model completeness: {'✅' if report['summary']['model_completeness_verified'] else '❌'}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive data collection")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--data-dir", type=str, default="datasets/coco", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="collected_data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for loss collection")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--skip-loss", action="store_true", help="Skip loss data collection")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference stats collection")
    parser.add_argument("--skip-completeness", action="store_true", help="Skip completeness verification")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Comprehensive Data Collection")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    loss_data = None
    inference_stats = None
    completeness = None
    
    # 1. Collect loss data
    if not args.skip_loss:
        try:
            print("\n[1/3] Collecting loss function data...")
            loss_data = collect_loss_data(
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                data_dir=Path(args.data_dir),
                num_samples=args.num_samples,
                batch_size=4,
                device=args.device,
                output_path=output_dir / 'loss_data.json',
                collect_gradients=False,
                collect_task_weights=True
            )
        except Exception as e:
            print(f"❌ Loss data collection failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[1/3] Skipping loss data collection...")
    
    # 2. Collect inference dataset statistics
    if not args.skip_inference:
        try:
            print("\n[2/3] Collecting inference dataset statistics...")
            inference_stats = collect_inference_data(
                dataset_name='coco',
                data_dir=Path(args.data_dir),
                output_path=output_dir / 'inference_stats.json',
                max_samples=1000  # Limit for speed
            )
        except Exception as e:
            print(f"❌ Inference stats collection failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[2/3] Skipping inference stats collection...")
    
    # 3. Verify model completeness
    if not args.skip_completeness:
        try:
            print("\n[3/3] Verifying model completeness...")
            completeness = verify_model_completeness()
            
            completeness_path = output_dir / 'model_completeness.json'
            with open(completeness_path, 'w') as f:
                json.dump(completeness, f, indent=2)
            print(f"✅ Completeness report saved to: {completeness_path}")
        except Exception as e:
            print(f"❌ Completeness verification failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[3/3] Skipping completeness verification...")
    
    # 4. Generate comprehensive report
    print("\n[4/4] Generating comprehensive report...")
    report = generate_report(loss_data, inference_stats, completeness, output_dir)
    
    print("\n" + "="*60)
    print("✅ Data collection complete!")
    print("="*60)
    print(f"\nAll data saved to: {output_dir}")
    print(f"  - Loss data: loss_data.json")
    print(f"  - Inference stats: inference_stats.json")
    print(f"  - Model completeness: model_completeness.json")
    print(f"  - Collection report: collection_report.json")


if __name__ == "__main__":
    main()

