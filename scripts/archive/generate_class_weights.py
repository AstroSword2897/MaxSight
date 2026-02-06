#!/usr/bin/env python3
"""Generate Class Weights for Weighted Loss..."""

import json
from pathlib import Path
from collections import Counter
import numpy as np

REPORT_FILE = Path("datasets/cleaned_splits/class_distribution_report.json")
OUTPUT_FILE = Path("datasets/cleaned_splits/class_weights.json")


def generate_detection_weights(distribution: dict, total_classes: int, 
                               method: str = 'inverse_sqrt') -> dict:
    """Generate class weights for detection head...."""
    # Get all class frequencies (0 for missing classes)
    all_classes = sorted(distribution.keys())
    frequencies = [distribution.get(cls, 0) for cls in all_classes]
    total_samples = sum(frequencies)
    
    if total_samples == 0:
        return {cls: 1.0 for cls in all_classes}
    
    # Normalize frequencies
    normalized_freq = [f / total_samples if total_samples > 0 else 0 for f in frequencies]
    
    if method == 'inverse_sqrt':
        # Inverse square root: less aggressive than pure inverse
        weights = [1.0 / np.sqrt(f + 1e-6) if f > 0 else 10.0 for f in normalized_freq]
    elif method == 'inverse':
        # Pure inverse: more aggressive
        weights = [1.0 / (f + 1e-6) if f > 0 else 10.0 for f in normalized_freq]
    elif method == 'focal_alpha':
        # Alpha values for Focal Loss (higher = more important)
        # Rare classes get higher alpha
        weights = [min(1.0, 0.25 / (f + 1e-6)) if f > 0 else 1.0 for f in normalized_freq]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to have mean=1.0 (optional, helps with learning rate)
    mean_weight = np.mean(weights)
    weights = [w / mean_weight for w in weights]
    
    return {cls: float(w) for cls, w in zip(all_classes, weights)}


def generate_urgency_weights() -> dict:
    """Generate weights for urgency head...."""
    return {
        0: 0.5,  # Safe - downweight
        1: 2.0,  # Caution - upweight
        2: 2.0,  # Warning - upweight
        3: 5.0   # Danger - heavily upweight
    }


def generate_distance_weights() -> dict:
    """Generate weights for distance head.
    
    Based on typical distribution:
    - Zone 0 (near): ~9% - weight 2.0
    - Zone 1 (medium): ~39% - weight 1.0
    - Zone 2 (far): ~52% - weight 0.5"""
    return {
        0: 2.0,  # Near - upweight
        1: 1.0,  # Medium - balanced
        2: 0.5   # Far - downweight
    }


def main():
    """Main execution."""
    if not REPORT_FILE.exists():
        print(f"❌ Class distribution report not found: {REPORT_FILE}")
        print("   Run scripts/fix_dataset_splits.py first!")
        return
    
    print("=" * 80)
    print("GENERATING CLASS WEIGHTS FOR WEIGHTED LOSS")
    print("=" * 80)
    
    # Load distribution report
    with open(REPORT_FILE, 'r') as f:
        reports = json.load(f)
    
    # Use train split for weight calculation
    train_report = reports.get('train', {})
    if not train_report:
        print("❌ Train split not found in report!")
        return
    
    distribution = train_report.get('distribution', {})
    total_classes = train_report.get('total_categories', 0)
    
    print(f"\n📊 Using train split distribution:")
    print(f"  Total categories: {total_classes}")
    print(f"  Annotated categories: {train_report.get('annotated_categories', 0)}")
    print(f"  Zero-annotation classes: {train_report.get('zero_annotation', 0)}")
    print(f"  Rare classes (<5 samples): {train_report.get('rare_<5', 0)}")
    
    # Generate weights
    print("\n🔧 Generating class weights...")
    
    # Detection head weights (multiple methods)
    detection_weights_inverse_sqrt = generate_detection_weights(
        distribution, total_classes, method='inverse_sqrt'
    )
    detection_weights_inverse = generate_detection_weights(
        distribution, total_classes, method='inverse'
    )
    focal_alpha = generate_detection_weights(
        distribution, total_classes, method='focal_alpha'
    )
    
    # Urgency and distance weights
    urgency_weights = generate_urgency_weights()
    distance_weights = generate_distance_weights()
    
    # Compile output
    output = {
        'detection_head': {
            'inverse_sqrt': detection_weights_inverse_sqrt,
            'inverse': detection_weights_inverse,
            'focal_alpha': focal_alpha,
            'recommendation': 'Use inverse_sqrt for balanced training, or inverse for more aggressive rare-class focus'
        },
        'urgency_head': {
            'weights': urgency_weights,
            'recommendation': 'Apply these weights to CrossEntropyLoss weight parameter'
        },
        'distance_head': {
            'weights': distance_weights,
            'recommendation': 'Apply these weights to CrossEntropyLoss weight parameter'
        },
        'statistics': {
            'total_classes': total_classes,
            'zero_annotation_classes': train_report.get('zero_annotation', 0),
            'rare_classes_<5': train_report.get('rare_<5', 0),
            'very_rare_classes_<2': train_report.get('very_rare_<2', 0)
        }
    }
    
    # Save weights
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Class weights saved: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("WEIGHT SUMMARY")
    print("=" * 80)
    
    # Show weight ranges
    inv_sqrt_vals = list(detection_weights_inverse_sqrt.values())
    inv_vals = list(detection_weights_inverse.values())
    
    print(f"\nDetection Head Weights (inverse_sqrt method):")
    print(f"  Min: {min(inv_sqrt_vals):.3f}")
    print(f"  Max: {max(inv_sqrt_vals):.3f}")
    print(f"  Mean: {np.mean(inv_sqrt_vals):.3f}")
    print(f"  Classes with weight > 5.0: {sum(1 for w in inv_sqrt_vals if w > 5.0)}")
    
    print(f"\nDetection Head Weights (inverse method):")
    print(f"  Min: {min(inv_vals):.3f}")
    print(f"  Max: {max(inv_vals):.3f}")
    print(f"  Mean: {np.mean(inv_vals):.3f}")
    print(f"  Classes with weight > 5.0: {sum(1 for w in inv_vals if w > 5.0)}")
    
    print(f"\nUrgency Head Weights:")
    for level, weight in urgency_weights.items():
        print(f"  Level {level}: {weight:.1f}")
    
    print(f"\nDistance Head Weights:")
    for zone, weight in distance_weights.items():
        print(f"  Zone {zone}: {weight:.1f}")
    
    print("\n📝 Usage in PyTorch:")
    print("  # Detection head (inverse_sqrt weights)")
    print("  import torch")
    print("  weights = torch.tensor([weights_dict['detection_head']['inverse_sqrt'][cls]")
    print("                          for cls in sorted_classes])")
    print("  criterion = nn.CrossEntropyLoss(weight=weights)")
    print("\n  # Urgency head")
    print("  urgency_weights = torch.tensor([weights_dict['urgency_head']['weights'][i]")
    print("                                    for i in range(4)])")
    print("  urgency_criterion = nn.CrossEntropyLoss(weight=urgency_weights)")


if __name__ == "__main__":
    main()

