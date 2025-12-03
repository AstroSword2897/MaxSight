"""Test model robustness with condition-specific impairment simulations."""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import (
    apply_refractive_error_blur,
    apply_cataract_contrast,
    apply_glaucoma_peripheral_mask,
    apply_amd_central_darkening,
    apply_rp_low_light,
    apply_color_blindness_shift
)


def test_condition_robustness():
    """Test model performance with each impairment simulation."""
    print("Condition-Specific Robustness Testing")
    
    model = create_model()
    model.eval()
    
    # Create test image
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Baseline: normal image
    print("\n1. Baseline (Normal Image)")
    with torch.no_grad():
        baseline_outputs = model(dummy_image)
    baseline_detections = model.get_detections(baseline_outputs, confidence_threshold=0.3)  # type: ignore
    baseline_count = len(baseline_detections[0]) if baseline_detections else 0
    print(f"   Detections: {baseline_count}")
    
    # Test each condition
    conditions = [
        ("Refractive Errors (Blur)", lambda img: apply_refractive_error_blur(img, sigma=3.0)),
        ("Cataracts (Contrast Reduction)", lambda img: apply_cataract_contrast(img, factor=0.5)),
        ("Glaucoma (Peripheral Mask)", lambda img: apply_glaucoma_peripheral_mask(img)),
        ("AMD (Central Darkening)", lambda img: apply_amd_central_darkening(img)),
        ("Retinitis Pigmentosa (Low Light)", lambda img: apply_rp_low_light(img)),
        ("Color Blindness (Color Shift)", lambda img: apply_color_blindness_shift(img, mode='protanopia')),
    ]
    
    results = []
    
    for condition_name, transform_fn in conditions:
        print(f"\n{len(results) + 2}. {condition_name}")
        
        # Apply impairment
        impaired_image = transform_fn(dummy_image.clone())
        
        # Run inference
        with torch.no_grad():
            impaired_outputs = model(impaired_image)
        impaired_detections = model.get_detections(impaired_outputs, confidence_threshold=0.3)  # type: ignore
        impaired_count = len(impaired_detections[0]) if impaired_detections else 0
        
        # Calculate degradation
        if baseline_count > 0:
            degradation = abs(baseline_count - impaired_count) / baseline_count * 100
        else:
            degradation = 0.0
        
        print(f"   Detections: {impaired_count}")
        print(f"   Degradation: {degradation:.1f}%")
        
        # Check if degradation is acceptable (<10%)
        status = "PASS" if degradation < 10.0 else "FAIL"
        print(f"   Status: {status} (<10% degradation target)")
        
        results.append({
            'condition': condition_name,
            'baseline_count': baseline_count,
            'impaired_count': impaired_count,
            'degradation': degradation,
            'passed': degradation < 10.0
        })
    
    # Summary
    print("\nSummary")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Target: All conditions <10% degradation")
    print(f"Status: {'ALL TESTS PASSED' if passed == total else 'SOME TESTS FAILED'}")
    
    return results


if __name__ == "__main__":
    test_condition_robustness()

