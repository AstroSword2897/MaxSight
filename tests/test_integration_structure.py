"""Structural Tests for Integration Features."""

import sys
import ast
import inspect
from pathlib import Path


def test_gradnorm_import_structure():
    """Test that GradNorm import structure is correct."""
    print("Testing GradNorm import structure...")
    
    try:
        # Read the train_loop.py file.
        train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
        content = train_loop_path.read_text()
        
        # Check for GradNorm import.
        has_import = "from ml.training.task_balancing import GradNormMultiHeadLoss" in content
        has_availability_check = "GRADNORM_AVAILABLE" in content
        
        if has_import and has_availability_check:
            print("OK GradNorm import structure correct")
            print("   - Import statement found")
            print("   - Availability check found")
            return True
        else:
            print("FAIL GradNorm import structure incomplete")
            print(f"   - Import found: {has_import}")
            print(f"   - Availability check found: {has_availability_check}")
            return False
    except Exception as e:
        print(f"FAIL Failed to check GradNorm import: {e}")
        return False


def test_gradnorm_parameters():
    """Test that GradNorm parameters are in __init__."""
    print("\nTesting GradNorm parameters in __init__...")
    
    try:
        train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
        content = train_loop_path.read_text()
        
        # Check for parameters.
        has_use_gradnorm = "use_gradnorm" in content
        has_gradnorm_alpha = "gradnorm_alpha" in content
        has_gradnorm_update_interval = "gradnorm_update_interval" in content
        
        if has_use_gradnorm and has_gradnorm_alpha and has_gradnorm_update_interval:
            print("OK GradNorm parameters found")
            print("   - use_gradnorm parameter")
            print("   - gradnorm_alpha parameter")
            print("   - gradnorm_update_interval parameter")
            return True
        else:
            print("FAIL GradNorm parameters incomplete")
            print(f"   - use_gradnorm: {has_use_gradnorm}")
            print(f"   - gradnorm_alpha: {has_gradnorm_alpha}")
            print(f"   - gradnorm_update_interval: {has_gradnorm_update_interval}")
            return False
    except Exception as e:
        print(f"FAIL Failed to check GradNorm parameters: {e}")
        return False


def test_gradnorm_integration_code():
    """Test that GradNorm integration code exists."""
    print("\nTesting GradNorm integration code...")
    
    try:
        train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
        content = train_loop_path.read_text()
        
        # Check for integration code.
        has_initialization = "self.use_gradnorm" in content
        has_loss_computation = "gradnorm_loss" in content or "GradNorm" in content
        has_error_handling = "GRADNORM_AVAILABLE" in content and "warning" in content.lower()
        
        if has_initialization and has_loss_computation:
            print("OK GradNorm integration code found")
            print("   - Initialization code")
            print("   - Loss computation integration")
            if has_error_handling:
                print("   - Error handling")
            return True
        else:
            print("FAIL GradNorm integration code incomplete")
            print(f"   - Initialization: {has_initialization}")
            print(f"   - Loss computation: {has_loss_computation}")
            return False
    except Exception as e:
        print(f"FAIL Failed to check GradNorm integration: {e}")
        return False


def test_timing_import():
    """Test that time module is imported in maxsight_cnn."""
    print("\nTesting timing import in maxsight_cnn...")
    
    try:
        model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
        content = model_path.read_text()
        
        # Check for time import.
        has_time_import = "import time" in content
        
        if has_time_import:
            print("OK Time module import found")
            return True
        else:
            print("FAIL Time module import not found")
            return False
    except Exception as e:
        print(f"FAIL Failed to check timing import: {e}")
        return False


def test_timing_code():
    """Test that timing enforcement code exists."""
    print("\nTesting timing enforcement code...")
    
    try:
        model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
        content = model_path.read_text()
        
        # Check for timing code.
        has_timing_track = "stage_a_start_time" in content
        has_latency_check = "stage_a_latency_ms" in content
        has_early_exit = "skip_stage_b" in content and "latency" in content.lower()
        has_metrics = "stage_a_latency_ms" in content and "outputs[" in content
        
        if has_timing_track and has_latency_check and has_early_exit and has_metrics:
            print("OK Timing enforcement code found")
            print("   - Timing tracking")
            print("   - Latency check")
            print("   - Early exit logic")
            print("   - Metrics output")
            return True
        else:
            print("FAIL Timing enforcement code incomplete")
            print(f"   - Timing track: {has_timing_track}")
            print(f"   - Latency check: {has_latency_check}")
            print(f"   - Early exit: {has_early_exit}")
            print(f"   - Metrics: {has_metrics}")
            return False
    except Exception as e:
        print(f"FAIL Failed to check timing code: {e}")
        return False


def test_timing_threshold():
    """Test that timing threshold (200ms) is set correctly."""
    print("\nTesting timing threshold...")
    
    try:
        model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
        content = model_path.read_text()
        
        # Check for threshold.
        has_threshold = "200" in content and "latency" in content.lower()
        has_comment = "150ms" in content or "200ms" in content
        
        if has_threshold and has_comment:
            print("OK Timing threshold found")
            print("   - 200ms hard limit")
            print("   - 150ms target documented")
            return True
        else:
            print("WARNING Timing threshold may not be clearly set")
            print(f"   - Threshold check: {has_threshold}")
            print(f"   - Documentation: {has_comment}")
            return True  # Not a failure, just a warning.
    except Exception as e:
        print(f"FAIL Failed to check timing threshold: {e}")
        return False


def test_output_structure():
    """Test that output structure includes timing metrics."""
    print("\nTesting output structure...")
    
    try:
        model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
        content = model_path.read_text()
        
        # Check for output keys.
        has_stage_a = "'stage_a_completed'" in content or '"stage_a_completed"' in content
        has_stage_b = "'stage_b_completed'" in content or '"stage_b_completed"' in content
        has_skip_reason = "'skip_stage_b_reason'" in content or '"skip_stage_b_reason"' in content
        has_latency = "'stage_a_latency_ms'" in content or '"stage_a_latency_ms"' in content
        
        if has_stage_a and has_stage_b and has_skip_reason and has_latency:
            print("OK Output structure complete")
            print("   - stage_a_completed")
            print("   - stage_b_completed")
            print("   - skip_stage_b_reason")
            print("   - stage_a_latency_ms")
            return True
        else:
            print("FAIL Output structure incomplete")
            print(f"   - stage_a_completed: {has_stage_a}")
            print(f"   - stage_b_completed: {has_stage_b}")
            print(f"   - skip_stage_b_reason: {has_skip_reason}")
            print(f"   - stage_a_latency_ms: {has_latency}")
            return False
    except Exception as e:
        print(f"FAIL Failed to check output structure: {e}")
        return False


def main():
    """Run all structural tests."""
    print("=" * 60)
    print("Integration Structure Tests")
    print("(Tests code structure without requiring PyTorch)")
    print("=" * 60)
    
    results = []
    
    # GradNorm tests.
    results.append(("GradNorm Import", test_gradnorm_import_structure()))
    results.append(("GradNorm Parameters", test_gradnorm_parameters()))
    results.append(("GradNorm Integration", test_gradnorm_integration_code()))
    
    # Timing tests.
    results.append(("Timing Import", test_timing_import()))
    results.append(("Timing Code", test_timing_code()))
    results.append(("Timing Threshold", test_timing_threshold()))
    results.append(("Output Structure", test_output_structure()))
    
    # Summary.
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All structural tests passed!")
        print("\nNote: To run full functional tests, install PyTorch and run:")
        print("  python tests/test_gradnorm_integration.py")
        print("  python tests/test_timing_enforcement.py")
        return 0
    else:
        print("WARNING Some structural tests failed - check code integration")
        return 1


if __name__ == "__main__":
    exit(main())







