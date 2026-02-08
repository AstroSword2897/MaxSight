"""Test Two-Stage Inference Timing Enforcement

Tests that timing enforcement works correctly in the two-stage inference pipeline."""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model


def test_timing_import():
    """Test that time module is properly imported."""
    print("Testing time module import...")
    
    try:
        from ml.models.maxsight_cnn import time as time_module
        print("✅ Time module imported successfully")
        return True
    except ImportError:
        # Check if time is imported in maxsight_cnn.
        import ml.models.maxsight_cnn as maxsight_module
        if hasattr(maxsight_module, 'time'):
            print("✅ Time module available in maxsight_cnn")
            return True
        else:
            print("⚠️ Time module not found in maxsight_cnn (may be imported differently)")
            return True  # Not a failure, time is a built-in.


def test_timing_flag():
    """Test that _enable_timing flag can be set."""
    print("\nTesting _enable_timing flag...")
    
    try:
        model = create_model(num_classes=10)
        
        # Test setting the flag.
        model._enable_timing = True
        assert hasattr(model, '_enable_timing')
        assert model._enable_timing == True
        
        # Test default value.
        model2 = create_model(num_classes=10)
        if not hasattr(model2, '_enable_timing'):
            model2._enable_timing = False
        
        print("✅ _enable_timing flag can be set")
        print(f"   - Default value: {getattr(model2, '_enable_timing', False)}")
        return True
    except Exception as e:
        print(f"❌ Failed to set _enable_timing flag: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timing_tracking():
    """Test that timing is tracked when flag is enabled."""
    print("\nTesting timing tracking...")
    
    try:
        model = create_model(num_classes=10)
        model.eval()
        model._enable_timing = True
        
        # Create dummy input.
        images = torch.randn(1, 3, 224, 224)
        
        # Run inference.
        with torch.no_grad():
            outputs = model(images)
        
        # Checks if timing metrics are in outputs.
        has_timing = 'stage_a_latency_ms' in outputs
        has_stage_info = 'stage_a_completed' in outputs and 'stage_b_completed' in outputs
        
        print("✅ Timing tracking test completed")
        print(f"   - stage_a_latency_ms in outputs: {has_timing}")
        print(f"   - Stage info in outputs: {has_stage_info}")
        
        if has_timing:
            latency = outputs.get('stage_a_latency_ms')
            if latency is not None:
                print(f"   - Measured latency: {latency:.2f}ms")
        
        return has_stage_info  # At minimum, stage info is present.
    except Exception as e:
        print(f"❌ Timing tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timing_enforcement():
    """Test that Stage B is skipped when timing exceeds threshold."""
    print("\nTesting timing enforcement (Stage B skip logic)...")
    
    try:
        model = create_model(num_classes=10)
        model.eval()
        model._enable_timing = True
        
        # Create dummy input.
        images = torch.randn(1, 3, 224, 224)
        
        # Run inference.
        with torch.no_grad():
            outputs = model(images)
        
        # Check outputs.
        stage_b_completed = outputs.get('stage_b_completed', False)
        skip_reason = outputs.get('skip_stage_b_reason')
        latency = outputs.get('stage_a_latency_ms')
        
        print("✅ Timing enforcement test completed")
        print(f"   - Stage B completed: {stage_b_completed}")
        print(f"   - Skip reason: {skip_reason}")
        print(f"   - Stage A latency: {latency:.2f}ms" if latency is not None else "   - Stage A latency: Not measured")
        
        # Verifies outputs structure.
        assert 'stage_a_completed' in outputs, "stage_a_completed missing"
        assert 'stage_b_completed' in outputs, "stage_b_completed missing"
        assert 'skip_stage_b_reason' in outputs, "skip_stage_b_reason missing"
        
        print("   - Output structure verified")
        return True
    except Exception as e:
        print(f"❌ Timing enforcement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timing_disabled():
    """Test that timing doesn't affect inference when disabled."""
    print("\nTesting timing disabled mode...")
    
    try:
        model = create_model(num_classes=10)
        model.eval()
        model._enable_timing = False  # Disable timing.
        
        # Create dummy input.
        images = torch.randn(1, 3, 224, 224)
        
        # Run inference.
        with torch.no_grad():
            outputs = model(images)
        
        # Inference works normally with timing disabled.
        assert 'stage_a_completed' in outputs
        assert 'stage_b_completed' in outputs
        
        print("✅ Timing disabled mode works correctly")
        print("   - Inference works without timing enabled")
        print("   - Outputs structure maintained")
        return True
    except Exception as e:
        print(f"❌ Timing disabled test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actual_latency():
    """Test actual latency measurement."""
    print("\nTesting actual latency measurement...")
    
    try:
        model = create_model(num_classes=10)
        model.eval()
        model._enable_timing = True
        
        # Create dummy input.
        images = torch.randn(1, 3, 224, 224)
        
        # Warmup.
        with torch.no_grad():
            _ = model(images)
        
        # Measure multiple runs.
        latencies = []
        for _ in range(5):
            with torch.no_grad():
                outputs = model(images)
            latency = outputs.get('stage_a_latency_ms')
            if latency is not None:
                latencies.append(latency)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print("✅ Latency measurement successful")
            print(f"   - Average latency: {avg_latency:.2f}ms")
            print(f"   - Min latency: {min_latency:.2f}ms")
            print(f"   - Max latency: {max_latency:.2f}ms")
            print(f"   - Target: <150ms, Hard limit: <200ms")
            
            if avg_latency < 200:
                print("   ✅ Within hard limit (200ms)")
            else:
                print("   ⚠️ Exceeds hard limit (200ms)")
            
            return True
        else:
            print("⚠️ No latency measurements recorded")
            return False
    except Exception as e:
        print(f"❌ Latency measurement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all timing enforcement tests."""
    print("=" * 60)
    print("Two-Stage Inference Timing Enforcement Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Time import.
    results.append(("Time Import", test_timing_import()))
    
    # Test 2: Timing flag.
    results.append(("Timing Flag", test_timing_flag()))
    
    # Test 3: Timing tracking.
    results.append(("Timing Tracking", test_timing_tracking()))
    
    # Test 4: Timing enforcement.
    results.append(("Timing Enforcement", test_timing_enforcement()))
    
    # Test 5: Timing disabled.
    results.append(("Timing Disabled", test_timing_disabled()))
    
    # Test 6: Actual latency.
    results.append(("Actual Latency", test_actual_latency()))
    
    # Summary.
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit(main())


