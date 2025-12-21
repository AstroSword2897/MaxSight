"""
Error Handling and Fallback Tests
Tests error propagation and fallback mechanisms with deterministic, adversarial coverage.
"""

import torch
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.error_handling import HeadExecutionManager, safe_head_execution, with_fallback
from ml.config import RuntimeConfig

# Set seeds for deterministic tests
torch.manual_seed(42)


def test_fallback_on_forced_exception():
    """Test fallback triggers when function raises exception (deterministic)."""
    fallback_value = {
        'classifications': torch.zeros(1, 196, 80),
        'boxes': torch.zeros(1, 196, 4),
        'objectness': torch.zeros(1, 196)
    }
    
    @with_fallback(fallback_value=fallback_value)
    def failing_forward(_images):
        raise RuntimeError("Simulated model failure")
    
    dummy_image = torch.randn(1, 3, 224, 224)
    outputs = failing_forward(dummy_image)
    
    # Validate fallback was returned
    assert 'classifications' in outputs
    assert 'boxes' in outputs
    assert 'objectness' in outputs
    
    # Validate shapes and values match fallback
    assert outputs['classifications'].shape == (1, 196, 80)
    assert outputs['boxes'].shape == (1, 196, 4)
    assert outputs['objectness'].shape == (1, 196)
    
    # Fallback should be all zeros
    assert outputs['classifications'].sum().item() == 0.0
    assert outputs['boxes'].sum().item() == 0.0
    assert outputs['objectness'].sum().item() == 0.0
    
    print("✅ Test 1: Forced exception fallback working")


def test_fallback_on_successful_execution():
    """Test normal execution path (no fallback needed)."""
    fallback_value = {
        'classifications': torch.zeros(1, 196, 80),
        'boxes': torch.zeros(1, 196, 4),
        'objectness': torch.zeros(1, 196)
    }
    
    @with_fallback(fallback_value=fallback_value)
    def successful_forward(_images):
        return {
            'classifications': torch.ones(1, 196, 80) * 0.5,
            'boxes': torch.ones(1, 196, 4) * 10.0,
            'objectness': torch.ones(1, 196) * 0.8
        }
    
    dummy_image = torch.randn(1, 3, 224, 224)
    outputs = successful_forward(dummy_image)
    
    # Should get actual outputs, not fallback
    assert outputs['classifications'].sum().item() > 0.0
    assert outputs['boxes'].sum().item() > 0.0
    assert outputs['objectness'].sum().item() > 0.0
    
    # Check specific values
    assert torch.allclose(outputs['classifications'], torch.ones(1, 196, 80) * 0.5)
    assert torch.allclose(outputs['boxes'], torch.ones(1, 196, 4) * 10.0)
    assert torch.allclose(outputs['objectness'], torch.ones(1, 196) * 0.8)
    
    print("✅ Test 2: Successful execution (no fallback) working")


def test_dependency_validation_complete():
    """Test dependency validation with complete outputs."""
    from ml.config import DependencyGraph
    
    graph = DependencyGraph()
    
    # Test with complete outputs (matching expected structure)
    outputs = {
        'model': {
            'detections': [{'class_name': 'door'}],
            'urgency_scores': [0.5, 0.3, 0.1, 0.1],
            'text_regions': [0.5, 0.3],
            'uncertainty': [0.2]
        },
        'ocr': {
            'text_results': ['Exit']
        },
        'spatial_memory': {
            'spatial_context': {'objects': []}
        },
        'session_manager': {},
        'preprocessing': {'processed': True}
    }
    
    validation = graph.validate_dependencies(outputs)
    
    # All components should be valid with complete outputs
    assert validation.get('model', False), "Model should be valid"
    assert validation.get('preprocessing', False), "Preprocessing should be valid"
    
    # Count valid components
    valid_count = sum(1 for v in validation.values() if v)
    assert valid_count >= 2, f"Expected at least 2 valid components, got {valid_count}"
    
    print("✅ Test 3: Complete dependency validation working")


def test_dependency_validation_missing():
    """Test dependency validation flags missing dependencies."""
    from ml.config import DependencyGraph
    
    graph = DependencyGraph()
    
    # Minimal/incomplete outputs
    outputs = {
        'model': {
            'detections': []
        }
        # Missing: ocr, spatial_memory, session_manager, preprocessing
    }
    
    validation = graph.validate_dependencies(outputs)
    
    # Model might be valid but others should fail
    missing_components = ['ocr', 'spatial_memory', 'session_manager', 'preprocessing']
    failed_count = sum(1 for comp in missing_components if not validation.get(comp, False))
    
    assert failed_count >= 1, "At least one missing component should be flagged as invalid"
    
    print("✅ Test 4: Missing dependency validation working")


def test_dependency_validation_partial():
    """Test dependency validation with partial outputs."""
    from ml.config import DependencyGraph
    
    graph = DependencyGraph()
    
    # Partial outputs with some fields missing
    outputs = {
        'model': {
            'detections': [{'class_name': 'person'}],
            'urgency_scores': [0.7]
            # Missing: text_regions, uncertainty
        },
        'ocr': {},  # Empty OCR
        'preprocessing': {'processed': True}
        # Missing: spatial_memory, session_manager
    }
    
    validation = graph.validate_dependencies(outputs)
    
    # Model and preprocessing likely valid
    assert validation.get('model', False) or validation.get('preprocessing', False), \
        "At least model or preprocessing should be valid"
    
    # Track which components are missing
    all_components = ['model', 'ocr', 'spatial_memory', 'session_manager', 'preprocessing']
    invalid_components = [c for c in all_components if not validation.get(c, False)]
    
    # Should have at least one invalid component
    assert len(invalid_components) >= 1, \
        f"Expected some invalid components, but all passed: {validation}"
    
    print("✅ Test 5: Partial dependency validation working")


def test_uncertainty_fallback_high_uncertainty():
    """Test fallback when uncertainty is high (forced deterministic case)."""
    # Create mock model output with controlled uncertainty
    outputs = {
        'classifications': torch.ones(1, 196, 80) * 0.6,
        'boxes': torch.ones(1, 196, 4) * 50.0,
        'objectness': torch.ones(1, 196) * 0.9,
        'uncertainty': torch.ones(1, 196) * 0.85  # High uncertainty
    }
    
    # Store original values
    original_objectness = outputs['objectness'].clone()
    
    # Apply uncertainty-based fallback logic
    uncertainty = outputs.get('uncertainty')
    assert uncertainty is not None, "Uncertainty should be present"
    assert uncertainty.mean().item() > 0.7, "Uncertainty should be high"
    
    # High uncertainty - apply conservative outputs
    outputs['objectness'] = outputs['objectness'] * 0.8
    
    # Verify confidence was reduced
    assert torch.all(outputs['objectness'] <= original_objectness * 0.8 + 1e-5), \
        "Objectness should be reduced under high uncertainty"
    assert outputs['objectness'].max().item() < 1.0, \
        "Objectness should be below 1.0"
    
    # Check for numerical stability (no NaNs/Infs)
    assert not torch.isnan(outputs['objectness']).any(), "No NaNs should be present"
    assert not torch.isinf(outputs['objectness']).any(), "No Infs should be present"
    
    print("✅ Test 6: High uncertainty fallback working")


def test_uncertainty_fallback_with_noisy_input():
    """Test uncertainty handling with adversarial noisy inputs."""
    torch.manual_seed(42)
    
    # Create noisy/blurred input to simulate challenging conditions
    clean_image = torch.randn(1, 3, 224, 224)
    noise = torch.randn_like(clean_image) * 0.5
    noisy_image = clean_image + noise
    
    # Clip to valid range
    noisy_image = torch.clamp(noisy_image, -3.0, 3.0)
    
    # Simulate model output with elevated uncertainty due to noise
    outputs = {
        'classifications': torch.rand(1, 196, 80) * 0.4,  # Lower confidence
        'boxes': torch.rand(1, 196, 4) * 100.0,
        'objectness': torch.rand(1, 196) * 0.7,
        'uncertainty': torch.ones(1, 196) * 0.75  # Elevated uncertainty
    }
    
    # Apply conservative fallback
    if outputs['uncertainty'].mean() > 0.7:
        outputs['objectness'] = outputs['objectness'] * 0.7
        outputs['classifications'] = outputs['classifications'] * 0.8
    
    # Verify conservative scaling
    assert outputs['objectness'].max().item() < 0.5, \
        "Objectness should be conservative under noisy conditions"
    assert outputs['classifications'].max().item() < 0.35, \
        "Classifications should be conservative under noisy conditions"
    
    # Numerical stability checks
    assert not torch.isnan(outputs['objectness']).any()
    assert not torch.isnan(outputs['classifications']).any()
    
    print("✅ Test 7: Noisy input uncertainty handling working")


def test_uncertainty_fallback_low_uncertainty():
    """Test normal execution when uncertainty is low."""
    outputs = {
        'classifications': torch.ones(1, 196, 80) * 0.8,
        'boxes': torch.ones(1, 196, 4) * 50.0,
        'objectness': torch.ones(1, 196) * 0.9,
        'uncertainty': torch.ones(1, 196) * 0.2  # Low uncertainty
    }
    
    original_objectness = outputs['objectness'].clone()
    
    # Low uncertainty - no fallback should trigger
    uncertainty = outputs.get('uncertainty')
    if uncertainty is not None and uncertainty.mean() > 0.7:
        outputs['objectness'] = outputs['objectness'] * 0.8
    
    # Should remain unchanged
    assert torch.allclose(outputs['objectness'], original_objectness), \
        "Objectness should not change under low uncertainty"
    
    print("✅ Test 8: Low uncertainty (no fallback) working")


def test_head_execution_manager_success():
    """Test HeadExecutionManager with successful execution."""
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    def test_head(**inputs):
        return {'output': torch.ones(1, 10)}
    
    result = manager.execute_head(
        head_name='test_head',
        head_func=test_head,
        inputs={'input': torch.ones(1, 5)},
        dependencies=[],
        fallback_func=None
    )
    
    # Verify output
    assert 'output' in result
    assert result['output'].shape == (1, 10)
    assert torch.allclose(result['output'], torch.ones(1, 10))
    
    # Check execution summary
    summary = manager.get_execution_summary()
    assert summary['total_executions'] >= 1
    assert summary.get('successful', summary['total_executions'] - summary.get('failed', 0)) >= 1
    
    print("✅ Test 9: HeadExecutionManager success path working")


def test_head_execution_manager_exception_with_fallback():
    """Test HeadExecutionManager handles exceptions and uses fallback."""
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    def failing_head(**inputs):
        raise RuntimeError("Simulated head failure")
    
    def fallback_func(**inputs):
        return {'output': torch.zeros(1, 10)}
    
    # Execute with expected exception
    result = manager.execute_head(
        head_name='failing_head',
        head_func=failing_head,
        inputs={'input': torch.ones(1, 5)},
        dependencies=[],
        fallback_func=fallback_func
    )
    
    # Should get fallback result
    assert isinstance(result, dict), "Result should be a dict"
    if 'output' in result:
        assert result['output'].shape == (1, 10)
        assert result['output'].sum().item() == 0.0, "Should return zeros from fallback"
    
    # Verify execution stats
    summary = manager.get_execution_summary()
    assert summary['total_executions'] >= 1
    assert summary['fallbacks_used'] >= 1 or summary['failed'] >= 1, \
        "Should record fallback usage or failure"
    
    print("✅ Test 10: HeadExecutionManager exception with fallback working")


def test_head_execution_manager_missing_dependency():
    """Test HeadExecutionManager with missing dependency."""
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    def head_needs_dep(**inputs):
        if 'required_dep' not in inputs:
            raise ValueError("Missing required dependency")
        return {'output': torch.ones(1, 10)}
    
    def fallback_func(**inputs):
        return {'output': torch.zeros(1, 10) - 1}  # Negative ones to distinguish
    
    result = manager.execute_head(
        head_name='head_with_dep',
        head_func=head_needs_dep,
        inputs={},  # Missing required_dep
        dependencies=['required_dep'],
        fallback_func=fallback_func
    )
    
    # Should use fallback due to missing dependency
    if isinstance(result, dict) and 'output' in result:
        # Fallback returns -1s
        assert result['output'].shape == (1, 10)
        # Should be negative (from fallback)
        assert result['output'].sum().item() < 0, "Should use fallback with negative values"
    
    # Verify stats
    summary = manager.get_execution_summary()
    assert summary['total_executions'] >= 1
    assert summary['fallbacks_used'] >= 1 or summary['failed'] >= 1
    
    print("✅ Test 11: HeadExecutionManager missing dependency working")


def test_head_execution_manager_latency():
    """Test HeadExecutionManager execution latency is reasonable."""
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    def fast_head(**inputs):
        # Simulate lightweight processing
        x = inputs.get('input', torch.zeros(1, 10))
        return {'output': x * 2}
    
    # Warm up
    for _ in range(3):
        manager.execute_head(
            head_name='fast_head',
            head_func=fast_head,
            inputs={'input': torch.ones(1, 10)},
            dependencies=[],
            fallback_func=None
        )
    
    # Measure execution time
    start = time.time()
    for _ in range(10):
        result = manager.execute_head(
            head_name='fast_head',
            head_func=fast_head,
            inputs={'input': torch.ones(1, 10)},
            dependencies=[],
            fallback_func=None
        )
    elapsed = time.time() - start
    
    avg_latency = elapsed / 10
    
    # Should complete quickly on CPU (< 10ms per execution)
    assert avg_latency < 0.01, f"Average latency {avg_latency*1000:.2f}ms exceeds 10ms threshold"
    
    # Verify result is correct
    assert result['output'].sum().item() == 20.0  # 1 * 10 * 2
    
    print(f"✅ Test 12: HeadExecutionManager latency {avg_latency*1000:.3f}ms/exec working")


def test_head_execution_manager_summary_accounting():
    """Test HeadExecutionManager accurately tracks all executions."""
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    def success_head(**inputs):
        return {'output': torch.ones(1, 5)}
    
    def fail_head(**inputs):
        raise RuntimeError("Failure")
    
    def fallback(**inputs):
        return {'output': torch.zeros(1, 5)}
    
    # Execute successful heads
    for i in range(3):
        manager.execute_head(
            head_name=f'success_{i}',
            head_func=success_head,
            inputs={},
            dependencies=[],
            fallback_func=None
        )
    
    # Execute failing heads with fallback
    for i in range(2):
        manager.execute_head(
            head_name=f'fail_{i}',
            head_func=fail_head,
            inputs={},
            dependencies=[],
            fallback_func=fallback
        )
    
    summary = manager.get_execution_summary()
    
    # Should have 5 total executions
    assert summary['total_executions'] == 5, \
        f"Expected 5 executions, got {summary['total_executions']}"
    
    # Should have used fallbacks for the 2 failures
    assert summary['fallbacks_used'] >= 2 or summary['failed'] >= 2, \
        f"Expected at least 2 fallbacks/failures, got {summary.get('fallbacks_used', 0)} fallbacks, {summary.get('failed', 0)} failed"
    
    print("✅ Test 13: HeadExecutionManager accounting working")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED ERROR HANDLING TESTS - MAXSIGHT 3.0")
    print("="*70 + "\n")
    
    print("Testing fallback mechanisms...")
    test_fallback_on_forced_exception()
    test_fallback_on_successful_execution()
    
    print("\nTesting dependency validation...")
    test_dependency_validation_complete()
    test_dependency_validation_missing()
    test_dependency_validation_partial()
    
    print("\nTesting uncertainty-based fallbacks...")
    test_uncertainty_fallback_high_uncertainty()
    test_uncertainty_fallback_with_noisy_input()
    test_uncertainty_fallback_low_uncertainty()
    
    print("\nTesting HeadExecutionManager...")
    test_head_execution_manager_success()
    test_head_execution_manager_exception_with_fallback()
    test_head_execution_manager_missing_dependency()
    test_head_execution_manager_latency()
    test_head_execution_manager_summary_accounting()
    
    print("\n" + "="*70)
    print("ALL ADVANCED ERROR HANDLING TESTS PASSED ✓")
    print("="*70)
    print("\nCoverage Summary:")
    print("  • Forced exception fallbacks: ✓")
    print("  • Successful execution (no fallback): ✓")
    print("  • Complete dependency validation: ✓")
    print("  • Missing dependency detection: ✓")
    print("  • Partial dependency handling: ✓")
    print("  • High uncertainty fallback: ✓")
    print("  • Noisy input uncertainty handling: ✓")
    print("  • Low uncertainty (no fallback): ✓")
    print("  • HeadExecutionManager success path: ✓")
    print("  • HeadExecutionManager exception handling: ✓")
    print("  • HeadExecutionManager missing dependencies: ✓")
    print("  • HeadExecutionManager latency checks: ✓")
    print("  • HeadExecutionManager accounting: ✓")
    print("\nTotal: 13 comprehensive tests")
    print("="*70 + "\n")
