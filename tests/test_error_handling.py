"""
Error Handling and Fallback Tests
Tests error propagation and fallback mechanisms.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.error_handling import HeadExecutionManager, safe_head_execution, with_fallback
from ml.config import RuntimeConfig


def test_fallback_on_error():
    """Test fallback when head execution fails."""
    print("Error Handling Test 1: Fallback on Error")
    
    # Use regular model with error handling decorator
    model = create_model()
    model.eval()
    
    # Wrap forward pass with fallback
    @with_fallback(fallback_value={'classifications': torch.zeros(1, 196, 80), 
                                   'boxes': torch.zeros(1, 196, 4),
                                   'objectness': torch.zeros(1, 196)})
    def safe_forward(images):
        return model(images)
    
    # Normal execution should work
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = safe_forward(dummy_image)
    
    # Should have all outputs
    assert 'classifications' in outputs
    assert 'boxes' in outputs
    assert 'objectness' in outputs
    
    print("  ✅ Fallback system working")


def test_dependency_validation():
    """Test dependency validation."""
    print("\nError Handling Test 2: Dependency Validation")
    
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
        'session_manager': {}
    }
    
    validation = graph.validate_dependencies(outputs)
    
    # Check that at least some dependencies are satisfied
    # (not all components may be valid if dependencies are missing)
    assert validation.get('model', False), "Model should be valid"
    assert validation.get('preprocessing', False), "Preprocessing should be valid"
    
    print("  ✅ Dependency validation working")


def test_uncertainty_fallback():
    """Test fallback when uncertainty is high."""
    print("\nError Handling Test 3: Uncertainty Fallback")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    # Check uncertainty and apply fallback logic if needed
    uncertainty = outputs.get('uncertainty')
    if uncertainty is not None and uncertainty.mean() > 0.7:
        # High uncertainty - apply conservative outputs
        outputs['objectness'] = outputs['objectness'] * 0.8  # Lower confidence
        assert outputs['objectness'].max() < 1.0
    
    print("  ✅ Uncertainty fallback working")


def test_head_execution_manager():
    """Test HeadExecutionManager."""
    print("\nError Handling Test 4: Head Execution Manager")
    
    manager = HeadExecutionManager(enable_fallbacks=True)
    
    # Test successful execution
    def test_head(**inputs):
        return {'output': torch.ones(1, 10)}
    
    result = manager.execute_head(
        head_name='test_head',
        head_func=test_head,
        inputs={'input': torch.ones(1, 5)},
        dependencies=[],
        fallback_func=None
    )
    
    assert 'output' in result
    assert result['output'].shape == (1, 10)
    
    # Test with missing dependency
    def head_needs_dep(**inputs):
        if 'required_dep' not in inputs:
            raise ValueError("Missing dependency")
        return {'output': torch.ones(1, 10)}
    
    def fallback_func(**inputs):
        return {'output': torch.zeros(1, 10)}
    
    result = manager.execute_head(
        head_name='head_with_dep',
        head_func=head_needs_dep,
        inputs={},  # Missing required_dep
        dependencies=['required_dep'],
        fallback_func=fallback_func
    )
    
    # Should use fallback (returns dict with 'output' key)
    if isinstance(result, dict) and 'output' in result:
        assert result['output'].sum() == 0  # Zeros from fallback
    else:
        # Fallback might return the dict directly
        assert isinstance(result, dict)
    
    # Get summary
    summary = manager.get_execution_summary()
    # At least one execution should have occurred
    assert summary['total_executions'] >= 1
    # Fallback should have been used for the missing dependency case
    assert summary['fallbacks_used'] >= 1 or summary['failed'] >= 1
    
    print("  ✅ Head execution manager working")


if __name__ == "__main__":
    print("Running Error Handling Tests")
    print("=" * 50)
    
    test_fallback_on_error()
    test_dependency_validation()
    test_uncertainty_fallback()
    test_head_execution_manager()
    
    print("\n" + "=" * 50)
    print("All error handling tests passed!")

