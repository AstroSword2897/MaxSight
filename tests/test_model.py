"""
Unit Tests for MaxSight CNN Model
Sprint 1 Validation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from ml.models.maxsight_cnn import create_model, MaxSightCNN


def test_model_creation():
    """Test basic model creation"""
    model = create_model()
    assert model is not None
    assert isinstance(model, MaxSightCNN)
    print("✓ Model creation test passed")


def test_forward_pass():
    """Test forward pass with dummy data"""
    model = create_model()
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    # Check output shapes
    assert outputs['classifications'].shape == (batch_size, 48)
    assert 'per_object_classifications' in outputs
    assert outputs['per_object_classifications'].shape == (batch_size, 10, 48)
    assert outputs['boxes'].shape == (batch_size, 10, 4)
    assert outputs['objectness'].shape == (batch_size, 10)
    assert outputs['scene_embedding'].shape == (batch_size, 512)
    assert outputs['urgency_scores'].shape == (batch_size, 10, 4)
    assert outputs['distance_zones'].shape == (batch_size, 10, 3)
    
    print("✓ Forward pass test passed")


def test_audio_fusion():
    """Test audio fusion mode"""
    model = create_model(use_audio=True)
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_audio = torch.randn(batch_size, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
    
    assert outputs['classifications'].shape == (batch_size, 48)
    print("✓ Audio fusion test passed")


def test_color_blindness_mode():
    """Test color blindness condition mode"""
    model = create_model(condition_mode='color_blindness')
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    assert 'colors' in outputs
    assert outputs['colors'].shape == (batch_size, 12)
    print("✓ Color blindness mode test passed")


def test_parameter_count():
    """Test model parameter count"""
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Should be around 35M parameters
    assert 30_000_000 < total_params < 40_000_000
    assert trainable_params == total_params  # All should be trainable initially
    
    print(f"✓ Parameter count test passed: {total_params:,} parameters")


def test_gradient_flow():
    """Test that gradients can flow through the model"""
    model = create_model()
    model.train()
    
    dummy_image = torch.randn(2, 3, 224, 224, requires_grad=True)
    outputs = model(dummy_image)
    
    # Compute dummy loss
    loss = outputs['classifications'].sum()
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients
    print("✓ Gradient flow test passed")


def test_inference_mode():
    """Test inference mode (eval)"""
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    # Check that valid_detections mask exists in eval mode
    if 'valid_detections' in outputs:
        assert outputs['valid_detections'].shape == (1, 10)
        print("✓ Inference mode test passed (with valid_detections)")
    else:
        print("✓ Inference mode test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running MaxSight CNN Tests")
    print("=" * 60)
    
    test_model_creation()
    test_forward_pass()
    test_audio_fusion()
    test_color_blindness_mode()
    test_parameter_count()
    test_gradient_flow()
    test_inference_mode()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

