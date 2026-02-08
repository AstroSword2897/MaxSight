#!/usr/bin/env python3
"""Quick test script to verify GPU setup for MaxSight training.
Run this after setting up cloud GPU to ensure everything works."""

import sys
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ml.models.maxsight_cnn import MaxSightCNN, CapabilityTier


def test_cuda():
    """Test CUDA availability and device info."""
    print("=" * 60)
    print("CUDA Test")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"OK CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"OK CUDA Devices: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n📱 Device {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
        
        print(f"\nOK CUDA Version: {torch.version.cuda}")
        print(f"OK cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("FAIL CUDA not available. Make sure:")
        print("   1. GPU runtime is enabled (Colab: Runtime → Change runtime type → GPU)")
        print("   2. PyTorch with CUDA is installed")
        print("   3. You're using a GPU instance (AWS/Paperspace)")
    
    return cuda_available


def test_model_loading():
    """Test loading MaxSight model."""
    print("\n" + "=" * 60)
    print("📦 Model Loading Test")
    print("=" * 60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        print("\nLoading T0_BASELINE_CNN...")
        model = MaxSightCNN(
            tier=CapabilityTier.T0_BASELINE_CNN,
            device=device
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"OK Model loaded: {param_count:,} parameters")
        
        # Move to GPU if available.
        if device == 'cuda':
            model = model.cuda()
            print("OK Model moved to GPU")
        
        return model, device
        
    except Exception as e:
        print(f"FAIL Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, device):
    """Test forward pass through model."""
    print("\n" + "=" * 60)
    print("🧪 Forward Pass Test")
    print("=" * 60)
    
    if model is None:
        print("FAIL Model not loaded, skipping forward pass test")
        return False
    
    try:
        # Create dummy input.
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        if device == 'cuda':
            dummy_input = dummy_input.cuda()
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input device: {dummy_input.device}")
        
        # Forward pass.
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        
        print("OK Forward pass successful!")
        print(f"\nOutput keys: {list(output.keys())}")
        
        # Print output shapes.
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory():
    """Test GPU memory availability."""
    print("\n" + "=" * 60)
    print("💾 Memory Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("WARNING  CUDA not available, skipping memory test")
        return
    
    try:
        # Get memory info.
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        
        print(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")
        print(f"Allocated: {allocated / 1e9:.2f} GB ({allocated / total_memory * 100:.1f}%)")
        print(f"Reserved: {reserved / 1e9:.2f} GB ({reserved / total_memory * 100:.1f}%)")
        print(f"Free: {(total_memory - reserved) / 1e9:.2f} GB")
        
        # Check if enough memory for training.
        free_gb = (total_memory - reserved) / 1e9
        if free_gb < 8:
            print("WARNING  Warning: Less than 8GB free. May need to reduce batch size.")
        else:
            print("OK Sufficient memory for training")
        
    except Exception as e:
        print(f"FAIL Error checking memory: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 MaxSight GPU Setup Test")
    print("=" * 60)
    print()
    
    # Test CUDA.
    cuda_available = test_cuda()
    
    # Test model loading.
    model, device = test_model_loading()
    
    # Test forward pass.
    if model is not None:
        forward_success = test_forward_pass(model, device)
    else:
        forward_success = False
    
    # Test memory.
    test_memory()
    
    # Summary.
    print("\n" + "=" * 60)
    print(" Test Summary")
    print("=" * 60)
    
    all_passed = cuda_available and (model is not None) and forward_success
    
    if all_passed:
        print("OK All tests passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Run smoke training:")
        print("     python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda")
        print("  2. Start full training:")
        print("     python scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda")
    else:
        print("FAIL Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        if not cuda_available:
            print("  - CUDA not available: Enable GPU runtime or use GPU instance")
        if model is None:
            print("  - Model loading failed: Check dependencies and imports")
        if not forward_success:
            print("  - Forward pass failed: Check model configuration")
    
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


