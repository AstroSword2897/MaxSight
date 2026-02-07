"""
Profile integrated MaxSightCNN forward pass.
"""
import torch
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.models.maxsight_cnn import MaxSightCNN


def profile_forward():
    """Profile the forward pass and verify performance constraints."""
    print("Initializing MaxSightCNN...")
    try:
        model = MaxSightCNN()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"⚠️  Warning: Missing optional dependencies: {e}")
        print("   Some features may be unavailable, but core functionality should work.")
        print("   To install: pip install scikit-learn transformers")
        # Continue with a minimal model when possible
        raise
    
    model.eval()
    
    # Uses CPU for consistent profiling (or MPS if available)
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS device: {device}")
    else:
        print(f"Using CPU device: {device}")
    
    model = model.to(device)
    
    images = torch.randn(2, 3, 224, 224).to(device)
    audio_features = torch.randn(2, 128).to(device)
    
    timings = []
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(images, audio_features)
    
    # Profile
    print("Profiling forward pass (50 iterations)...")
    with torch.no_grad():
        for i in range(50):
            t0 = time.perf_counter()
            outputs = model(images, audio_features)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # Convert to ms
    
    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    
    print(f"\n{'='*60}")
    print(f"Forward Pass Performance:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"{'='*60}\n")
    
    # Verify outputs
    print("Verifying outputs...")
    required_keys = [
        'depth_map',
        'depth_uncertainty',
        'sound_classifications',
        'scene_description',
        'classifications',
        'boxes',
        'objectness'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in outputs:
            missing_keys.append(key)
        elif outputs[key] is None:
            print(f"  ⚠️  {key}: None (optional)")
        else:
            print(f"  ✅ {key}: {outputs[key].shape if hasattr(outputs[key], 'shape') else type(outputs[key])}")
    
    if missing_keys:
        print(f"\n❌ Missing required keys: {missing_keys}")
        return False
    
    # Verifies performance constraint
    if avg_time < 85.0:
        print(f"\n✅ Performance constraint satisfied: {avg_time:.2f}ms < 85ms")
        return True
    else:
        print(f"\n❌ Performance constraint violated: {avg_time:.2f}ms >= 85ms")
        return False


if __name__ == '__main__':
    success = profile_forward()
    sys.exit(0 if success else 1)

