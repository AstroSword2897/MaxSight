#!/usr/bin/env python3
"""Verify a CoreML .mlpackage file is valid and can be loaded."""

import sys
from pathlib import Path

def verify_mlpackage(mlpackage_path: Path):
    """Check if .mlpackage file exists and can be loaded."""
    print(f"Checking: {mlpackage_path}")
    
    # Check if file/directory exists
    if not mlpackage_path.exists():
        print(f"❌ File not found: {mlpackage_path}")
        return False
    
    # Check if it's a directory (mlpackage is a directory)
    if not mlpackage_path.is_dir():
        print(f"❌ Not a directory (mlpackage should be a directory): {mlpackage_path}")
        return False
    
    # Check for required files inside
    required_files = ["model.mlmodel", "metadata.json"]
    missing = []
    for req_file in required_files:
        if not (mlpackage_path / req_file).exists():
            missing.append(req_file)
    
    if missing:
        print(f"⚠️  Missing files: {', '.join(missing)}")
        print(f"   This might be an incomplete conversion")
        return False
    
    # Try to load with coremltools
    try:
        import coremltools as ct
        print("Loading model with coremltools...")
        model = ct.models.MLModel(str(mlpackage_path))
        
        # Get model description
        print("\n✅ Model loaded successfully!")
        print(f"\nModel Info:")
        print(f"  Input: {model.input_description}")
        print(f"  Output: {model.output_description}")
        print(f"  Size: {sum(f.stat().st_size for f in mlpackage_path.rglob('*') if f.is_file()) / (1024*1024):.1f} MB")
        
        return True
        
    except ImportError:
        print("⚠️  coremltools not installed - cannot verify model loading")
        print("   Install with: pip install coremltools")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_coreml.py <path/to/model.mlpackage>")
        print("\nExample:")
        print("  python scripts/verify_coreml.py checkpoints/checkpoints_color_blindness/best_model.mlpackage")
        return 1
    
    mlpackage_path = Path(sys.argv[1]).resolve()
    
    if verify_mlpackage(mlpackage_path):
        print("\n✅ Verification passed - model is ready for Xcode!")
        return 0
    else:
        print("\n❌ Verification failed - model may be incomplete or corrupted")
        return 1

if __name__ == "__main__":
    sys.exit(main())
