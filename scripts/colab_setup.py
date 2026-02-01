#!/usr/bin/env python3
"""
Colab-specific setup script.
Run this in a Colab notebook cell to set up MaxSight training environment.
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run shell command and print output."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result

def main():
    print("=" * 60)
    print("🚀 MaxSight Colab Setup")
    print("=" * 60)
    print()
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Colab, but continuing anyway...")
    
    # Step 1: Install git if needed
    print("\n📦 Step 1: Installing git...")
    run_command("apt-get update", check=False)
    run_command("apt-get install -y git", check=False)
    
    # Step 2: Clone repository
    print("\n📥 Step 2: Cloning repository...")
    if not os.path.exists("2026-Prototype"):
        run_command("git clone https://github.com/AstroSword2897/2026-Prototype.git")
    else:
        print("✅ Repository already exists, skipping clone")
    
    # Step 3: Checkout correct branch
    print("\n🔀 Step 3: Checking out branch...")
    os.chdir("2026-Prototype")
    run_command("git checkout feature/multimodal_refactor", check=False)
    
    # Step 4: Install PyTorch with CUDA (if not already installed)
    print("\n🔥 Step 4: Installing PyTorch with CUDA...")
    import torch
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Installing PyTorch with CUDA support...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("✅ PyTorch with CUDA already installed")
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Step 5: Install dependencies
    print("\n📚 Step 5: Installing dependencies...")
    run_command("pip install -r requirements.txt")
    run_command("pip install faiss-cpu")
    
    # Step 6: Verify setup
    print("\n✅ Step 6: Verifying setup...")
    import torch
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Step 7: Test import
    print("\n🧪 Step 7: Testing imports...")
    sys.path.insert(0, os.getcwd())
    try:
        from ml.models.maxsight_cnn import MaxSightCNN, CapabilityTier
        print("✅ MaxSight imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run test script:")
    print("     !python /content/2026-Prototype/scripts/test_gpu_setup.py")
    print("  2. Start smoke training:")
    print("     !cd /content/2026-Prototype && python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda")
    print("  3. Start full training:")
    print("     !cd /content/2026-Prototype && python scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda")
    print()

if __name__ == "__main__":
    main()

