#!/usr/bin/env python3
"""Colab script to convert a single .pt checkpoint to CoreML format.
Run this in a Google Colab notebook cell.
"""

# Install dependencies
print("Installing dependencies...")
!pip install -q coremltools

# Setup
import sys
from pathlib import Path
import torch

# Mount Google Drive (if not already mounted)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted")
except Exception as e:
    print(f"Drive mount: {e}")

# Setup repo
REPO_PATH = "/content/2026-Prototype"
if not Path(REPO_PATH).exists():
    print("Cloning repository...")
    !git clone https://github.com/AstroSword2897/2026-Prototype.git /content/2026-Prototype
else:
    print("Repository already exists, pulling latest...")
    %cd {REPO_PATH}
    !git pull

sys.path.insert(0, REPO_PATH)

# Configuration
CONDITION = "color_blindness"  # Change this to your condition
DRIVE_CHECKPOINT_PATH = f"/content/drive/MyDrive/MaxSight/checkpoints_{CONDITION}/best_model.pt"
LOCAL_CHECKPOINT_PATH = f"{REPO_PATH}/checkpoints/checkpoints_{CONDITION}/best_model.pt"
OUTPUT_PATH = f"/content/drive/MyDrive/MaxSight/checkpoints_{CONDITION}/best_model.mlpackage"

# Check if checkpoint exists in Drive
if Path(DRIVE_CHECKPOINT_PATH).exists():
    print(f"✅ Found checkpoint in Drive: {DRIVE_CHECKPOINT_PATH}")
    checkpoint_path = DRIVE_CHECKPOINT_PATH
elif Path(LOCAL_CHECKPOINT_PATH).exists():
    print(f"✅ Found checkpoint locally: {LOCAL_CHECKPOINT_PATH}")
    checkpoint_path = LOCAL_CHECKPOINT_PATH
else:
    print(f"❌ Checkpoint not found!")
    print(f"   Looked in: {DRIVE_CHECKPOINT_PATH}")
    print(f"   Looked in: {LOCAL_CHECKPOINT_PATH}")
    print("\nPlease upload your best_model.pt to one of these locations:")
    print(f"   - Google Drive: /content/drive/MyDrive/MaxSight/checkpoints_{CONDITION}/")
    print(f"   - Or update DRIVE_CHECKPOINT_PATH variable")
    raise FileNotFoundError(f"Checkpoint not found")

# Import model creation and export functions
from ml.models.maxsight_cnn import (
    COCO_CLASSES,
    CapabilityTier,
    TierConfig,
    create_model,
)
from ml.training.export import export_to_coreml

# Create model architecture
print(f"\n📦 Creating model architecture for condition: {CONDITION}")
tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
model = create_model(
    num_classes=len(COCO_CLASSES),
    use_audio=False,
    condition_mode=CONDITION,
    tier_config=tier_config,
)

# Load checkpoint
print(f"\n📥 Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
model.cpu()

# Verify model loads
print("\n✅ Model loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test inference
print("\n🧪 Testing inference...")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"   Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"   Output keys: {list(output.keys())[:5]}...")

# Export to CoreML
print(f"\n🔄 Exporting to CoreML: {OUTPUT_PATH}")
result = export_to_coreml(
    model,
    save_path=OUTPUT_PATH,
    input_size=(1, 3, 224, 224),
    device="cpu",
    validate=True,  # Colab environment is stable, so we can validate
)

if result:
    print(f"\n✅ Conversion complete!")
    print(f"   Saved to: {result}")
    
    # Check file size
    size_mb = Path(result).stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    
    # Verify model
    print("\n🔍 Verifying CoreML model...")
    try:
        import coremltools as ct
        coreml_model = ct.models.MLModel(str(result))
        print("   ✅ Model loads successfully")
        print(f"   Input: {coreml_model.input_description}")
        print(f"   Output: {coreml_model.output_description}")
    except Exception as e:
        print(f"   ⚠️  Verification warning: {e}")
    
    print(f"\n📥 Download the file from Google Drive:")
    print(f"   {OUTPUT_PATH}")
    print("\nOr download directly:")
    from google.colab import files
    files.download(str(result))
else:
    print("\n❌ Conversion failed!")
    print("Check error messages above for details")
