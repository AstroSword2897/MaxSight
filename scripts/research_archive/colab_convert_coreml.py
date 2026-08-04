#!/usr/bin/env python3
"""Colab-oriented script to convert a single .pt checkpoint to CoreML format.

Valid Python (no IPython magics). Safe to run under flake8 / CI syntax gates.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "coremltools"])

import torch  # noqa: E402

# Mount Google Drive when running inside Colab.
try:
    from google.colab import drive

    drive.mount("/content/drive")
    print("Google Drive mounted")
except Exception as e:
    print(f"Drive mount: {e}")

REPO_PATH = "/content/2026-Prototype"
repo = Path(REPO_PATH)
if not repo.exists():
    print("Cloning repository...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "https://github.com/AstroSword2897/2026-Prototype.git",
            REPO_PATH,
        ]
    )
else:
    print("Repository already exists, pulling latest...")
    subprocess.check_call(["git", "-C", REPO_PATH, "pull"])

sys.path.insert(0, REPO_PATH)

CONDITION = "color_blindness"
DRIVE_CHECKPOINT_PATH = f"/content/drive/MyDrive/MaxSight/checkpoints_{CONDITION}/best_model.pt"
LOCAL_CHECKPOINT_PATH = f"{REPO_PATH}/checkpoints/checkpoints_{CONDITION}/best_model.pt"
OUTPUT_PATH = f"/content/drive/MyDrive/MaxSight/checkpoints_{CONDITION}/best_model.mlpackage"

if Path(DRIVE_CHECKPOINT_PATH).exists():
    print(f"Found checkpoint in Drive: {DRIVE_CHECKPOINT_PATH}")
    checkpoint_path = DRIVE_CHECKPOINT_PATH
elif Path(LOCAL_CHECKPOINT_PATH).exists():
    print(f"Found checkpoint locally: {LOCAL_CHECKPOINT_PATH}")
    checkpoint_path = LOCAL_CHECKPOINT_PATH
else:
    print("Checkpoint not found!")
    print(f"   Looked in: {DRIVE_CHECKPOINT_PATH}")
    print(f"   Looked in: {LOCAL_CHECKPOINT_PATH}")
    raise FileNotFoundError("Checkpoint not found")

from ml.models.maxsight_cnn import (  # noqa: E402
    COCO_CLASSES,
    CapabilityTier,
    TierConfig,
    create_model,
)
from ml.training.export import export_to_coreml  # noqa: E402

print(f"\nCreating model architecture for condition: {CONDITION}")
tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
model = create_model(
    num_classes=len(COCO_CLASSES),
    use_audio=False,
    condition_mode=CONDITION,
    tier_config=tier_config,
)

print(f"\nLoading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
model.cpu()

print("\nModel loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTesting inference...")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"   Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"   Output keys: {list(output.keys())[:5]}...")

print(f"\nExporting to CoreML: {OUTPUT_PATH}")
result = export_to_coreml(
    model,
    save_path=OUTPUT_PATH,
    input_size=(1, 3, 224, 224),
    device="cpu",
    validate=True,
)

if result:
    print("\nConversion complete!")
    print(f"   Saved to: {result}")
    size_mb = Path(result).stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")

    print("\nVerifying CoreML model...")
    try:
        import coremltools as ct

        coreml_model = ct.models.MLModel(str(result))
        print("   Model loads successfully")
        print(f"   Input: {coreml_model.input_description}")
        print(f"   Output: {coreml_model.output_description}")
    except Exception as e:
        print(f"   Verification warning: {e}")

    print("\nDownload the file from Google Drive:")
    print(f"   {OUTPUT_PATH}")
    try:
        from google.colab import files

        files.download(str(result))
    except Exception as e:
        print(f"   Colab download unavailable: {e}")
else:
    print("\nConversion failed!")
    print("Check error messages above for details")
    raise SystemExit(1)
