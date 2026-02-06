#!/usr/bin/env python3
"""Export MaxSight model for Xcode integration...."""
import sys
from pathlib import Path
import torch
import logging

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.maxsight_cnn import create_model
from ml.training.export import export_ios_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_model.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "maxsight_ios_bundle"
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pt"):
                logger.info(f"  - {ckpt}")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = create_model()
    
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=False)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)
    
    model.eval()
    
    logger.info(f"Exporting iOS bundle to: {output_dir}")
    try:
        bundle_path = export_ios_bundle(
            model=model,
            output_dir=output_dir,
            input_size=(1, 3, 224, 224)
        )
        
        logger.info(f"✅ Export complete!")
        logger.info(f"\nBundle location: {bundle_path}")
        logger.info(f"\nFiles created:")
        for file in sorted(bundle_path.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
        
        logger.info(f"\nNext steps:")
        logger.info(f"1. Copy {bundle_path} to your Xcode project")
        logger.info(f"2. Follow instructions in {bundle_path}/README_XCODE.md")
        logger.info(f"3. Port preprocessing from {bundle_path}/processing_reference.py to Swift")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
