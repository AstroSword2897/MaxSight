#!/usr/bin/env python3
"""Load a .pt checkpoint and export to CoreML (.mlpackage) for Xcode/iOS."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch .pt checkpoint to CoreML .mlpackage.")
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt (or other .pt checkpoint)")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output path (default: same dir as checkpoint, name with .mlpackage)")
    parser.add_argument("--condition", type=str, default=None, help="Condition name if not inferrable from path (e.g. amblyopia)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device for loading (export uses CPU)")
    parser.add_argument("--no-validate", action="store_true", help="Skip CoreML validation after convert")
    args = parser.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        out_path = ckpt_path.parent / (ckpt_path.stem + ".mlpackage")
    else:
        out_path = Path(out_path).resolve()

    cond = args.condition
    if cond is None and "checkpoints_" in str(ckpt_path):
        try:
            cond = ckpt_path.parent.name.replace("checkpoints_", "")
        except Exception:
            cond = "amblyopia"
    if cond is None:
        cond = "amblyopia"

    print("Loading model and checkpoint...", flush=True)
    from ml.models.maxsight_cnn import (
        COCO_CLASSES,
        CapabilityTier,
        TierConfig,
        create_model,
    )
    from ml.training.export import export_to_coreml
    import torch

    # Create model architecture (empty/untrained)
    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    model = create_model(
        num_classes=len(COCO_CLASSES),
        use_audio=False,
        condition_mode=cond,
        tier_config=tier_config,
    )

    # Load trained weights from .pt checkpoint file
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    # Extract state dict (handles both {"model_state_dict": {...}} and direct state dict formats)
    state = ckpt.get("model_state_dict", ckpt)
    # Apply trained weights to model
    model.load_state_dict(state, strict=False)
    model.eval()
    model.cpu()

    print(f"Exporting to CoreML: {out_path}", flush=True)
    result = export_to_coreml(
        model,
        save_path=str(out_path),
        input_size=(1, 3, 224, 224),
        device="cpu",
        validate=not args.no_validate,
    )
    if result is None:
        print("CoreML export failed. Install coremltools if needed: pip install coremltools", file=sys.stderr)
        return 1
    print(f"Done. Saved: {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
