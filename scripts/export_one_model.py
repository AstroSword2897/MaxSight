#!/usr/bin/env python3
"""Minimal script: load one checkpoint and export to JIT. Prints full traceback on any error.

Use this when deploy keeps failing and you need to see the real error.

  # From repo root (e.g. on Colab after git pull):
  python scripts/export_one_model.py --checkpoint /path/to/checkpoints_amblyopia/best_model.pt --out maxsight.pt

  # Or with condition name (auto-finds checkpoint under --checkpoints-base):
  python scripts/export_one_model.py --condition amblyopia --checkpoints-base /content/drive/MyDrive/MaxSight --out /tmp/amblyopia.pt
"""

import argparse
import sys
import traceback
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description="Load one model and export to JIT; print full traceback on error.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best_model.pt")
    parser.add_argument("--condition", type=str, default=None, help="Condition name (e.g. amblyopia); used with --checkpoints-base")
    parser.add_argument("--checkpoints-base", type=Path, default=None, help="Base dir; with --condition uses <base>/checkpoints_<cond>/best_model.pt")
    parser.add_argument("--out", type=Path, default=Path("maxsight_traced.pt"), help="Output .pt path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for load/forward (export uses cpu)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None and args.condition and args.checkpoints_base:
        ckpt_path = Path(args.checkpoints_base).resolve() / f"checkpoints_{args.condition}" / "best_model.pt"
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Use --checkpoint /path/to/best_model.pt or --condition NAME --checkpoints-base /path", file=sys.stderr)
        return 1

    ckpt_path = Path(ckpt_path).resolve()
    out_path = Path(args.out).resolve()
    device = args.device

    try:
        print("Step 1: Importing model and export...", flush=True)
        from ml.models.maxsight_cnn import (
            COCO_CLASSES,
            CapabilityTier,
            TierConfig,
            create_model,
        )
        from ml.training.export import export_to_jit

        cond = args.condition or ckpt_path.parent.name.replace("checkpoints_", "")
        print(f"Step 2: Creating model (condition={cond})...", flush=True)
        tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
        model = create_model(
            num_classes=len(COCO_CLASSES),
            use_audio=False,
            condition_mode=cond,
            tier_config=tier_config,
        )

        print(f"Step 3: Loading weights from {ckpt_path}...", flush=True)
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        print("Step 4: One forward pass...", flush=True)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            out = model(dummy)
        if not isinstance(out, dict) or "objectness" not in out:
            print("Forward pass failed: output missing 'objectness'", file=sys.stderr)
            return 1
        print("  Forward OK.", flush=True)

        print("Step 5: Exporting to JIT...", flush=True)
        model.cpu()
        export_to_jit(
            model,
            save_path=str(out_path),
            input_size=(1, 3, 224, 224),
            device="cpu",
            validate=False,
        )
        print(f"Done. Saved: {out_path}", flush=True)
        return 0

    except Exception as e:
        print("\n" + "=" * 60, flush=True)
        print("ERROR (full traceback below)", flush=True)
        print("=" * 60, flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        print(f"Exception: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
