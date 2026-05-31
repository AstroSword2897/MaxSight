#!/usr/bin/env python3
"""Export only CoreML for the top 7 conditions (no JIT/PTE). Use when deploy_top7.py lacks --coreml-only or JIT crashes."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOP7 = [
    "amblyopia",
    "amd",
    "color_blindness",
    "cvi",
    "glaucoma",
    "retinitis_pigmentosa",
    "strabismus",
]


def main():
    p = argparse.ArgumentParser(description="Export 7 CoreML .mlpackage only (no JIT).")
    p.add_argument(
        "--checkpoints-base",
        type=Path,
        default=None,
        help="Base dir with checkpoints_<cond>/best_model.pt",
    )
    p.add_argument("--output-dir", type=Path, default=REPO / "exports" / "top7", help="Output root")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    base = (
        args.checkpoints_base
        or Path(os.environ.get("CHECKPOINTS_BASE", REPO / "checkpoints")).resolve()
    )
    out_root = Path(args.output_dir).resolve()
    verbose = not args.quiet

    if not base.exists():
        print(f"Checkpoints base not found: {base}", file=sys.stderr)
        print("Set CHECKPOINTS_BASE or pass --checkpoints-base", file=sys.stderr)
        return 1

    import torch
    from ml.models.maxsight_cnn import COCO_CLASSES, CapabilityTier, TierConfig, create_model
    from ml.training.export import export_to_coreml

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    num_classes = len(COCO_CLASSES)
    manifest = {"checkpoints_base": str(base), "output_dir": str(out_root), "conditions": {}}

    for cond in TOP7:
        ckpt_path = base / f"checkpoints_{cond}" / "best_model.pt"
        manifest["conditions"][cond] = {
            "checkpoint": str(ckpt_path),
            "exists": ckpt_path.exists(),
            "coreml_path": None,
            "error": None,
        }
        if not ckpt_path.exists():
            if verbose:
                print(f"  {cond}: missing checkpoint")
            continue

        if verbose:
            print(f"  {cond}: loading...", flush=True)
        try:
            model = create_model(
                num_classes=num_classes,
                use_audio=False,
                condition_mode=cond,
                tier_config=tier_config,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            model.cpu()
            if verbose:
                print("    Running 1-batch check...", flush=True)
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = model(dummy)
            if not isinstance(out, dict) or "objectness" not in out:
                manifest["conditions"][cond]["error"] = "inference check failed"
                continue
            if verbose:
                print("    Exporting CoreML...", flush=True)
            coreml_path = export_to_coreml(
                model=model,
                save_path=str(out_root / cond / f"{cond}.mlpackage"),
                input_size=(1, 3, 224, 224),
                device="cpu",
                validate=False,
            )
            if coreml_path:
                manifest["conditions"][cond]["coreml_path"] = str(coreml_path)
                if verbose:
                    print(f"    -> {coreml_path}")
        except Exception as e:
            manifest["conditions"][cond]["error"] = str(e)
            if verbose:
                print(f"    error: {e}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        n = sum(1 for c in TOP7 if manifest["conditions"][c].get("coreml_path"))
        print(f"\nCoreML exported: {n}/7")
    return 0 if all(manifest["conditions"][c].get("coreml_path") for c in TOP7) else 1


if __name__ == "__main__":
    sys.exit(main())
