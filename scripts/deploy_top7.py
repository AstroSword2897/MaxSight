#!/usr/bin/env python3
"""Validate and export the top 7 (alive) inference conditions for deployment.

- Checks all 7 checkpoints exist and run a one-batch inference sanity check.
- Exports each to an iOS-ready bundle (PTE + configs) under --output-dir/<condition>/.
- Writes manifest.json for deployment.

Run in under an hour (e.g. ~5 min per condition export on CPU).

Usage:
  python scripts/deploy_top7.py --checkpoints-base /path/to/checkpoints --output-dir exports/top7
  python scripts/deploy_top7.py --checkpoints-base /content/drive/MyDrive/MaxSight --output-dir /content/drive/MyDrive/MaxSight/exports_top7 --validate-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOP7_CONDITIONS = [
    "amblyopia", "amd", "color_blindness", "cvi", "glaucoma",
    "retinitis_pigmentosa", "strabismus",
]


def _find_checkpoints_base() -> Optional[Path]:
    """Discover base dir that has at least one checkpoints_<cond>/best_model.pt."""
    import os
    candidates = [os.environ.get("CHECKPOINTS_BASE")]
    if candidates[0]:
        candidates = [Path(candidates[0])]
    else:
        candidates = []
    candidates += [REPO / "checkpoints", REPO / "backups"]
    home = Path.home()
    candidates += [home / "Google Drive" / "My Drive" / "MaxSight"]
    candidates += list(home.glob("Library/CloudStorage/GoogleDrive-*/My Drive/MaxSight"))
    candidates += [Path("/content/drive/MyDrive/MaxSight")]
    for base in candidates:
        if not base:
            continue
        base = Path(base)
        if not base.exists():
            continue
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.startswith("checkpoints_") and (d / "best_model.pt").exists():
                    return base.resolve()
        except OSError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate and export top 7 condition models for deployment."
    )
    parser.add_argument("--checkpoints-base", type=Path, default=None,
                        help="Base dir with checkpoints_<cond>/best_model.pt (default: auto-detect or CHECKPOINTS_BASE)")
    parser.add_argument("--output-dir", type=Path, default=REPO / "exports" / "top7",
                        help="Output root; each condition gets output_dir/<cond>/")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help=f"Conditions to deploy (default: top 7 list, or top 7 by mAP if --top-by-map)")
    parser.add_argument("--top-by-map", action="store_true",
                        help="Use top 7 conditions by mAP from --inference-data (overrides --conditions)")
    parser.add_argument("--inference-data", type=Path, default=REPO / "inference_data.json",
                        help="Path to inference_data.json (used with --top-by-map)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only check checkpoints and run 1-batch inference; no export")
    parser.add_argument("--skip-export", action="store_true",
                        help="Same as --validate-only")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    from ml.models.maxsight_cnn import (
        COCO_CLASSES,
        CapabilityTier,
        TierConfig,
        create_model,
    )
    from ml.training.export import export_ios_bundle

    def _is_placeholder(p) -> bool:
        if p is None:
            return True
        s = str(p)
        return "/path/to" in s or "path/to" in s
    if _is_placeholder(args.checkpoints_base):
        args.checkpoints_base = None
    base = Path(args.checkpoints_base).resolve() if args.checkpoints_base else None
    if base is None:
        base = _find_checkpoints_base()
    if base is None:
        print("No checkpoints base found. Either:", file=sys.stderr)
        print("  1. Pass --checkpoints-base /path/to/dir (folder containing checkpoints_amblyopia/best_model.pt etc.)", file=sys.stderr)
        print("  2. Set CHECKPOINTS_BASE and run again", file=sys.stderr)
        print("  3. Run: python scripts/find_trained_checkpoints.py  (to see if any known path has checkpoints)", file=sys.stderr)
        return 1
    base = base.resolve()
    out_root = Path(args.output_dir).resolve()
    validate_only = args.validate_only or args.skip_export
    device = args.device
    verbose = not args.quiet
    if getattr(args, "top_by_map", False) and args.inference_data and Path(args.inference_data).exists():
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "get_top7_by_map.py"),
             "--inference-data", str(args.inference_data), "--k", "7", "--print"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            conditions = [c.strip() for c in r.stdout.strip().splitlines() if c.strip()]
        else:
            conditions = []
        if not conditions:
            print("No conditions found in inference data; falling back to default top 7.", file=sys.stderr)
            conditions = TOP7_CONDITIONS
        else:
            conditions = conditions[:7]
            print("Top 7 by mAP (mAP@0.5, descending):", ", ".join(conditions))
            if verbose:
                try:
                    with open(args.inference_data) as f:
                        idata = json.load(f)
                    map_by_cond = {}
                    for r in idata.get("results", []):
                        if isinstance(r, dict) and "error" not in r:
                            map_by_cond[r["condition"]] = r.get("mAP_50", r.get("mAP", 0))
                    for c in conditions:
                        print(f"  {c}: mAP_50={map_by_cond.get(c, '?')}")
                except Exception:
                    pass
    else:
        conditions = (args.conditions or TOP7_CONDITIONS)[:7]
    conditions = conditions[:7]

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    num_classes = len(COCO_CLASSES)
    manifest = {"checkpoints_base": str(base), "output_dir": str(out_root), "conditions": {}}

    for cond in conditions:
        ckpt_path = base / f"checkpoints_{cond}" / "best_model.pt"
        manifest["conditions"][cond] = {
            "checkpoint": str(ckpt_path),
            "exists": ckpt_path.exists(),
            "inference_ok": False,
            "export_path": None,
            "error": None,
        }
        if not ckpt_path.exists():
            if verbose:
                print(f"  {cond}: missing {ckpt_path}")
            continue

        if verbose:
            print(f"  {cond}: loading and validating...")
        try:
            model = create_model(
                num_classes=num_classes,
                use_audio=False,
                condition_mode=cond,
                tier_config=tier_config,
            )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224, device=device)
                out = model(dummy)
            if not isinstance(out, dict) or "objectness" not in out:
                manifest["conditions"][cond]["error"] = "forward pass missing objectness"
                if verbose:
                    print(f"    inference check failed: no objectness in output")
                continue
            manifest["conditions"][cond]["inference_ok"] = True
            if verbose:
                print(f"    inference OK")
        except Exception as e:
            manifest["conditions"][cond]["error"] = str(e)
            if verbose:
                print(f"    error: {e}")
            continue

        if validate_only:
            continue

        cond_out = out_root / cond
        cond_out.mkdir(parents=True, exist_ok=True)
        try:
            model.cpu()
            bundle_path = export_ios_bundle(
                model=model,
                output_dir=str(cond_out),
                input_size=(1, 3, 224, 224),
            )
            manifest["conditions"][cond]["export_path"] = str(bundle_path)
            if verbose:
                print(f"    exported -> {bundle_path}")
        except Exception as e:
            import traceback
            manifest["conditions"][cond]["error"] = f"export: {e}"
            if verbose:
                print(f"    export failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    manifest_path = out_root / "manifest.json"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"\nManifest: {manifest_path}")

    all_ok = all(
        manifest["conditions"][c].get("inference_ok") for c in conditions
    )
    all_exported = validate_only or all(
        not manifest["conditions"][c].get("inference_ok")
        or manifest["conditions"][c].get("export_path")
        for c in conditions
    )
    if verbose:
        print(f"Validated: {sum(1 for c in conditions if manifest['conditions'][c].get('inference_ok'))}/{len(conditions)}")
        if not validate_only:
            print(f"Exported:  {sum(1 for c in conditions if manifest['conditions'][c].get('export_path'))}/{len(conditions)}")
    found = sum(1 for c in conditions if manifest["conditions"][c].get("exists"))
    if found == 0:
        print(f"No best_model.pt found for any of the top 7 under: {base}", file=sys.stderr)
        print("Each condition needs: <base>/checkpoints_<cond>/best_model.pt", file=sys.stderr)
        print("Train first: python scripts/train_alive_models.py --checkpoints-base <base> --data-dir <data> --train-annotation ... --val-annotation ...", file=sys.stderr)
        print("Or copy trained checkpoints into that layout. Discover path: python scripts/find_trained_checkpoints.py", file=sys.stderr)
    return 0 if (all_ok and (validate_only or all_exported)) else 1


if __name__ == "__main__":
    sys.exit(main())
