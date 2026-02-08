#!/usr/bin/env python3
"""Load one checkpoint and export to JIT; print full traceback on error.

Use when deploy fails and you need the real error. From repo root (e.g. on Colab after git pull):

  python scripts/export_one_model.py --checkpoint /path/to/checkpoints_amblyopia/best_model.pt --out maxsight.pt
  python scripts/export_one_model.py --condition amblyopia --checkpoints-base /content/drive/MyDrive/MaxSight --out /tmp/amblyopia.pt

If the run stops after "JIT export: running torch.jit.trace" with no error, the runtime was likely killed (OOM).
Try --fp16 to trace in half precision, or Runtime → Factory reset runtime. Use --no-subprocess to run in the same process.
"""

import argparse
import os
import subprocess
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
    parser.add_argument("--no-subprocess", action="store_true", help="Run in this process (default: run in subprocess with 10 min timeout to detect OOM/kill)")
    parser.add_argument("--fp16", action="store_true", help="Trace in half precision (uses less memory; can avoid OOM on Colab)")
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

    # Run export in a subprocess so OOM or timeout yields a clear exit message instead of silent kill.
    if not args.no_subprocess and os.environ.get("EXPORT_ONE_CHILD") != "1":
        cmd = [sys.executable, str(REPO / "scripts" / "export_one_model.py")]
        if args.checkpoint:
            cmd += ["--checkpoint", str(ckpt_path)]
        if args.condition:
            cmd += ["--condition", args.condition]
        if args.checkpoints_base:
            cmd += ["--checkpoints-base", str(args.checkpoints_base)]
        cmd += ["--out", str(out_path), "--device", device, "--no-subprocess"]
        env = {**os.environ, "EXPORT_ONE_CHILD": "1"}
        print("Running export in subprocess (timeout 10 min). If killed, you'll see exit code below.", flush=True)
        try:
            r = subprocess.run(cmd, cwd=str(REPO), env=env, timeout=600)
            if r.returncode == 0:
                print(f"Done. Saved: {out_path}", flush=True)
                return 0
            if r.returncode == -9:
                print("Process killed (signal 9). Likely out of memory during JIT trace.", flush=True)
            else:
                print(f"Subprocess exited with code {r.returncode}", flush=True)
            return 1
        except subprocess.TimeoutExpired:
            print("Export timed out after 10 minutes. JIT trace may be stuck or very slow.", flush=True)
            return 1

    def progress(msg: str) -> None:
        print(msg, flush=True)
        p = os.environ.get("EXPORT_ONE_PROGRESS")
        if p:
            Path(p).write_text(msg + "\n")

    try:
        progress("Step 1: Importing model and export...")
        from ml.models.maxsight_cnn import (
            COCO_CLASSES,
            CapabilityTier,
            TierConfig,
            create_model,
        )
        from ml.training.export import export_to_jit

        cond = args.condition or ckpt_path.parent.name.replace("checkpoints_", "")
        progress(f"Step 2: Loading model from checkpoint (condition={cond})...")
        tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
        model = create_model(
            num_classes=len(COCO_CLASSES),
            use_audio=False,
            condition_mode=cond,
            tier_config=tier_config,
        )

        progress("Step 3: Loading best_model.pt weights into model...")
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        progress("Step 4: One forward pass...")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            out = model(dummy)
        if not isinstance(out, dict) or "objectness" not in out:
            print("Forward pass failed: output missing 'objectness'", file=sys.stderr)
            return 1
        print("  Forward OK.", flush=True)

        # Free memory before trace (trace is memory-heavy)
        del dummy, out
        del ckpt, state
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        progress("Step 5: Exporting to JIT...")
        model.cpu()
        use_fp16 = getattr(args, "fp16", False)
        if use_fp16:
            progress("Step 5b: Using FP16 (half precision) to reduce memory.")
        export_to_jit(
            model,
            save_path=str(out_path),
            input_size=(1, 3, 224, 224),
            device="cpu",
            validate=False,
            use_fp16=use_fp16,
        )
        progress(f"Done. Saved: {out_path}")
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


