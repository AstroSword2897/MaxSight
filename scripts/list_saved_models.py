#!/usr/bin/env python3
"""List paths to saved condition models (best_model.pt per condition)...."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="List saved condition models; optionally copy best_model.pt per condition")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/content/drive/MyDrive/MaxSight"),
        help="Base directory containing checkpoints_<condition> folders",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="If set, copy each best_model.pt here as best_model_<condition>.pt",
    )
    args = parser.parse_args()

    base = args.base
    if not base.exists():
        print(f"Base directory not found: {base}")
        return 1

    dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoints_"))
    if not dirs:
        print("No checkpoints_* directories found.")
        return 0

    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)

    print("Condition models (best_model.pt):\n")
    for d in dirs:
        cond = d.name.replace("checkpoints_", "")
        best = d / "best_model.pt"
        last = d / "last_checkpoint.pt"
        if best.exists():
            path = best.resolve()
            print(f"  {cond}")
            print(f"    best:  {path}")
            if last.exists():
                print(f"    last:  {last.resolve()}")
            if args.copy_to is not None:
                dest = args.copy_to / f"best_model_{cond}.pt"
                shutil.copy2(best, dest)
                print(f"    copied to: {dest}")
            print()
        else:
            print(f"  {cond}: no best_model.pt")
            print()

    if args.copy_to is not None:
        print(f"All best models copied to: {args.copy_to.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

