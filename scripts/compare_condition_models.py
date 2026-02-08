#!/usr/bin/env python3
"""Compare trained condition models by best validation loss and mAP...."""
import argparse
import json
from pathlib import Path


def get_best_metrics(ckpt_dir: Path) -> dict | None:
    """Read best val loss and mAP from resume_info.json, training_history.json, or best_model.pt."""
    # 1. resume_info.json (written on every save)
    resume_info = ckpt_dir / "resume_info.json"
    if resume_info.exists():
        try:
            with open(resume_info) as f:
                data = json.load(f)
            return {
                "best_val_loss": data.get("best_val_loss"),
                "best_val_map": data.get("best_val_map"),
                "epoch": data.get("epoch"),
                "total_epochs": data.get("total_epochs"),
            }
        except Exception:
            pass

    # 2. training_history.json (written when training completes)
    history_path = ckpt_dir / "training_history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            val_loss = history.get("val_loss", [])
            val_map = history.get("val_map", [])
            if val_loss:
                return {
                    "best_val_loss": min(float(x) for x in val_loss),
                    "best_val_map": max(float(x) for x in val_map) if val_map else 0.0,
                    "epoch": len(val_loss) - 1,
                    "total_epochs": len(val_loss),
                }
        except Exception:
            pass

    best_pt = ckpt_dir / "best_model.pt"
    if best_pt.exists():
        try:
            import torch
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                return {
                    "best_val_loss": ckpt.get("best_val_loss"),
                    "best_val_map": ckpt.get("best_val_map"),
                    "epoch": ckpt.get("epoch"),
                    "total_epochs": ckpt.get("config", {}).get("num_epochs"),
                }
        except Exception:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Compare condition models by best val loss / mAP")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/content/drive/MyDrive/MaxSight"),
        help="Base directory containing checkpoints_<condition> folders",
    )
    parser.add_argument("--sort", choices=["loss", "map"], default="loss", help="Sort by val loss (lower better) or mAP (higher better)")
    args = parser.parse_args()

    base = args.base
    if not base.exists():
        print(f"Base directory not found: {base}")
        return

    rows = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("checkpoints_"):
            continue
        cond = d.name.replace("checkpoints_", "")
        metrics = get_best_metrics(d)
        if metrics and (metrics.get("best_val_loss") is not None or metrics.get("best_val_map") is not None):
            rows.append({
                "condition": cond,
                "best_val_loss": metrics.get("best_val_loss"),
                "best_val_map": metrics.get("best_val_map"),
                "epoch": metrics.get("epoch"),
                "total_epochs": metrics.get("total_epochs"),
            })
        else:
            rows.append({
                "condition": cond,
                "best_val_loss": None,
                "best_val_map": None,
                "epoch": None,
                "total_epochs": None,
            })

    if not rows:
        print("No checkpoints_* directories found.")
        return

    # Sort: by loss (asc) or by mAP (desc)
    def sort_key(r):
        loss = r["best_val_loss"] if r["best_val_loss"] is not None else float("inf")
        map_ = r["best_val_map"] if r["best_val_map"] is not None else -1.0
        if args.sort == "loss":
            return (loss, -map_)
        return (-map_, loss)

    rows.sort(key=sort_key)

    # Print table.
    print(f"\n{'Condition':<25} {'Best val loss':>14} {'Best val mAP':>14} {'Epoch':>8}")
    print("-" * 65)
    for r in rows:
        loss_s = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "—"
        map_s = f"{r['best_val_map']:.4f}" if r["best_val_map"] is not None else "—"
        ep_s = f"{r['epoch']+1}/{r['total_epochs']}" if r["epoch"] is not None and r["total_epochs"] is not None else "—"
        print(f"{r['condition']:<25} {loss_s:>14} {map_s:>14} {ep_s:>8}")
    print()
    print("Lower val loss = better. Higher mAP = better.")
    if args.sort == "loss":
        print("(Sorted by best val loss, best first.)")
    else:
        print("(Sorted by best val mAP, best first.)")


if __name__ == "__main__":
    main()

