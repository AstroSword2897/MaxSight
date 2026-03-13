#!/usr/bin/env python3
"""Train alive-condition models on the same train/val data (inference splits)."""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

CONDITIONS_DEFAULT = [
    "amblyopia", "amd", "color_blindness", "cvi", "glaucoma",
    "retinitis_pigmentosa", "strabismus",
]
TRAIN_SCRIPT = REPO / "scripts" / "train_maxsight.py"


def main():
    parser = argparse.ArgumentParser(
        description="Train alive-condition models on train/val splits; one run per condition."
    )
    parser.add_argument("--checkpoints-base", type=Path, required=True,
                        help="Base dir; each condition saves to checkpoints_base/checkpoints_<cond>/")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data root (passed to train_maxsight --data-dir)")
    parser.add_argument("--train-annotation", type=Path, required=True,
                        help="Train split JSON (e.g. .../cleaned_splits/maxsight_train.json)")
    parser.add_argument("--val-annotation", type=Path, required=True,
                        help="Val split JSON (e.g. .../cleaned_splits/maxsight_val.json)")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Image root (default: data-dir)")
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS_DEFAULT,
                        help=f"Conditions to train (default: {CONDITIONS_DEFAULT})")
    # Training args passed through.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--use-gradnorm", action="store_true", default=True,
                        help="Use GradNorm (default: True)")
    parser.add_argument("--no-gradnorm", action="store_false", dest="use_gradnorm")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each condition from last/best in its checkpoint dir")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    args = parser.parse_args()

    if not TRAIN_SCRIPT.exists():
        print(f"Not found: {TRAIN_SCRIPT}", file=sys.stderr)
        return 1

    base = Path(args.checkpoints_base).resolve()
    data_dir = Path(args.data_dir).resolve()
    train_ann = Path(args.train_annotation).resolve()
    val_ann = Path(args.val_annotation).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else data_dir

    conditions = args.conditions or CONDITIONS_DEFAULT
    n = len(conditions)
    for i, cond in enumerate(conditions, 1):
        ckpt_dir = base / f"checkpoints_{cond}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data-dir", str(data_dir),
            "--train-annotation", str(train_ann),
            "--val-annotation", str(val_ann),
            "--image-dir", str(image_dir),
            "--checkpoint-dir", str(ckpt_dir),
            "--condition-mode", cond,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--grad-clip", str(args.grad_clip),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--early-stopping-patience", str(args.early_stopping_patience),
        ]
        if args.use_gradnorm:
            cmd.append("--use-gradnorm")
        if args.resume:
            cmd.append("--resume")

        print(f"[{i}/{n}] Training condition={cond} -> {ckpt_dir}")
        result = subprocess.run(cmd, cwd=str(REPO))
        if result.returncode != 0:
            print(f"  FAILED condition={cond} (exit {result.returncode})", file=sys.stderr)
            return result.returncode
        print(f"  Done {cond} -> {ckpt_dir / 'best_model.pt'}")

    print(f"\nAll {n} conditions trained. Run inference with improve_map_all_models.py or run_checkpoint_inference.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())






