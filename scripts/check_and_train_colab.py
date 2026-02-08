#!/usr/bin/env python3
"""One script: check setup for training, then run training.
Run in Colab after mounting Drive. Set DATA_DIR and RUN_TRAINING below."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Override via env: DATA_DIR, IMAGE_DIR, REPO_DIR, RUN_TRAINING, MODE, BATCH_SIZE, EPOCHS, etc.
DATA_DIR = os.environ.get("DATA_DIR", "/content/drive/MyDrive/MaxSight_Training")
IMAGE_DIR = os.environ.get("IMAGE_DIR", DATA_DIR)
REPO_DIR = Path(os.environ.get("REPO_DIR", "/content/2026-Prototype"))
RUN_TRAINING = os.environ.get("RUN_TRAINING", "1") == "1"
MODE = os.environ.get("MODE", "train")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
N_TRIALS = int(os.environ.get("N_TRIALS", "5"))
EPOCHS_PER_TRIAL = int(os.environ.get("EPOCHS_PER_TRIAL", "2"))

DRIVE_MOUNT = "/content/drive"
CLEANED = "cleaned_splits"
TRAIN_ANN = "maxsight_train.json"
VAL_ANN = "maxsight_val.json"


def log(msg: str, ok: bool = None) -> None:
    prefix = "OK " if ok is True else "FAIL " if ok is False else "INFO "
    print(prefix + msg)


def main() -> int:
    print("=" * 60)
    print("Check & Train – setup then training")
    print("=" * 60)

    run_training = RUN_TRAINING
    data = Path(DATA_DIR)
    image_dir = Path(IMAGE_DIR)
    repo = REPO_DIR
    cleaned_dir = data / CLEANED
    train_json = cleaned_dir / TRAIN_ANN
    val_json = cleaned_dir / VAL_ANN

    # 1. Drive mounted.
    if not Path(DRIVE_MOUNT).exists():
        log("Drive not mounted. Run: from google.colab import drive; drive.mount('/content/drive')", ok=False)
        return 1
    log("Drive mounted")

    # 2. Data dir exists (create if missing)
    if not data.exists():
        log(f"Data dir missing: {data}", ok=False)
        return 1
    log(f"Data dir: {data}")

    # 3. cleaned_splits and JSONs.
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    log(f"Cleaned splits dir: {cleaned_dir}")

    if not train_json.exists() or not val_json.exists():
        # Copy from repo when present.
        repo_splits = repo / "datasets" / "cleaned_splits"
        for name in [TRAIN_ANN, VAL_ANN]:
            src = repo_splits / name
            dst = cleaned_dir / name
            if src.exists():
                shutil.copy(src, dst)
                log(f"Copied {name} from repo to {cleaned_dir}")
        # If still missing, write minimal empty JSONs so check passes.
        annotations_empty = False
        if not train_json.exists():
            train_json.write_text("[]")
            log(f"Created empty {TRAIN_ANN} – replace with real annotations for training", ok=None)
            annotations_empty = True
        if not val_json.exists():
            val_json.write_text("[]")
            log(f"Created empty {VAL_ANN} – replace with real annotations for training", ok=None)
            annotations_empty = True
        if annotations_empty:
            print()
            print("WARNING EMPTY annotation files. Skipping training (would fail with empty dataset).")
            print("   Add real maxsight_train.json and maxsight_val.json to Drive/MaxSight_Training/cleaned_splits/")
            print("   Then re-run this script or run the training command below.")
            print()
            run_training = False
    else:
        log(f"Annotations found: {TRAIN_ANN}, {VAL_ANN}")

    # 4. Repo.
    if not repo.exists():
        log(f"Repo not found: {repo}. Clone 2026-Prototype first.", ok=False)
        return 1
    log(f"Repo: {repo}")

    train_script = repo / "scripts" / "train_maxsight.py"
    automl_script = repo / "scripts" / "AutoMLType.py"
    if not train_script.exists():
        log(f"Train script missing: {train_script}", ok=False)
        return 1
    log("Train script found")

    # 5. Optional: check image dir has some content.
    if image_dir.exists():
        try:
            next(image_dir.rglob("*.jpg"), None) or next(image_dir.rglob("*.jpeg"), None)
            log("Image dir has images")
        except StopIteration:
            log("Image dir has no .jpg/.jpeg (annotations may point to other paths)", ok=None)
    else:
        log(f"Image dir not found: {image_dir}. Set IMAGE_DIR if images are elsewhere.", ok=None)

    print()
    print("Setup OK. Starting training..." if run_training else "Setup OK. Run training manually (see command below).")
    print()

    if not run_training:
        print("Command to run training:")
        print(f"  cd {repo} && python scripts/train_maxsight.py \\")
        print(f"    --data-dir {data} --train-annotation {TRAIN_ANN} --val-annotation {VAL_ANN} \\")
        print(f"    --image-dir {image_dir} --device cuda --batch-size {BATCH_SIZE} --epochs {EPOCHS} \\")
        print(f"    --num-workers {NUM_WORKERS} --checkpoint-interval 0 --use-gradnorm")
        return 0

    # Run training.
    os.chdir(repo)
    if MODE == "automl":
        cmd = [
            sys.executable,
            str(automl_script),
            "--data-dir", str(data),
            "--train-annotation", TRAIN_ANN,
            "--val-annotation", VAL_ANN,
            "--image-dir", str(image_dir),
            "--checkpoint-dir", str(repo / "checkpoints_automl"),
            "--n-trials", str(N_TRIALS),
            "--epochs-per-trial", str(EPOCHS_PER_TRIAL),
            "--num-workers", str(NUM_WORKERS),
            "--device", "cuda",
            "--use-gradnorm",
        ]
    else:
        cmd = [
            sys.executable,
            str(train_script),
            "--data-dir", str(data),
            "--train-annotation", TRAIN_ANN,
            "--val-annotation", VAL_ANN,
            "--image-dir", str(image_dir),
            "--checkpoint-dir", str(repo / "checkpoints"),
            "--device", "cuda",
            "--batch-size", str(BATCH_SIZE),
            "--epochs", str(EPOCHS),
            "--num-workers", str(NUM_WORKERS),
            "--checkpoint-interval", "0",
            "--use-gradnorm",
        ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())


