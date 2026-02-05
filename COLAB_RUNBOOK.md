# MaxSight Colab Runbook — Full Guide

This guide walks you through running MaxSight training and export in Google Colab, step by step. Use it to clone the repo, point to your data on Drive, train the model, export for iOS/Xcode, and recover if the session disconnects.

---

## Table of contents

1. [What this runbook does (big picture)](#1-what-this-runbook-does-big-picture)
2. [Before you start](#2-before-you-start)
3. [Quick path finder](#3-quick-path-finder)
4. [Cells 1–10 (step-by-step)](#4-cells-110-step-by-step)
5. [Recovery and resume](#5-recovery-and-resume)
6. [Troubleshooting](#6-troubleshooting)
7. [Outputs and errors to report](#7-outputs-and-errors-to-report)

---

## 1. What this runbook does (big picture)

**Goal:** Train a MaxSight model (optionally the T5 “temporal” variant) in Colab, save checkpoints to Google Drive, and export the model for use in Xcode or elsewhere.

**Flow in simple terms:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COLAB (temporary)                    │  GOOGLE DRIVE (permanent)            │
├───────────────────────────────────────┼──────────────────────────────────────┤
│  1. Clone repo + install packages     │                                      │
│  2. Mount Drive  ────────────────────┼──► You see your Drive folders       │
│  3. (Optional) Cleanup                 │                                      │
│  4. Verify GPU / run tests            │                                      │
│  5. TRAIN (5a, 5b, 5c, or 5d)        │                                      │
│       • Reads data from Drive         │  ◄── DATA_DIR, SPLITS_DIR            │
│       • Saves checkpoints to Drive    │  ──► CHECKPOINT_DIR (e.g. .pt files) │
│  6. (Optional) Smoke train            │                                      │
│  7. (Optional) GradNorm test          │                                      │
│  8. EXPORT (CoreML / ExecuTorch)      │  ──► EXPORT_DIR (.mlpackage, .pte)   │
│  9. EXPORT iOS bundle (Xcode)         │  ──► EXPORT_DIR (bundle folder)      │
│ 10. Zip exports for download          │  (or download from Drive in browser)  │
└───────────────────────────────────────┴──────────────────────────────────────┘
```

**Important:** Everything that must **survive a disconnect** (checkpoints, exports, data) lives on **Drive**. The Colab machine (`/content/`) is wiped when the runtime stops. So we always set `CHECKPOINT_DIR`, `DATA_DIR`, `SPLITS_DIR`, and `EXPORT_DIR` to paths **on Drive**.

---

## 2. Before you start

### 2.1 What you need

- A **Google account** and **Google Drive** with some free space.
- **Colab**: [colab.research.google.com](https://colab.research.google.com). Create a new notebook.
- **Runtime choice:**
  - **GPU** (recommended for training): **Runtime → Change runtime type → Hardware accelerator → GPU**. Faster training; free tier may time out after a while.
  - **CPU**: **Runtime → Change runtime type → Hardware accelerator → None**. Slower, but no GPU timeout. Use `--device cpu` in every training cell instead of `--device cuda`. Data and checkpoints still sync via Drive.

### 2.2 Where your data lives (Drive layout)

The runbook assumes you have (or will create) folders on Drive like this:

| Folder on Drive | What it’s for |
|-----------------|----------------|
| `My Drive/MaxSight/datasets/coco_raw/` | COCO images (e.g. subfolders `train2017/`, `val2017/`) |
| `My Drive/MaxSight/datasets/coco_raw/cleaned_splits/` | Annotation files: `maxsight_train.json`, `maxsight_val.json` |
| `My Drive/MaxSight/checkpoints/` | Training will save `.pt` checkpoints here (created automatically if missing) |
| `My Drive/MaxSight/exports/` | Exported models will go here (created automatically if missing) |

If your data is in different folders, you’ll set `DATA_DIR` and `SPLITS_DIR` in Cell 2 to match. Cell 2b helps you **find** your current Drive layout.

### 2.3 CPU vs GPU and “sync”

- **Same Drive paths** are used whether you run on CPU or GPU. So:
  - Checkpoints saved on CPU are visible when you later run on GPU (and vice versa).
  - You can resume from `last_checkpoint.pt` on either; no extra “sync” step—Drive is the single place everything is stored.

---

## 3. Quick path finder

Use this to jump to the right place:

| Situation | What to do |
|-----------|------------|
| **First time, I want to train T5 on Drive** | Run cells **1 → 2 → 2b** (optionally 3, 4), then **5b**. Later: **8** (Option B) and **9** if you want export. |
| **I want 80k+ training images (download in Colab)** | Run **1 → 2**, then **2c** (download COCO train2017). Then train with **5b** or **5c** using `--image-dir /content` (see Cell 2c). |
| **First time, I have data in the repo (e.g. `datasets/`)** | Run **1 → 2 → 4**, then **5a**. |
| **I already trained; I just want to export** | Run **1 → 2**, then **8** (and **9** if you want the iOS bundle). Use Option B if you trained T5 (Cell 5b). |
| **Colab disconnected / GPU timed out; I want to resume** | Run **1 → 2**, then the **resume** command in Cell 5b (or 5c). Checkpoints on Drive are used automatically. |
| **I’m on CPU to avoid timeouts** | Use `--device cpu` in cells 5a, 5b, 5c, 5d, 6. Everything else is the same; data and checkpoints stay in sync on Drive. |

---

## 4. Cells 1–10 (step-by-step)

Each cell is a code block you can copy-paste into a Colab cell and run. Under each one we explain what it does, what you should see when it works, and what to watch for.

---

### Cell 1: Clone the repo and install dependencies

**What it does:** Clones the MaxSight repo (branch with T5 Colab script) into Colab and installs Python packages with versions that work on Colab.

**Run this:** Once per new Colab session. **If you see “getcwd: cannot access parent directories” or “The folder you are executing pip from can no longer be found”, do Runtime → Restart session, then run this entire cell from the top (do not run any other cell first).**

```python
# Always start in /content so we're not inside a folder we might delete
%cd /content

# If you're re-running and need a fresh clone, uncomment the next line:
# !rm -rf /content/2026-Prototype

# Clone the branch that has train_t5_fast_colab.py (Cell 5b)
!git clone -q -b feature/multimodal_refactor https://github.com/AstroSword2897/2026-Prototype.git
%cd 2026-Prototype

# Install deps (quoted so shell doesn't break on >=)
!pip install -q "pandas==2.2.2"
!pip install -q "numpy<2.1.0,>=1.26.0"
!pip install -q "pillow<12.0,>=8.0"
!pip install -q "torchvision>=0.24.1" "torchaudio>=2.9.1"
!pip install -q "opencv-python>=4.8.0" "scipy>=1.11.0" "scikit-learn>=1.3.0"
!pip install -q "pytest>=9.0.1" "optuna>=3.0.0" "torchao>=0.14.1"
!pip install -q "matplotlib>=3.10.7" "tqdm>=4.66.0" "flask>=3.0.0" "flask-cors>=4.0.0"
```

**Expected:** No errors. You may see “Requirement already satisfied” for some packages. The working directory will be `/content/2026-Prototype`.

**If something fails:** Note the last error message and paste it in [§7 Outputs and errors to report](#7-outputs-and-errors-to-report) so we can adjust the runbook.

---

### Cell 2: Mount Google Drive and set paths

**What it does:** Connects your Google Drive to Colab and defines where data, checkpoints, and exports live. All of these paths are on Drive so they persist after disconnect.

**Run this:** Once per session. You may be asked to log in to Google and allow Colab to access Drive.

```python
from google.colab import drive
drive.mount("/content/drive")

# Edit these to match your Drive layout (see Cell 2b to discover folders):
DATA_DIR = "/content/drive/MyDrive/MaxSight/datasets/coco_raw"
SPLITS_DIR = "/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits"
CHECKPOINT_DIR = "/content/drive/MyDrive/MaxSight/checkpoints"
EXPORT_DIR = "/content/drive/MyDrive/MaxSight/exports"

# Export to environment so training cells can use $SPLITS_DIR etc. in shell commands
import os
os.environ["DATA_DIR"] = DATA_DIR
os.environ["SPLITS_DIR"] = SPLITS_DIR
os.environ["CHECKPOINT_DIR"] = CHECKPOINT_DIR
os.environ["EXPORT_DIR"] = EXPORT_DIR

!mkdir -p "{CHECKPOINT_DIR}" "{EXPORT_DIR}"
```

**Expected:** A link to authorize access; after you click it, you should see “Mounted at /content/drive”. The `mkdir` creates the checkpoint and export folders if they don’t exist.

**If your data is elsewhere:** Change `DATA_DIR` and `SPLITS_DIR` to the full path (e.g. `"/content/drive/MyDrive/MyData/coco"`). Use Cell 2b to list your Drive and find the right folders.

---

### Cell 2b: Find your Drive layout (optional but helpful)

**What it does:** Lists the top-level contents of “My Drive” and, if present, the contents of a folder like `MaxSight` or `datasets` so you can see where your data actually is.

**Run this:** After Cell 2, if you’re not sure where you put your data or if you need to create folders.

```python
import os
mydrive = "/content/drive/MyDrive"
print("=== Top-level folders in My Drive ===")
for name in sorted(os.listdir(mydrive)):
    path = os.path.join(mydrive, name)
    print(f"  {'(dir) ' if os.path.isdir(path) else ''}{name}")

for candidate in ["MaxSight", "2026-Prototype", "coco_raw", "datasets"]:
    path = os.path.join(mydrive, candidate)
    if os.path.isdir(path):
        print(f"\n=== Contents of MyDrive/{candidate}/ ===")
        for name in sorted(os.listdir(path))[:30]:
            sub = os.path.join(path, name)
            print(f"  {'(dir) ' if os.path.isdir(sub) else ''}{name}")
        if len(os.listdir(path)) > 30:
            print("  ...")
        break
else:
    print("\nNo MaxSight/2026-Prototype/coco_raw/datasets folder found.")
```

**Create default folders if needed:**

```python
!mkdir -p "/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits"
!mkdir -p "/content/drive/MyDrive/MaxSight/checkpoints"
!mkdir -p "/content/drive/MyDrive/MaxSight/exports"
!ls -la "/content/drive/MyDrive/MaxSight/"
```

Then upload your COCO images into `MaxSight/datasets/coco_raw/` (e.g. `train2017/`, `val2017/`) and put `maxsight_train.json` and `maxsight_val.json` in `MaxSight/datasets/coco_raw/cleaned_splits/`.

---

### Cell 2c: Download COCO train2017 in Colab (optional, ~118k train images)

**What it does:** Downloads COCO train2017 (~18 GB) and val2017 (~1 GB) from the official COCO site into `/content/train2017` and `/content/val2017`. Gives you **~118k training images** without storing them on Drive. Annotations still come from Drive (`SPLITS_DIR`).

**When to use:** You want 80k+ training images and are okay with a one-time download in Colab (~30–60 min depending on connection). After this cell, use `--image-dir /content` in training cells (5b, 5c, 5d) instead of `--image-dir "{DATA_DIR}"`.

**Run this:** After Cell 2 (Drive mounted). Run once per session; if the runtime restarts, run again (script skips existing files).

```python
import zipfile
import urllib.request
from pathlib import Path

CONTENT = Path("/content")
ZIP_TRAIN = CONTENT / "train2017.zip"
ZIP_VAL = CONTENT / "val2017.zip"
URL_TRAIN = "http://images.cocodataset.org/zips/train2017.zip"
URL_VAL = "http://images.cocodataset.org/zips/val2017.zip"

def download_with_progress(url: str, dest: Path, label: str = "Downloading"):
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"{label}: found existing {dest.name}, skipping.")
        return
    print(f"{label} {dest.name}...")
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            mb = (block_num * block_size) // (1024 * 1024)
            print(f"\r  {mb} MB ({pct}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()

download_with_progress(URL_TRAIN, ZIP_TRAIN, "Downloading train2017")
download_with_progress(URL_VAL, ZIP_VAL, "Downloading val2017")

for zpath, name in [(ZIP_TRAIN, "train2017"), (ZIP_VAL, "val2017")]:
    out_dir = CONTENT / name
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 1000:
        print(f"Extract: {name} already has images, skipping.")
        continue
    print(f"Extracting {zpath.name}...")
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(CONTENT)
    print(f"  Done {name}.")

train_count = len(list((CONTENT / "train2017").glob("*.jpg")))
val_count = len(list((CONTENT / "val2017").glob("*.jpg")))
print(f"\nTrain: {train_count:,}  |  Val: {val_count:,}")
print("Use --image-dir /content when training.")
```

**After download:** Use training commands with `--image-dir /content`, for example:

```python
%cd /content/2026-Prototype
!python scripts/train_t5_fast_colab.py \
  --data-dir /content \
  --train-annotation "{SPLITS_DIR}/maxsight_train.json" \
  --val-annotation "{SPLITS_DIR}/maxsight_val.json" \
  --image-dir /content \
  --checkpoint-dir "{CHECKPOINT_DIR}" \
  --epochs 55 --warmup-epochs 5 --batch-size 8 --grad-accumulation-steps 4 \
  --train-fraction 0.08 --checkpoint-interval 1 --device cuda
```

(Annotations stay on Drive; images are read from `/content/train2017` and `/content/val2017`.)

---

### Cell 3: (Optional) Cleanup old checkpoints

**What it does:** Deletes old checkpoint files from the **local** `checkpoints/` directory (inside the repo), to free space. It does **not** delete files in `CHECKPOINT_DIR` on Drive.

**When to use:** After a kernel restart if you had been saving checkpoints locally. Skip if you only use Drive paths.

```python
!python scripts/cleanup_cloud_checkpoints.py --checkpoint-dir checkpoints --execute
```

**Expected:** A short log of what was removed (or that nothing was removed). No errors.

---

### Cell 4: Verify GPU and systems

**What it does:** Checks whether CUDA (GPU) is available and runs a small system test suite (imports, model creation, data pipeline, etc.).

```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
!python scripts/test_systems_comprehensive.py --test all
```

**Expected:**  
- With GPU: `CUDA: True Tesla T4` (or similar).  
- With CPU: `CUDA: False N/A`.  
- Tests should finish with “PASS” / “OK” for the components that were run. Some tests may be skipped if data or GPU is missing; that’s often fine.

**If tests fail:** Copy the failing test name and the traceback into [§7](#7-outputs-and-errors-to-report).

---

### Cell 5: Training

**Important:** For cells that use Drive paths (5b, 5c, 5d), **run Cell 2 first**. Those cells use `{SPLITS_DIR}`, `{CHECKPOINT_DIR}`, etc. (curly braces) so the notebook substitutes the **Python** variables from Cell 2. If you see `FileNotFoundError: Annotation files not found: /maxsight_train.json`, the shell got empty paths because Cell 2 was not run (or the variables weren’t set).

---

### Cell 5a: Train with data in the repo (no Drive data)

**What it does:** Trains the model using data under the cloned repo (e.g. `datasets/coco_raw`, `datasets/cleaned_splits`). Checkpoints go to the repo’s `checkpoints/` unless you change the script; they are **lost** when the runtime is recycled. Prefer 5b/5c if you have data on Drive.

**When to use:** You already ran something like `gather_training_data.py` and have `datasets/coco_raw` and `datasets/cleaned_splits` in the repo.

```python
%cd /content/2026-Prototype
!python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --epochs 5 --batch-size 8 --device cuda \
  --use-gradnorm --checkpoint-interval 0
```

Use `--device cpu` if you’re not using a GPU.

**Expected:** Training logs (epoch, loss, etc.). No Python traceback.

---

### Cell 5b: T5 fast training (recommended for Drive)

**What it does:** Trains the **T5 (temporal)** model on a subset of your training data (default 8%), full validation set, 55 epochs (5 warmup + 50 training), and saves a checkpoint to **Drive** every epoch. Designed to fit in a ~4 h run on an A100; on CPU or smaller GPU it will be slower.

**When to use:** You have data on Drive (Cell 2 paths set) and want the main “production-style” Colab training path.

```python
%cd /content/2026-Prototype
!python scripts/train_t5_fast_colab.py \
  --data-dir "{DATA_DIR}" \
  --train-annotation "{SPLITS_DIR}/maxsight_train.json" \
  --val-annotation "{SPLITS_DIR}/maxsight_val.json" \
  --image-dir "{DATA_DIR}" \
  --checkpoint-dir "{CHECKPOINT_DIR}" \
  --epochs 55 --warmup-epochs 5 --batch-size 8 --grad-accumulation-steps 4 \
  --train-fraction 0.08 --checkpoint-interval 1 --device cuda
```

**On CPU:** Use `--device cpu` instead of `--device cuda`.

**Expected:** Logs like “Creating data loaders…”, “Subset: … samples”, “Train: …/… samples”, then per-epoch train/val loss and checkpoint saves. No unhandled exception.

**Resume after disconnect:** Run Cell 1 and 2 again, then:

```python
%cd /content/2026-Prototype
!python scripts/train_t5_fast_colab.py \
  --data-dir "{DATA_DIR}" \
  --train-annotation "{SPLITS_DIR}/maxsight_train.json" \
  --val-annotation "{SPLITS_DIR}/maxsight_val.json" \
  --image-dir "{DATA_DIR}" \
  --checkpoint-dir "{CHECKPOINT_DIR}" \
  --epochs 55 --warmup-epochs 5 --batch-size 8 --grad-accumulation-steps 4 \
  --train-fraction 0.08 --checkpoint-interval 1 --device cuda \
  --resume-from "{CHECKPOINT_DIR}/last_checkpoint.pt"
```

(Use `--device cpu` if you’re on CPU.)

**Useful flags:**  
- `--train-fraction 0.05` → smaller subset, faster.  
- `--epochs 40 --warmup-epochs 3` → shorter run.

---

### Cell 5c: Generic training (any tier, data on Drive)

**What it does:** Same as 5a but reads data and writes checkpoints to the **Drive** paths you set in Cell 2. Good for non-T5 tiers or custom configs.

```python
%cd /content/2026-Prototype
!python scripts/train_maxsight.py \
  --data-dir "{DATA_DIR}" \
  --train-annotation "{SPLITS_DIR}/maxsight_train.json" \
  --val-annotation "{SPLITS_DIR}/maxsight_val.json" \
  --image-dir "{DATA_DIR}" \
  --checkpoint-dir "{CHECKPOINT_DIR}" \
  --epochs 5 --batch-size 8 --device cuda \
  --use-gradnorm --checkpoint-interval 1
```

Use `--device cpu` if needed. **Expected:** Same kind of training logs as 5a, with checkpoints in `CHECKPOINT_DIR` on Drive.

---

### Cell 5d: One-shot production script (env-based)

**What it does:** Runs the production training script with options set by environment variables. Can use repo defaults or Drive paths.

**With Drive paths (after Cell 2):**

```python
%cd /content/2026-Prototype
!DATA_DIR="{DATA_DIR}" TRAIN_ANN="{SPLITS_DIR}/maxsight_train.json" VAL_ANN="{SPLITS_DIR}/maxsight_val.json" IMAGE_DIR="{DATA_DIR}" EPOCHS=5 BATCH_SIZE=8 DEVICE=cuda ./scripts/run_production_training.sh --no-export
```

Use `DEVICE=cpu` if you’re on CPU.

---

### Cell 6: (Optional) Smoke train

**What it does:** Very short training run (tiny batch, 1 epoch) to check that the model and data pipeline run without crashing. Useful before a long 5b run.

```python
%cd /content/2026-Prototype
!python scripts/smoke_train.py --device cuda
```

Use `--device cpu` if needed. **Expected:** A few lines of output and exit without traceback.

---

### Cell 7: (Optional) GradNorm test

**What it does:** Runs the GradNorm integration tests only.

```python
!python -m pytest tests/test_gradnorm_integration.py -v
```

**Expected:** Tests pass or skip; no critical failures.

---

### Cell 8: Export model to Drive (CoreML + ExecuTorch)

**What it does:** Loads the best (or last) checkpoint from Drive and exports it as CoreML (`.mlpackage`) and ExecuTorch (`.pte`) into `EXPORT_DIR` on Drive. Run **after** at least one training run that wrote to `CHECKPOINT_DIR`.

**Option A — You used generic training (5a, 5c, 5d, not T5):**

```python
%cd /content/2026-Prototype
!pip install -q coremltools 2>/dev/null || true
!pip install -q executorch 2>/dev/null || true
from pathlib import Path
ckpt = Path(CHECKPOINT_DIR) / "best_model.pt"
if not ckpt.exists():
    ckpt = Path(CHECKPOINT_DIR) / "last_checkpoint.pt"
!python -m ml.training.export --checkpoint "{ckpt}" --format coreml --output "{EXPORT_DIR}/maxsight.mlpackage" --device cpu
!python -m ml.training.export --checkpoint "{ckpt}" --format executorch --output "{EXPORT_DIR}/maxsight.pte" --device cpu
print("Exports saved to", EXPORT_DIR)
```

**Option B — You trained T5 (Cell 5b):** The exported model must be T5 as well. Run this instead:

```python
%cd /content/2026-Prototype
!pip install -q coremltools 2>/dev/null || true
!pip install -q executorch 2>/dev/null || true
import torch
from pathlib import Path
from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier
from ml.training.export import export_to_coreml, export_to_executorch
ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
if not ckpt_path.exists():
    ckpt_path = Path(CHECKPOINT_DIR) / "last_checkpoint.pt"
model = create_model(tier_config=TierConfig.for_tier(CapabilityTier.T5_TEMPORAL))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
export_to_coreml(model, str(Path(EXPORT_DIR) / "maxsight_t5.mlpackage"), device="cpu")
export_to_executorch(model, str(Path(EXPORT_DIR) / "maxsight_t5.pte"))
print("T5 exports saved to", EXPORT_DIR)
```

**Expected:** No Python errors; “Exports saved to …” and files under `EXPORT_DIR` on Drive.

---

### Cell 9: Export iOS bundle (Xcode-ready)

**What it does:** Builds a folder with `maxsight.pte`, `model_config.json`, `runtime_config.json`, `processing_reference.py`, and `README_XCODE.md` and saves it to Drive. Use this if you want to integrate the model into an Xcode project.

**If you trained T5 (5b):**

```python
%cd /content/2026-Prototype
import torch
from pathlib import Path
from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier
from ml.training.export import export_ios_bundle
ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
if not ckpt_path.exists():
    ckpt_path = Path(CHECKPOINT_DIR) / "last_checkpoint.pt"
model = create_model(tier_config=TierConfig.for_tier(CapabilityTier.T5_TEMPORAL))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
bundle_dir = str(Path(EXPORT_DIR) / "maxsight_ios_bundle")
export_ios_bundle(model, output_dir=bundle_dir)
print("iOS bundle at", bundle_dir)
```

For **generic (T0) training**, use `model = create_model()` and remove the `tier_config=...` argument.

**Expected:** “iOS bundle at …” and the bundle folder created under `EXPORT_DIR`.

---

### Cell 10: Download exports from Colab

**What it does:** Zips the contents of `EXPORT_DIR` (on Drive) into a file under `/content/` so you can download it from the Colab “Files” panel. Alternatively, you can open Drive in the browser and download the export folder directly.

```python
!ls -la "{EXPORT_DIR}"
!zip -r /content/maxsight_exports.zip "{EXPORT_DIR}"
print("Download /content/maxsight_exports.zip from the Colab Files panel (left sidebar)")
```

**Expected:** A zip file appears in the file browser; you can right-click → Download.

---

## 5. Recovery and resume

**After a disconnect or GPU timeout:**

1. Open the same (or a new) Colab notebook.
2. Run **Cell 1** (clone + install) and **Cell 2** (mount Drive, same paths).
3. Check that checkpoints exist:
   ```python
   !ls -lh "{CHECKPOINT_DIR}"
   ```
4. Resume with the **resume** command from Cell 5b (or 5c), using `--resume-from "{CHECKPOINT_DIR}/last_checkpoint.pt"` and the same flags (including `--device cpu` or `cuda`) as before.

**If you didn’t save checkpoints to Drive:** Anything in `/content/` is gone. Start again from Cell 1 and 2, and use `CHECKPOINT_DIR` (and optionally 5b/5c) so future runs save to Drive.

**Export only:** If you already have checkpoints on Drive and only want to export, run Cell 1, Cell 2, then Cell 8 (and 9 if you want the iOS bundle). Use Option B in Cell 8/9 if the checkpoint is from T5 training.

---

## 6. Troubleshooting

| Symptom | What to try |
|--------|--------------|
| **“No such file or directory” for annotation or image path** | Run Cell 2b and confirm `DATA_DIR` and `SPLITS_DIR` point to folders that contain `maxsight_train.json`, `maxsight_val.json`, and the image directories. Fix paths in Cell 2 and re-run. |
| **“CUDA out of memory”** | Use a smaller batch size (e.g. `--batch-size 4`) or `--train-fraction 0.05`. Or switch to CPU with `--device cpu`. |
| **GPU timeout / runtime disconnect** | Use CPU (`--device cpu`) or a smaller run (`--epochs 40`, `--train-fraction 0.05`). Checkpoints on Drive are still there; resume with the 5b resume command. |
| **Training loss is NaN** | See **COLAB_RESTART_GUIDE.md** (NaN loss, GradNorm, learning rate). Reduce learning rate or try a smaller subset. |
| **Export fails with “missing keys” or “unexpected keys”** | You may be loading a T5 checkpoint into the default (T0) model. Use **Option B** in Cell 8 and 9 for T5 checkpoints. |
| **Mount Drive fails or “access denied”** | Complete the Google auth in the popup/link and allow Colab to access Drive. Try again in a new cell. |
| **“getcwd: cannot access parent directories” / “The folder you are executing pip from can no longer be found” / “=0.24.1: No such file or directory”** | The shell’s current directory was deleted (e.g. you ran `rm -rf` on the repo while the kernel was inside it). **Fix:** Runtime → Restart session. Then run **only Cell 1** from the top (whole cell at once). Cell 1 now starts with `%cd /content` so the CWD is safe before any clone. Do not run a cell that only does `rm -rf` and clone while the kernel was already inside `2026-Prototype`. |

More detailed troubleshooting: **COLAB_RESTART_GUIDE.md**, **QUICK_START_CLOUD.md** (for other clouds).

---

## 7. Outputs and errors to report

To improve this runbook, please paste here (or send) **successful outputs** or **error messages** you see:

- **Which cell** (e.g. “Cell 5b”, “Cell 8 Option B”).
- **Runtime:** GPU or CPU; Colab free or Pro.
- **Exact error message** and last 20–30 lines of traceback (if any).
- **A short snippet of successful output** (e.g. last few lines of a training epoch or export), so we can document “what good looks like”.

Example:

```
Cell: 5b
Runtime: CPU, Colab free
Error: FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits/maxsight_train.json'
```

Or:

```
Cell: 5b
Runtime: GPU T4, Colab free
Success: Training completed; last line was "Best val loss: 0.8234 at epoch 55"
```

---

## Quick reference

| Goal | Cells |
|------|--------|
| First time, train T5 on Drive | 1 → 2 → 2b (optional) → 4 (optional) → **5b** |
| First time, train with repo data | 1 → 2 → 4 → **5a** |
| Resume after disconnect | 1 → 2 → **5b resume** (or 5c resume) |
| Export after training | 1 → 2 → **8** (Option A or B) → **9** (optional) |
| Find Drive layout | **2b** |
| Use CPU to avoid timeouts | Use `--device cpu` in 5a, 5b, 5c, 5d, 6 |

---

*See **COLAB_COMMANDS.md** for a copy-paste-only list of commands without explanations.*
