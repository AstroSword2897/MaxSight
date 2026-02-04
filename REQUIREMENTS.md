# MaxSight – Software and Hardware Requirements

Single reference for what you need to run and develop MaxSight.

---

## Software

### Python

- **Version:** 3.10+ (tested on 3.10, 3.12)
- **Note:** `pyproject.toml` and CI use 3.10; local env may be 3.12.

### Core dependencies (required)

From `requirements.txt`:

| Package       | Version  | Purpose                |
|---------------|----------|------------------------|
| torch         | >=2.9.1  | Core ML                |
| torchvision   | >=0.24.1 | Models, NMS            |
| torchaudio    | >=2.9.1  | Audio (optional branch)|
| numpy         | >=2.2.6  | Arrays                  |
| pandas        | >=2.3.3  | Data                   |
| pillow        | >=12.0.0 | Images                 |
| opencv-python | >=4.8.0  | Preprocessing          |
| scipy         | >=1.11.0 | Hungarian matching     |
| scikit-learn  | >=1.3.0  | Clustering (e.g. OCR)  |
| pytest        | >=9.0.1  | Tests                   |
| torchao       | >=0.14.1 | Model optimization     |
| matplotlib    | >=3.10.7 | Viz                    |
| tqdm          | >=4.66.0 | Progress                |
| flask         | >=3.0.0  | Web simulator          |
| flask-cors     | >=4.0.0  | Web simulator CORS     |

Install:

```bash
pip install -r requirements.txt
```

### Optional (for specific features)

| Package               | Purpose                    | When needed                    |
|-----------------------|----------------------------|--------------------------------|
| librosa               | MFCC / audio features      | Audio branch / T4+             |
| sentence-transformers | OCR text encoder           | OCR encoder (non-fallback)     |
| coremltools           | CoreML export (iOS)        | `export_to_coreml`            |
| psutil                | CPU % for adaptive skip    | Production rehearsal / skip    |
| faiss-cpu / faiss-gpu | Retrieval indexing         | Retrieval / knowledge augment  |
| Redis                 | Caching                    | Production deployment          |

### Environment

- **Conda (optional):** `environment.yml` – full conda env (Python 3.12, many system libs). Use if you prefer conda over pip.
- **Env vars (optional):** `.env` for deployment (see `docs/DEPLOYMENT.md`). No `.env` required for local dev or tests.

---

## Hardware

### Minimum (local dev / inference only)

- **CPU:** Any modern x86_64 or ARM64 (Apple Silicon).
- **RAM:** 8 GB (16 GB recommended for large batches or stress tests).
- **Disk:** ~5 GB for repo + venv + pip cache; more for data and checkpoints (see below).
- **GPU:** None required. Inference and tests run on CPU; optional MPS (Apple) or CUDA.

### Recommended for local dev (Apple Silicon M1/M2/M3)

- **RAM:** 16 GB+ (unified memory for MPS).
- **GPU:** MPS used if available; `torch.mps.synchronize()` used for accurate latency.
- **Use:** Forward pass, simulator, benchmarks, small fine-tuning. GradNorm weight updates run on CPU when device is MPS.
- **Not for:** Full production-scale training (use cloud GPU).

### Training (all tiers T0–T5)

- **GPU:** **Required.** CUDA GPU (e.g. cloud).
- **VRAM:** 16 GB minimum (e.g. T4); 40 GB+ recommended (A100).
- **RAM:** 32 GB minimum; 64 GB recommended.
- **Disk:** 100 GB+ free for data, checkpoints, logs (COCO ~25 GB; checkpoints and logs add more).
- **Where:** Colab, AWS (e.g. g4dn/g5/p3/p4), Paperspace, Lambda, RunPod, etc. See `QUICK_START_CLOUD.md` and `CURRENT_STATUS_AND_BLOCKERS.md`.

### Production deployment (serving)

- **CPU:** Multi-core recommended.
- **RAM:** 16 GB+.
- **GPU:** Optional but recommended for low-latency inference (CUDA or MPS).
- **Disk:** Space for model weights, logs, and any cache (e.g. Redis).
- **Network:** Only if serving remote clients.

### M3 Pro / Apple Silicon dev readiness

- **Supported** for development: inference, benchmarks, simulator, and short training runs. MPS is used when available; `torch.mps.synchronize()` ensures accurate latency; GradNorm task-weight updates run on CPU and are copied back to device.
- **Production-scale training** should use a cloud CUDA GPU; M3 Pro is for dev, small fine-tuning, and forward-pass validation.

### CoreML export (iOS)

- **Image input only.** Export via `export_to_coreml()` uses a single fixed-shape image input. Audio and temporal inputs are not yet supported; add them with fixed shapes in `ml/training/export.py` if required.

### arm64 (Apple Silicon) and multiple systems

- **Supported:** x86_64 and arm64 (Apple Silicon). Download, extraction, and split creation are platform-agnostic; inference and benchmarks run on CPU or MPS. Use `--device cpu` if MPS hits unsupported ops (e.g. some padding); use `--device mps` when supported for faster runs.
- **One data layout:** Run `scripts/gather_training_data.py` once to download (optional), extract COCO, and create train/val/test splits. Then use `--data-dir`, `--train-annotation`, `--val-annotation`, and `--image-dir` with `train_maxsight.py` (and with any Optuna tuning script) so the model and AutoML share the same data layout and requirements.

### Requirements before training (checklist)

Complete these before running `train_maxsight.py` or AutoML:

1. **Software**
   - Python 3.10+ and `pip install -r requirements.txt` (see Core dependencies above).
   - Optuna is required for `scripts/AutoMLType.py` (in `requirements.txt`).

2. **Data**
   - Run **once**: `python scripts/gather_training_data.py [--data-dir datasets/coco_raw] [--splits-dir datasets/cleaned_splits]` (use `--skip-download` / `--skip-extract` if you already have COCO).
   - This produces COCO images under `data_dir` and split JSONs under `splits_dir`: `maxsight_train.json`, `maxsight_val.json`, `maxsight_test.json`.
   - Alternatively, you can use legacy layout: `data_dir/train/` and `data_dir/val/` (each with images or annotation); then run training without `--train-annotation` / `--val-annotation`.

3. **Runtime modules**
   - The annotation-based pipeline (used when you pass `--train-annotation` / `--val-annotation`) requires `ml.utils.preprocessing.ImagePreprocessor`. If your repo excludes it via `.gitignore`, ensure that file exists in `ml/utils/` or training will fail on import.

4. **Hardware (for full training)**
   - Full training: CUDA GPU (see Training hardware above). Short or smoke runs: CPU or MPS is fine.
   - Disk: enough space for checkpoints and logs (e.g. 20–30 GB free if saving to local disk).

5. **Then run training**
   - With splits: `python scripts/train_maxsight.py --data-dir <data_dir> --train-annotation <splits_dir>/maxsight_train.json --val-annotation <splits_dir>/maxsight_val.json --image-dir <data_dir> [--epochs 100] [--device cuda]`
   - Quick check: `python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --force-cpu` (no COCO needed).

### Data requirements and gathering

- **To satisfy all data requirements for training and AutoML:** run `scripts/gather_training_data.py` (optionally with `--skip-download` / `--skip-extract` if you already have COCO). This produces:
  - COCO data under `--data-dir` (default `datasets/coco_raw`): `train2017/`, `val2017/`, `annotations/`.
  - Splits under `--splits-dir` (default `datasets/cleaned_splits`): `maxsight_train.json`, `maxsight_val.json`, `maxsight_test.json`.
- **Training:** `train_maxsight.py --data-dir <data_dir> --train-annotation <splits_dir>/maxsight_train.json --val-annotation <splits_dir>/maxsight_val.json --image-dir <data_dir> ...`
- **AutoML (Optuna):** If using a tuning script, pass the same `--data-dir`, annotation paths, and `--image-dir` so trials use the same data. Works on arm64 with `--device cpu` or `--device mps` as appropriate.

### AutoML (hyperparameter tuning)

- **Optuna** is listed in `requirements.txt` for hyperparameter tuning. Use a single tuning script (e.g. `tune_hyperparameters.py` if present) with the same data layout as above; best params are typically written to a JSON under the checkpoint dir for use with `train_maxsight.py`.

---

## Quick checks

```bash
# Python
python --version   # 3.10+

# Core deps
pip install -r requirements.txt
python -c "import torch, torchvision, PIL, cv2, scipy, sklearn; print('OK')"

# GPU (optional)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

---

## Where requirements live in the repo

| What                | File / place                          |
|---------------------|----------------------------------------|
| Pip packages        | `requirements.txt`                    |
| Conda env           | `environment.yml`                     |
| Python/type check   | `pyproject.toml`                      |
| CI (Python, install)| `.github/workflows/ci.yml`             |
| Data gathering      | `scripts/gather_training_data.py`     |
| Validation/benchmark/rehearsal | `scripts/archive/` (e.g. `validate_forward_passes.py`, `benchmark_tiers.py`, `full_production_rehearsal.py`) — run from archive if needed |
| Feature → code      | `docs/REQUIREMENTS_MAP.md`             |
| Cloud GPU setup     | `QUICK_START_CLOUD.md`                |
| Deployment          | `docs/DEPLOYMENT.md`                  |
| Blockers / GPU      | `CURRENT_STATUS_AND_BLOCKERS.md`     |

---

**Summary:** You have **software** requirements in `requirements.txt` (and optional `environment.yml`), and **hardware** requirements described in README, `CURRENT_STATUS_AND_BLOCKERS.md`, and `QUICK_START_CLOUD.md`. This file (`REQUIREMENTS.md`) pulls them into one place. For a single “requirements file” in the usual sense, use **`requirements.txt`** plus the hardware guidance above.
