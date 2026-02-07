# Inference: run in the terminal

Run checkpoint inference and mAP optimization **in the terminal only**. All paths are local (no Colab).

**Goal:** Improve mAP for inference (metrics and tuning), then run full checkpoint inference when ready.

---

## Prerequisites

- **Val annotation JSON** – COCO-style with `images` (with `file_name`) and `annotations`.
- **Image directory** – Root such that `image_dir / file_name` resolves to real image files (e.g. `./datasets` if `file_name` is `coco_raw/val2017/xxx.jpg`).
- **Checkpoints base** – Directory containing `checkpoints_<condition>/best_model.pt` (e.g. `./checkpoints` or a path to your condition checkpoints).

**Where trained checkpoints live (canonical locations):**
- **Colab (after training there):** `/content/drive/MyDrive/MaxSight` — parent of `checkpoints_amblyopia/`, `checkpoints_cvi/`, etc., each with `best_model.pt`.
- **Local (Google Drive for Desktop):** e.g. `~/Google Drive/My Drive/MaxSight` or `~/Library/CloudStorage/GoogleDrive-<you>/My Drive/MaxSight`. Same layout: `checkpoints_<condition>/best_model.pt`.
- **Repo:** `./checkpoints` if you copied trained weights here.
- **Discover:** `CHECKPOINTS_BASE=$(python scripts/find_trained_checkpoints.py)` — tries env, then repo, then common Drive paths.

**Minimal checkpoints (for testing):** To run inference without trained weights, create one or all condition checkpoints with `python scripts/create_minimal_checkpoint.py`. This writes untrained `best_model.pt` files so the pipeline runs and reports mAP (values will be low until you train).

---

## 1. Discover annotation paths (optional)

If you don’t know where your val JSON lives:

```bash
cd /path/to/2026-Prototype
python scripts/run_checkpoint_inference.py --find-annotations
# Searches current directory; prints paths to .json files.

python scripts/run_checkpoint_inference.py --find-annotations /path/to/data
# Searches under /path/to/data.
```

Use one of the printed paths as `--val-annotation`.

---

## 2. Run inference (mAP + metrics per condition)

**Required args:** `--val-annotation`, `--image-dir`, `--checkpoints-base`.

**Monitor output (terminal + log file):** Run the helper script so every line is printed to the terminal and appended to a timestamped log. Use this when you want to watch results live and keep a copy for review.

```bash
./scripts/run_inference_and_monitor.sh
# With options (e.g. quick check):
./scripts/run_inference_and_monitor.sh --max-batches 2 --conditions cvi
# Custom paths via env:
VAL_ANNOTATION=/path/to/val.json IMAGE_DIR=/path/to/datasets CHECKPOINTS_BASE=/path/to/checkpoints ./scripts/run_inference_and_monitor.sh
```

Logs go to `./inference_YYYYMMDD_HHMMSS.log` (override with `LOG_FILE=/path/to/log.log`).

**Or run the Python script directly:**

**Quick check (few batches, one condition)** – verify pipeline before full run:

```bash
python scripts/run_checkpoint_inference.py \
  --val-annotation ./datasets/cleaned_splits/maxsight_val.json \
  --image-dir ./datasets \
  --checkpoints-base ./checkpoints \
  --max-batches 2 \
  --conditions cvi \
  --output inference_data.json
```

**Full run (all conditions, all val batches):**

```bash
python scripts/run_checkpoint_inference.py \
  --val-annotation ./datasets/cleaned_splits/maxsight_val.json \
  --image-dir ./datasets \
  --checkpoints-base ./checkpoints \
  --output inference_data.json
```

**Intended output:**

- **Logs:** One line per condition, e.g.  
  `cvi: mAP=0.1234 mAP@0.5=0.2345 prec=0.34 rec=0.45 F1=0.39 mean_latency_ms=12.34`
- **JSON** (`inference_data.json`):  
  `inference_data: true`, `val_annotation`, `image_dir`, `checkpoints_base`, and `results[]` with per-condition:
  - `mAP`, `mAP_50`, `mAP_75`
  - `precision`, `recall`, `f1`
  - `mean_latency_ms`, `num_images`, etc.

If a condition fails, that entry in `results` will have an `"error"` field instead of metrics.

---

## 3. Improve mAP for all models (trained checkpoints)

Find trained checkpoints, sweep confidence and NMS IoU, then run full inference with the best params. Use this to **improve mAP scores of all condition models** (no retraining).

```bash
# Shell (uses same discovery + optimize_inference + run_checkpoint_inference)
./scripts/improve_map_all_models.sh

# Python (same flow, saves improved_inference_config.json with best confidence/nms)
python scripts/improve_map_all_models.py

# Or point explicitly at trained weights (e.g. Drive mount)
CHECKPOINTS_BASE="$HOME/Google Drive/My Drive/MaxSight" ./scripts/improve_map_all_models.sh
python scripts/improve_map_all_models.py --checkpoints-base "$HOME/Google Drive/My Drive/MaxSight"

# Quick sweep: limit conditions or batches
CONDITIONS="cvi amd" MAX_BATCHES=4 ./scripts/improve_map_all_models.sh
python scripts/improve_map_all_models.py --conditions cvi amd --max-batches 4

# Skip sweep and run inference once with fixed thresholds
python scripts/improve_map_all_models.py --skip-sweep --confidence 0.05 --nms-iou 0.5
```

If you don’t set `CHECKPOINTS_BASE`, the script uses `scripts/find_trained_checkpoints.py` to look in the repo, then common Google Drive mount paths. If none are found, set `CHECKPOINTS_BASE` to the folder that contains your `checkpoints_<condition>/best_model.pt` trained weights.

---

## 4. Improve mAP (sweep only, manual run)

No retraining; only inference with different confidence and NMS thresholds:

```bash
python scripts/optimize_inference.py \
  --val-annotation ./datasets/cleaned_splits/maxsight_val.json \
  --image-dir ./datasets \
  --checkpoints-base ./checkpoints
```

Defaults (if you run from repo root and have `datasets/` and `checkpoints/`):

- `--val-annotation` → `./datasets/cleaned_splits/maxsight_val.json`
- `--image-dir` → `./datasets`
- `--checkpoints-base` → `./checkpoints`

**Faster sweep:** one condition, capped batches:

```bash
python scripts/optimize_inference.py --conditions cvi --max-batches 4
```

**Intended output:**

- For each (confidence, nms_iou) pair: log line and `mAP@0.5: 0.xxxx`.
- At the end: **BEST RESULT** with best `mAP@0.5`, `confidence`, `nms_iou`, and a suggested command with `--confidence` and `--nms-iou` for `run_checkpoint_inference.py`.

---

## 5. Optional flags (both scripts)

| Flag | Meaning |
|------|--------|
| `--device cuda` or `cpu` | Device for inference (default: cuda if available). |
| `--batch-size 32` | Val batch size. |
| `--tier T5_TEMPORAL` | Model tier (must match trained checkpoints). |
| `--confidence 0.05` or `auto` | Detection confidence threshold. |
| `--nms-iou 0.5` | NMS IoU threshold. |
| `--eval-class-id 0` or `-1` | Remap predicted class for evaluation; `-1` disables. |

`run_checkpoint_inference.py` only:

| Flag | Meaning |
|------|--------|
| `--output path.json` | Output JSON path (default: `inference_data.json`). |
| `--conditions cvi amd ...` | Run only these conditions. |
| `--max-batches N` | Cap val batches per condition (for quick runs). |
| `--diagnose` | Log objectness stats for first batch. |

---

## 6. Run inference for all models in Colab

To run inference for **all condition models** in Google Colab (checkpoints on Drive, optional sweep for best mAP):

**1. Mount Drive and go to repo**
```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content
# !rm -rf /content/2026-Prototype   # if re-running and need fresh clone
!git clone -q -b feature/multimodal_refactor https://github.com/AstroSword2897/2026-Prototype.git
%cd 2026-Prototype
!pip install -q -r requirements_colab.txt
```

**2. Paths (adjust if your layout differs)**  
- Checkpoints base: `/content/drive/MyDrive/MaxSight` (folder containing `checkpoints_amblyopia/`, `checkpoints_cvi/`, … each with `best_model.pt`).  
- **Val JSON and image root** depend on your Drive layout:
  - **Cleaned MaxSight splits:**  
    `--val-annotation .../datasets/coco_raw/cleaned_splits/maxsight_val.json` and `--image-dir .../datasets` (so `image_dir` + annotation `file_name` = image path).
  - **Raw COCO layout** (only `coco_raw/annotations/`, `val2017/`, `train2017/`):  
    Use COCO val file and coco_raw as image root:  
    `--val-annotation /content/drive/MyDrive/MaxSight/datasets/coco_raw/annotations/instances_val2017.json`  
    `--image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw`  
    (COCO `file_name` is like `val2017/xxx.jpg`, so image_dir must be the folder that contains `val2017/`.)

**3. All models – full run (sweep + inference with best params)**  
Use **raw COCO** paths if you don’t have `cleaned_splits/maxsight_val.json`:

```python
# Raw COCO layout (annotations/instances_val2017.json, val2017/, train2017/)
!python scripts/improve_map_all_models.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --val-annotation /content/drive/MyDrive/MaxSight/datasets/coco_raw/annotations/instances_val2017.json \
  --image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --output /content/drive/MyDrive/MaxSight/inference_data.json
```

If you have cleaned splits instead, use `--val-annotation .../cleaned_splits/maxsight_val.json` and `--image-dir .../datasets`.

**4. All models – single inference (no sweep, faster)**  
```python
!python scripts/improve_map_all_models.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --val-annotation /content/drive/MyDrive/MaxSight/datasets/coco_raw/annotations/instances_val2017.json \
  --image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --output /content/drive/MyDrive/MaxSight/inference_data.json \
  --skip-sweep --confidence 0.05 --nms-iou 0.5
```

**5. Limit conditions or batches (optional)**  
```python
!python scripts/improve_map_all_models.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --val-annotation /content/drive/MyDrive/MaxSight/datasets/coco_raw/annotations/instances_val2017.json \
  --image-dir /content/drive/MyDrive/MaxSight/datasets/coco_raw \
  --conditions cvi amd --max-batches 4 --skip-sweep
```

Results: per-condition metrics in `inference_data.json`; with sweep, best confidence/nms in `improved_inference_config.json`. Copy paths from step 2 if your Drive layout is different.

### Getting to mAP@0.5 when you see 0

If inference reports mAP@0.5 = 0:

1. **Diagnose objectness** (see if the model outputs any scores above threshold):
   ```python
   !python scripts/run_checkpoint_inference.py \
     --val-annotation /content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json \
     --image-dir /content/drive/MyDrive/MaxSight_Training \
     --checkpoints-base /content/drive/MyDrive/MaxSight \
     --confidence 0.01 --diagnose --max-batches 2
   ```
   Check the log: if objectness max/p95 are below 0.05, a higher threshold filters everything out.

2. **Sweep confidence and NMS** (finds best threshold without retraining; now includes 0.005):
   ```python
   !python scripts/improve_map_all_models.py \
     --checkpoints-base /content/drive/MyDrive/MaxSight \
     --val-annotation /content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json \
     --image-dir /content/drive/MyDrive/MaxSight_Training \
     --output /content/drive/MyDrive/MaxSight/inference_data.json
   ```
   Omit `--skip-sweep` so it tries multiple confidence (0.3 down to 0.005) and NMS values, then runs full inference with the best. Use `--max-batches 20` for a quicker sweep.

3. **Try adaptive confidence** (per-batch threshold from model scores):
   ```python
   !python scripts/run_checkpoint_inference.py \
     --val-annotation /content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json \
     --image-dir /content/drive/MyDrive/MaxSight_Training \
     --checkpoints-base /content/drive/MyDrive/MaxSight \
     --confidence auto --output /content/drive/MyDrive/MaxSight/inference_data.json
   ```

4. **If mAP stays 0:** Checkpoints may be **untrained** (e.g. from `create_minimal_checkpoint.py`) or trained on different data. Reaching mAP@0.5 then requires **training** (e.g. `scripts/train_maxsight.py` or `scripts/train_t5_fast_colab.py`) on your val/train splits, then re-running inference.

---

## 7. Data on another disk (e.g. mounted Drive)

Use local paths only. If your data is on a mounted volume (e.g. Google Drive for Desktop), use that path:

```bash
python scripts/run_checkpoint_inference.py \
  --val-annotation "/path/to/your/val.json" \
  --image-dir "/path/to/your/datasets" \
  --checkpoints-base "/path/to/your/checkpoints" \
  --output inference_data.json
```
