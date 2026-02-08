# Top 7 conditions – list and commands

Single reference for the **top 7** conditions. You can use either the **fixed list** below or the **top 7 by mAP** (best validation mAP) with `--top-by-map`.

---

## 1. The top 7 (in order)

| # | Condition               | Checkpoint path (under `CHECKPOINTS_BASE`)   |
|---|-------------------------|----------------------------------------------|
| 1 | amblyopia               | `checkpoints_amblyopia/best_model.pt`        |
| 2 | amd                     | `checkpoints_amd/best_model.pt`             |
| 3 | color_blindness         | `checkpoints_color_blindness/best_model.pt`  |
| 4 | cvi                     | `checkpoints_cvi/best_model.pt`             |
| 5 | glaucoma                | `checkpoints_glaucoma/best_model.pt`         |
| 6 | retinitis_pigmentosa    | `checkpoints_retinitis_pigmentosa/best_model.pt` |
| 7 | strabismus              | `checkpoints_strabismus/best_model.pt`      |

**As a list (for scripts):**  
`amblyopia amd color_blindness cvi glaucoma retinitis_pigmentosa strabismus`

**Top 7 by mAP:** To use the seven conditions with the highest validation mAP instead of this fixed list, run inference first (so `inference_data.json` exists), then pass `--top-by-map` to deploy or inference_and_deploy. See section 3 below.

---

## 2. Set your paths (choose one)

**Local (repo or Drive):**
```bash
export REPO=/path/to/2026-Prototype
export CHECKPOINTS_BASE=/path/to/MaxSight
```

**Colab:**
```bash
export REPO=/content/2026-Prototype
export CHECKPOINTS_BASE=/content/drive/MyDrive/MaxSight
export DATA_DIR=/content/drive/MyDrive/MaxSight_Training
```

---

## 3. Commands to run from here

**A. Validate that all 7 checkpoints exist and pass one-batch inference**
```bash
cd $REPO
python scripts/deploy_top7.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --validate-only
```

**B. Train the top 7** (if you need to train or retrain; uses train/val splits)
```bash
cd $REPO
python scripts/train_alive_models.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --data-dir "$DATA_DIR" \
  --train-annotation "$DATA_DIR/cleaned_splits/maxsight_train.json" \
  --val-annotation "$DATA_DIR/cleaned_splits/maxsight_val.json" \
  --epochs 30 --batch-size 8
```
*(Omit `--conditions` to use the default top 7.)*

**C. Run inference / mAP (COCO-style val set)**
```bash
cd $REPO
python scripts/improve_map_all_models.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --val-annotation "$DATA_DIR/cleaned_splits/maxsight_val.json" \
  --image-dir "$DATA_DIR" \
  --conditions amblyopia amd color_blindness cvi glaucoma retinitis_pigmentosa strabismus
```

**D. Deploy: export all 7 to iOS bundles**
```bash
cd $REPO
python scripts/deploy_top7.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --output-dir "$CHECKPOINTS_BASE/exports_top7"
```
To deploy the **top 7 by mAP** (from a previous inference run): pass `--top-by-map` and `--inference-data`:
```bash
python scripts/deploy_top7.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --output-dir "$CHECKPOINTS_BASE/exports_top7" \
  --top-by-map --inference-data inference_data.json
```
Output: `$CHECKPOINTS_BASE/exports_top7/<condition>/` per condition + `manifest.json`.

**E. Inference + deploy using top 7 by mAP** (run inference over all conditions, then deploy the 7 with highest mAP)
```bash
cd $REPO
python scripts/inference_and_deploy_top7.py \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --output-dir "$CHECKPOINTS_BASE/exports_top7" \
  --val-annotation "$DATA_DIR/cleaned_splits/maxsight_val.json" \
  --image-dir "$DATA_DIR" \
  --top-by-map --max-batches 10
```
If `inference_data.json` already exists, you can omit val/image and use `--top-by-map` so deploy uses that file to pick the top 7.

**F. Run inference on Open Images + ADE20K** (when those datasets are present)
```bash
cd $REPO
python scripts/run_inference_on_inference_datasets.py \
  --datasets-dir "$DATA_DIR/datasets" \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --conditions amblyopia amd color_blindness cvi glaucoma retinitis_pigmentosa strabismus \
  --max-samples 64
```

---

## 4. Quick copy-paste (Colab)

**Use these paths on Colab (do not use `/path/to/...` – that is a placeholder):**

```bash
cd /content/2026-Prototype
# Deploy only (no inference). Checkpoints must already be on Drive.
!python scripts/inference_and_deploy_top7.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --output-dir /content/drive/MyDrive/MaxSight/exports_top7
```

With inference (if you have val data on Drive):
```bash
cd /content/2026-Prototype
!python scripts/inference_and_deploy_top7.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --output-dir /content/drive/MyDrive/MaxSight/exports_top7 \
  --val-annotation /content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json \
  --image-dir /content/drive/MyDrive/MaxSight_Training \
  --max-batches 8
```

Mount Drive first: `from google.colab import drive; drive.mount("/content/drive")`. If you omit `--checkpoints-base`, the script tries to find checkpoints (e.g. under `/content/drive/MyDrive/MaxSight`).

---

## 5. Where things live

- **Checkpoints:** `$CHECKPOINTS_BASE/checkpoints_<cond>/best_model.pt`
- **Exports (after deploy):** `$CHECKPOINTS_BASE/exports_top7/<cond>/` (PTE, configs, README)
- **Manifest:** `$CHECKPOINTS_BASE/exports_top7/manifest.json` (status of each condition)

Work from this list and these commands; the scripts use the same top 7 order and names.
