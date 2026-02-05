# Colab — copy-paste commands only

Use **Runtime → Change runtime type → GPU** first. Run cells in order.  
If you see "getcwd: cannot access parent directories": **Runtime → Restart session**, then run Cell 1 only (whole cell).

---

## Cell 1: Clone and install

```python
%cd /content
# !rm -rf /content/2026-Prototype   # uncomment only if re-cloning
!git clone -q -b feature/multimodal_refactor https://github.com/AstroSword2897/2026-Prototype.git
%cd 2026-Prototype
!pip install -q "pandas==2.2.2"
!pip install -q "numpy<2.1.0,>=1.26.0"
!pip install -q "pillow<12.0,>=8.0"
!pip install -q "torchvision>=0.24.1" "torchaudio>=2.9.1"
!pip install -q "opencv-python>=4.8.0" "scipy>=1.11.0" "scikit-learn>=1.3.0"
!pip install -q "pytest>=9.0.1" "optuna>=3.0.0" "torchao>=0.14.1"
!pip install -q "matplotlib>=3.10.7" "tqdm>=4.66.0" "flask>=3.0.0" "flask-cors>=4.0.0"
```

---

## Cell 2: Mount Drive + paths

```python
from google.colab import drive
drive.mount("/content/drive")
DATA_DIR = "/content/drive/MyDrive/MaxSight/datasets/coco_raw"
SPLITS_DIR = "/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits"
CHECKPOINT_DIR = "/content/drive/MyDrive/MaxSight/checkpoints"
EXPORT_DIR = "/content/drive/MyDrive/MaxSight/exports"
!mkdir -p "$CHECKPOINT_DIR" "$EXPORT_DIR"
```

---

## Cell 2b: Find your Drive layout (run after Cell 2)

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
        print(f"\n=== MyDrive/{candidate}/ ===")
        for name in sorted(os.listdir(path))[:25]:
            print(f"  {name}")
        break
```

Create default folders if needed:
```python
!mkdir -p "/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits"
!mkdir -p "/content/drive/MyDrive/MaxSight/checkpoints"
!mkdir -p "/content/drive/MyDrive/MaxSight/exports"
!ls -la "/content/drive/MyDrive/MaxSight/"
```

---

## Cell 3: (Optional) Cleanup checkpoints

```python
!python scripts/cleanup_cloud_checkpoints.py --checkpoint-dir checkpoints --execute
```

---

## Cell 4: Verify GPU

```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
!python scripts/test_systems_comprehensive.py --test all
```

---

## Cell 5a: Train (data in repo)

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

---

## Cell 5b: T5 Fast Training (~4 h, Drive)

```python
%cd /content/2026-Prototype
!python scripts/train_t5_fast_colab.py \
  --data-dir "$DATA_DIR" \
  --train-annotation "$SPLITS_DIR/maxsight_train.json" \
  --val-annotation "$SPLITS_DIR/maxsight_val.json" \
  --image-dir "$DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs 55 --warmup-epochs 5 --batch-size 8 --grad-accumulation-steps 4 \
  --train-fraction 0.08 --checkpoint-interval 1 --device cuda
```

---

## Cell 5b resume (after disconnect)

```python
%cd /content/2026-Prototype
!python scripts/train_t5_fast_colab.py \
  --data-dir "$DATA_DIR" \
  --train-annotation "$SPLITS_DIR/maxsight_train.json" \
  --val-annotation "$SPLITS_DIR/maxsight_val.json" \
  --image-dir "$DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs 55 --warmup-epochs 5 --batch-size 8 --grad-accumulation-steps 4 \
  --train-fraction 0.08 --checkpoint-interval 1 --device cuda \
  --resume-from "$CHECKPOINT_DIR/last_checkpoint.pt"
```

---

## Cell 5c: Generic training (Drive)

```python
%cd /content/2026-Prototype
!python scripts/train_maxsight.py \
  --data-dir "$DATA_DIR" \
  --train-annotation "$SPLITS_DIR/maxsight_train.json" \
  --val-annotation "$SPLITS_DIR/maxsight_val.json" \
  --image-dir "$DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs 5 --batch-size 8 --device cuda \
  --use-gradnorm --checkpoint-interval 1
```

---

## Cell 5d: One-shot production (Drive)

```python
%cd /content/2026-Prototype
!DATA_DIR="$DATA_DIR" TRAIN_ANN="$SPLITS_DIR/maxsight_train.json" VAL_ANN="$SPLITS_DIR/maxsight_val.json" IMAGE_DIR="$DATA_DIR" EPOCHS=5 BATCH_SIZE=8 DEVICE=cuda ./scripts/run_production_training.sh --no-export
```

---

## Cell 6: Smoke train

```python
%cd /content/2026-Prototype
!python scripts/smoke_train.py --device cuda
```

---

## Cell 7: GradNorm test

```python
!python -m pytest tests/test_gradnorm_integration.py -v
```

---

## Cell 8a: Export T0 (generic training) to Drive

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

---

## Cell 8b: Export T5 to Drive

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

---

## Cell 9: iOS bundle (Xcode) — use if you trained T5

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

---

## Cell 10: Zip exports for download

```python
!ls -la "$EXPORT_DIR"
!zip -r /content/maxsight_exports.zip "$EXPORT_DIR"
print("Download /content/maxsight_exports.zip from Colab Files panel (left sidebar)")
```

---

See **COLAB_RUNBOOK.md** for full descriptions and recovery.
