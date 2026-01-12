# Dataset Setup Summary

## ✅ Completed Setup

### 1. COCO Dataset Splits (Train/Val/Test)

**Files Created:**
- `ml/data/coco_dataset_splitter.py` - COCO dataset splitting utilities
- `scripts/setup_coco_splits.py` - Setup script for COCO splits
- Updated `ml/data/generate_annotations.py` - Now supports train/val/test splits

**Features:**
- ✅ Train/Val/Test splits (default: 70%/15%/15%)
- ✅ COCO format preservation
- ✅ MaxSight format conversion
- ✅ Reproducible splits (seed-based)
- ✅ Minimum object filtering

**Usage:**
```bash
# Setup COCO splits
python scripts/setup_coco_splits.py \
    --coco_dir datasets/coco \
    --output_dir datasets/coco_splits \
    --format maxsight \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15
```

### 2. Inference Datasets for Evaluation

**Files Created:**
- `ml/data/inference_datasets.py` - Inference dataset loaders for Open Images V6, BDD100K, and ADE20K

**Features:**
- ✅ Open Images V6: Broad semantic diversity (9M images, 600 classes)
- ✅ BDD100K: Motion/outdoor/hazard realism (100K images, driving scenarios)
- ✅ ADE20K: Indoor structure & objects (20K images, 150 classes)
- ✅ Automatic resizing to 224x224
- ✅ ImageNet normalization
- ✅ Inference utilities
- ✅ Batch processing

**Usage:**
```bash
# Run inference on BDD100K (outdoor safety)
python -m ml.data.inference_datasets \
    --dataset bdd100k \
    --root datasets/bdd100k \
    --split val \
    --batch_size 32 \
    --model_path checkpoints/maxsight_best.pth

# Run inference on Open Images V6 (diversity)
python -m ml.data.inference_datasets \
    --dataset open_images_v6 \
    --root datasets/open_images_v6 \
    --split validation \
    --model_path checkpoints/maxsight_best.pth

# Run inference on ADE20K (indoor)
python -m ml.data.inference_datasets \
    --dataset ade20k \
    --root datasets/ade20k \
    --split validation \
    --model_path checkpoints/maxsight_best.pth
```

## Default Split Ratios

- **Training**: 70% (0.7)
- **Validation**: 15% (0.15)
- **Testing**: 15% (0.15)

## Integration with Training

```python
from pathlib import Path
from ml.data.dataset import MaxSightDataset
from torch.utils.data import DataLoader

# Training
train_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_train.json'),
    image_dir=Path('datasets/coco/train2017')
)

# Validation
val_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_val.json'),
    image_dir=Path('datasets/coco/train2017')
)

# Testing
test_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_test.json'),
    image_dir=Path('datasets/coco/train2017')
)
```

## Next Steps

1. **Download COCO Dataset** (if not already done)
2. **Run setup script** to create splits
3. **Train model** using train/val splits
4. **Evaluate** on test split
5. **Run inference** on evaluation datasets:
   - Open Images V6 for diversity testing
   - BDD100K for outdoor safety testing
   - ADE20K for indoor navigation testing

See `docs/COCO_INFERENCE_DATASETS_SETUP.md` and `docs/INFERENCE_DATASETS_SETUP.md` for detailed instructions.

