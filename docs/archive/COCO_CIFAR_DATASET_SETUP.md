# COCO Dataset Splits & Inference Datasets Setup

This document explains how to set up COCO dataset for training/validation/testing and inference datasets (Open Images V6, BDD100K, ADE20K) for evaluation.

## Overview

- **COCO Dataset**: Used for training, validation, and testing MaxSightCNN
- **Inference Datasets**: Used for evaluation across different scenarios:
  - **Open Images V6**: Broad semantic diversity
  - **BDD100K**: Motion / outdoor / hazard realism
  - **ADE20K**: Indoor structure & objects

## COCO Dataset Setup

### 1. Download COCO Dataset

Download COCO 2017 dataset from [COCO website](https://cocodataset.org/#download):

```bash
# Structure should be:
datasets/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   └── [images]
└── val2017/
    └── [images]
```

### 2. Create Train/Val/Test Splits

#### Option A: Using Setup Script (Recommended)

```bash
python scripts/setup_coco_splits.py \
    --coco_dir datasets/coco \
    --output_dir datasets/coco_splits \
    --format maxsight \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15 \
    --seed 42
```

#### Option B: Using Python API

```python
from pathlib import Path
from ml.data.coco_dataset_splitter import create_maxsight_splits_from_coco

train_file, val_file, test_file = create_maxsight_splits_from_coco(
    coco_annotation_file=Path('datasets/coco/annotations/instances_train2017.json'),
    image_dir=Path('datasets/coco/train2017'),
    output_dir=Path('datasets/coco_splits'),
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42
)
```

#### Option C: Using Original Function (Backward Compatible)

```python
from ml.data.generate_annotations import generate_annotations_from_coco

train_file, val_file, test_file = generate_annotations_from_coco(
    coco_annotation_file=Path('datasets/coco/annotations/instances_train2017.json'),
    image_dir=Path('datasets/coco/train2017'),
    output_file=Path('datasets/coco_splits/maxsight_annotations.json'),
    num_samples=6000,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

### 3. Use Splits in Training

```python
from pathlib import Path
from ml.data.dataset import MaxSightDataset
from torch.utils.data import DataLoader

# Training dataset
train_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_train.json'),
    image_dir=Path('datasets/coco/train2017')
)

# Validation dataset
val_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_val.json'),
    image_dir=Path('datasets/coco/train2017')
)

# Test dataset
test_dataset = MaxSightDataset(
    data_dir=Path('datasets/coco'),
    annotation_file=Path('datasets/coco_splits/maxsight_test.json'),
    image_dir=Path('datasets/coco/train2017')
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

## Inference Datasets for Evaluation

### Overview

Three specialized datasets for comprehensive evaluation:

| Dataset | Coverage | Use Case |
|---------|----------|----------|
| **Open Images V6** | Broad semantic diversity | General object detection, diverse scenes |
| **BDD100K** | Motion / outdoor / hazard realism | Navigation, outdoor safety, vehicle detection |
| **ADE20K** | Indoor structure & objects | Indoor navigation, furniture, room understanding |

### 1. Setup Inference Datasets

See `docs/INFERENCE_DATASETS_SETUP.md` for detailed setup instructions.

### 2. Run Inference

#### Option A: Using Command Line

```bash
# Open Images V6
python -m ml.data.inference_datasets \
    --dataset open_images_v6 \
    --root datasets/open_images_v6 \
    --split validation \
    --batch_size 32 \
    --model_path checkpoints/maxsight_best.pth \
    --device cuda

# BDD100K
python -m ml.data.inference_datasets \
    --dataset bdd100k \
    --root datasets/bdd100k \
    --split val \
    --batch_size 32 \
    --model_path checkpoints/maxsight_best.pth \
    --device cuda

# ADE20K
python -m ml.data.inference_datasets \
    --dataset ade20k \
    --root datasets/ade20k \
    --split validation \
    --batch_size 32 \
    --model_path checkpoints/maxsight_best.pth \
    --device cuda
```

#### Option B: Using Python API

```python
from ml.data.inference_datasets import (
    create_inference_dataloader,
    run_inference_on_dataset
)
from ml.models.maxsight_cnn import MaxSightCNN
import torch

# Load model
model = MaxSightCNN(num_classes=80, use_audio=False)
checkpoint = torch.load('checkpoints/maxsight_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dataloader (example: BDD100K)
dataloader = create_inference_dataloader(
    dataset_name='bdd100k',
    root=Path('datasets/bdd100k'),
    split='val',
    batch_size=32
)

# Run inference
results = run_inference_on_dataset(
    model=model,
    dataloader=dataloader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
)

# Print results
print(f"Total images: {results['stats']['total_images']}")
print(f"Total detections: {results['stats']['total_detections']}")
print(f"Avg detections per image: {results['stats']['avg_detections_per_image']:.2f}")
```

### 3. Dataset Details

- **Open Images V6**: 9M images, 600 classes, diverse scenes
- **BDD100K**: 100K images, driving scenarios, outdoor hazards
- **ADE20K**: 20K images, 150 classes, indoor structures
- **Image Size**: All automatically resized to 224x224 for MaxSight
- **Normalization**: ImageNet stats (for pretrained ResNet50 compatibility)

## Default Split Ratios

- **Training**: 70% (0.7)
- **Validation**: 15% (0.15)
- **Testing**: 15% (0.15)

These ratios ensure:
- Sufficient training data (70%)
- Adequate validation for hyperparameter tuning (15%)
- Independent test set for final evaluation (15%)

## Notes

1. **COCO Format**: Original COCO format preserves all metadata. MaxSight format adds environmental categories, urgency scores, and distance zones.

2. **Inference Dataset Limitations**: Inference datasets (Open Images V6, BDD100K, ADE20K) may have different class mappings than COCO. Results show detection statistics for MaxSight's object detection capabilities.

3. **Reproducibility**: All splits use random seed (default: 42) for reproducibility.

4. **Image Resizing**: All inference dataset images are automatically resized to 224x224 using bilinear interpolation to match MaxSight input size.

5. **Memory Efficiency**: For large datasets, use `num_samples` parameter to limit processing during development.

## Troubleshooting

### COCO Dataset Not Found
```bash
# Verify COCO structure
ls datasets/coco/annotations/
ls datasets/coco/train2017/ | head -5
```

### Inference Dataset Download Issues
- Open Images V6: Requires manual download from Google Cloud Storage
- BDD100K: Requires registration and download from BDD website
- ADE20K: Requires download from MIT Vision Group

See `docs/INFERENCE_DATASETS_SETUP.md` for detailed download instructions.

### Split Validation Errors
- Ensure splits sum to 1.0
- Check that annotation file is valid JSON
- Verify image directory exists

## Example Workflow

```bash
# 1. Setup COCO splits
python scripts/setup_coco_splits.py \
    --coco_dir datasets/coco \
    --output_dir datasets/coco_splits \
    --format maxsight

# 2. Train model (using train/val splits)
python ml/training/train_loop.py \
    --train_annotations datasets/coco_splits/maxsight_train.json \
    --val_annotations datasets/coco_splits/maxsight_val.json

# 3. Evaluate on test set
python ml/training/evaluation.py \
    --test_annotations datasets/coco_splits/maxsight_test.json \
    --checkpoint checkpoints/maxsight_best.pth

# 4. Run inference on evaluation datasets
python -m ml.data.inference_datasets \
    --dataset bdd100k \
    --root datasets/bdd100k \
    --split val \
    --model_path checkpoints/maxsight_best.pth
```

