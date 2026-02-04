# Inference Datasets Setup for MaxSight

This document explains how to set up and use the three inference datasets optimized for MaxSight evaluation.

## Overview

MaxSight uses three specialized datasets for inference testing, each covering different aspects of assistive vision:

| Dataset | What it covers | Use case |
|---------|---------------|----------|
| **Open Images V6** | Broad semantic diversity | General object detection, diverse scenes |
| **BDD100K** | Motion / outdoor / hazard realism | Navigation, outdoor safety, vehicle detection |
| **ADE20K** | Indoor structure & objects | Indoor navigation, furniture, room understanding |

## Dataset Details

### 1. Open Images V6

**Coverage:** Broad semantic diversity
- 9M images with 600 object classes
- Diverse scenes, objects, and contexts
- Real-world complexity

**Download:**
```bash
# Download from: https://storage.googleapis.com/openimages/web/index.html
# Structure:
datasets/open_images_v6/
├── validation/
│   ├── [subdirectories by image ID prefix]/
│   │   └── *.jpg
└── validation-annotations-bbox.csv
```

**Usage:**
```python
from ml.data.inference_datasets import create_inference_dataloader

dataloader = create_inference_dataloader(
    dataset_name='open_images_v6',
    root=Path('datasets/open_images_v6'),
    split='validation',
    batch_size=32
)
```

### 2. BDD100K

**Coverage:** Motion / outdoor / hazard realism
- 100K images with driving scenarios
- Outdoor scenes, vehicles, pedestrians
- Real-world hazards and motion
- Weather and scene attributes

**Download:**
```bash
# Download from: https://bdd-data.berkeley.edu/
# Structure:
datasets/bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       ├── val/
│       └── test/
└── labels/
    ├── bdd100k_labels_images_train.json
    ├── bdd100k_labels_images_val.json
    └── bdd100k_labels_images_test.json
```

**Usage:**
```python
dataloader = create_inference_dataloader(
    dataset_name='bdd100k',
    root=Path('datasets/bdd100k'),
    split='val',
    batch_size=32
)
```

### 3. ADE20K

**Coverage:** Indoor structure & objects
- 20K images with 150 object classes
- Indoor scenes, furniture, structures
- Detailed object segmentation

**Download:**
```bash
# Download from: https://groups.csail.mit.edu/vision/datasets/ADE20K/
# Structure:
datasets/ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

**Usage:**
```python
dataloader = create_inference_dataloader(
    dataset_name='ade20k',
    root=Path('datasets/ade20k'),
    split='validation',
    batch_size=32
)
```

## Running Inference

### Command Line

```bash
# Open Images V6
python -m ml.data.inference_datasets \
    --dataset open_images_v6 \
    --root datasets/open_images_v6 \
    --split validation \
    --batch_size 32 \
    --model_path checkpoints/maxsight_best.pth \
    --device cuda \
    --max_samples 1000

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

### Python API

```python
from pathlib import Path
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

# Create dataloader
dataloader = create_inference_dataloader(
    dataset_name='bdd100k',
    root=Path('datasets/bdd100k'),
    split='val',
    batch_size=32,
    max_samples=1000
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

## Dataset-Specific Features

### Open Images V6
- **Image IDs**: Unique identifiers for each image
- **Labels**: Object class labels
- **Confidence**: Label confidence scores

### BDD100K
- **Weather attributes**: Clear, rainy, snowy, foggy, cloudy
- **Scene attributes**: Highway, residential, parking lot, etc.
- **Time of day**: Daytime, dawn/dusk, night
- **Labels**: Detailed object annotations

### ADE20K
- **Segmentation annotations**: Pixel-level object segmentation
- **Indoor focus**: Furniture, structures, room layouts
- **150 object classes**: Detailed indoor object taxonomy

## Evaluation Strategy

### 1. Open Images V6 - General Performance
Test model's ability to detect diverse objects across many scenarios.

### 2. BDD100K - Outdoor Safety
Evaluate:
- Vehicle detection
- Pedestrian detection
- Hazard identification
- Motion understanding
- Weather robustness

### 3. ADE20K - Indoor Navigation
Evaluate:
- Furniture detection
- Room structure understanding
- Indoor object recognition
- Spatial relationships

## Comparison with COCO

| Aspect | COCO | Open Images V6 | BDD100K | ADE20K |
|--------|------|----------------|---------|--------|
| **Size** | 118K train, 5K val | 9M images | 100K images | 20K images |
| **Classes** | 80 classes | 600 classes | 10 classes | 150 classes |
| **Focus** | General objects | Semantic diversity | Outdoor/driving | Indoor/structure |
| **Use case** | Training | Inference (diversity) | Inference (safety) | Inference (indoor) |

## Notes

1. **Image Resizing**: All datasets are automatically resized to 224x224 for MaxSight input
2. **Normalization**: ImageNet stats used for pretrained ResNet50 compatibility
3. **Batch Processing**: All datasets support efficient batch processing
4. **Memory Efficiency**: Use `max_samples` parameter to limit processing during development

## Troubleshooting

### Dataset Not Found
```bash
# Verify dataset structure
ls datasets/open_images_v6/validation/
ls datasets/bdd100k/images/100k/val/
ls datasets/ade20k/images/validation/
```

### Missing Annotations
- Open Images: Annotation file is optional (will scan directory if missing)
- BDD100K: Labels are optional but recommended
- ADE20K: Segmentation annotations are optional

### Download Issues
- Open Images V6: Requires manual download from Google Cloud
- BDD100K: Requires registration and download from BDD website
- ADE20K: Requires download from MIT Vision Group

## Example Workflow

```bash
# 1. Download datasets (manual)
# 2. Run inference on each dataset

# Open Images V6 - test diversity
python -m ml.data.inference_datasets \
    --dataset open_images_v6 \
    --root datasets/open_images_v6 \
    --split validation \
    --model_path checkpoints/maxsight_best.pth \
    --max_samples 1000

# BDD100K - test outdoor safety
python -m ml.data.inference_datasets \
    --dataset bdd100k \
    --root datasets/bdd100k \
    --split val \
    --model_path checkpoints/maxsight_best.pth

# ADE20K - test indoor understanding
python -m ml.data.inference_datasets \
    --dataset ade20k \
    --root datasets/ade20k \
    --split validation \
    --model_path checkpoints/maxsight_best.pth
```

