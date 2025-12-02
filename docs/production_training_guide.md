# Production Training Guide - MaxSight CNN

## Overview

This guide covers the complete production training pipeline for MaxSight CNN, from FP32 training through quantization to deployment.

## Architecture

```
MaxSight CNN Architecture:
├── ResNet50 Backbone (pretrained ImageNet)
├── Simplified FPN (Feature Pyramid Network)
├── Multi-Head Detection
│   ├── Classification Head (48 classes)
│   ├── Bounding Box Head (center format)
│   ├── Objectness Head
│   ├── Scene Embedding Head
│   ├── Urgency Head (4 levels)
│   └── Distance Zone Head (3 zones)
└── Audio Branch (optional, 128-dim MFCC)
```

## Sprint 1: FP32 Training

### Step 1: Prepare Data

```bash
# Organize your dataset
datasets/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

### Step 2: Train FP32 Model

```bash
python scripts/train_maxsight.py \
    --data-dir datasets/ \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --device cuda \
    --checkpoint-dir checkpoints \
    --num-classes 48
```

### Step 3: Validate Training

Check training history:
```bash
cat checkpoints/training_history.json
```

Load best model:
```python
import torch
from ml.models.maxsight_cnn import create_model

model = create_model(num_classes=48)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Sprint 2: Quantization Pipeline

### Step 1: Post-Training Quantization (PTQ)

First, try static PTQ:

```python
from ml.training.quantization import quantize_model_int8

# Load FP32 model
model_fp32 = create_model(num_classes=48)
checkpoint = torch.load('checkpoints/best_model.pt')
model_fp32.load_state_dict(checkpoint['model_state_dict'])

# Quantize
model_int8 = quantize_model_int8(
    model=model_fp32,
    calibration_data=calibration_loader,  # 200 diverse samples
    backend='qnnpack'  # For iOS/ARM
)
```

### Step 2: Validate PTQ Results

```bash
python tools/quantization/validate_and_bench.py \
    --fp32-model checkpoints/best_model.pt \
    --int8-model artifacts/ptq/model_int8.pt \
    --model-file ml/models/maxsight_cnn.py \
    --data-dir datasets/val \
    --benchmark \
    --output-file results/ptq_validation.json
```

**Decision Point:**
- If accuracy drop < 1% → **Ship PTQ model**
- If accuracy drop > 1% → **Continue to QAT**

### Step 3: Quantization-Aware Training (QAT)

If PTQ degrades accuracy, use QAT:

```bash
python tools/quantization/qat_finetune.py \
    --model-file ml/models/maxsight_cnn.py \
    --data-dir datasets/ \
    --epochs 5 \
    --lr 1e-5 \
    --batch-size 32 \
    --device cuda \
    --backend qnnpack \
    --num-classes 48 \
    --output-dir artifacts/qat
```

QAT will:
1. Fuse Conv+BN+ReLU patterns
2. Insert fake quantization modules
3. Warmup: observers ON, fake quant OFF (1 epoch)
4. Full QAT: observers ON, fake quant ON (4 epochs)
5. Convert to INT8
6. Save best model

### Step 4: Validate QAT Results

```bash
python tools/quantization/validate_and_bench.py \
    --fp32-model checkpoints/best_model.pt \
    --int8-model artifacts/qat/model_int8_from_qat.pt \
    --model-file ml/models/maxsight_cnn.py \
    --data-dir datasets/val \
    --benchmark \
    --output-file results/qat_validation.json
```

## Training Loop Details

### Production Training Loop (`ml/training/train_loop.py`)

The production training loop matches the pseudo-code specification:

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = move_targets_to_device(labels, device)
        
        with autocast():  # Mixed precision
            outputs = model(images)
            loss = compute_multihead_loss(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    scheduler.step()
    validate(model, val_loader)
    save_checkpoint(model, optimizer, epoch)
```

**Features:**
- Mixed precision training (FP16)
- Multi-head loss (detection, scene, urgency, distance)
- Gradient clipping
- Learning rate scheduling
- Automatic checkpointing
- Best model tracking

### Loss Function (`ml/training/losses.py`)

`MaxSightLoss` computes:
- **Classification Loss**: Focal loss for class imbalance
- **Bounding Box Loss**: IoU loss (1 - IoU)
- **Objectness Loss**: BCE loss
- **Urgency Loss**: Cross-entropy
- **Distance Loss**: (currently disabled)

## Model Output Format

MaxSight CNN returns:

```python
{
    'classifications': [B, 196, 48],      # Per-location class logits
    'boxes': [B, 196, 4],                 # Per-location boxes (cx, cy, w, h)
    'objectness': [B, 196],                # Per-location objectness scores
    'text_regions': [B, 196],              # Per-location text probability
    'scene_embedding': [B, 256],           # Global scene embedding
    'urgency_scores': [B, 4],              # Scene-level urgency
    'distance_zones': [B, 196, 3],         # Per-location distance zones
    'num_locations': 196                   # Grid size (14x14)
}
```

## Post-Processing

Use `model.get_detections()` for structured output:

```python
detections = model.get_detections(
    images,
    confidence_threshold=0.5,
    nms_threshold=0.5,
    max_detections=10
)

# Returns: List[List[Dict]] per image
# Each detection has:
# {
#     'class': int,
#     'confidence': float,
#     'box': [x, y, w, h],
#     'distance': str,  # 'near', 'medium', 'far'
#     'urgency': int,   # 0-3
#     'is_text': bool
# }
```

## Acceptance Criteria

### Sprint 1 End
- [ ] FP32 model trained to convergence
- [ ] Validation mAP > 0.30
- [ ] Model size < 200 MB
- [ ] Inference time < 100ms on CPU

### Sprint 2 End
- [ ] INT8 model < 50 MB
- [ ] Classification accuracy drop < 1%
- [ ] Embedding cosine similarity > 0.99
- [ ] BBox IoU drop < 0.02
- [ ] Urgency accuracy drop < 1%
- [ ] TorchScript export succeeds
- [ ] ExecuTorch `.pte` file generated

## Common Issues & Solutions

### Issue: Training loss not decreasing
**Solution:**
- Check learning rate (try 1e-4 or 1e-5)
- Verify data loading (print batch shapes)
- Check loss function (print individual loss components)

### Issue: Out of memory
**Solution:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (automatic)

### Issue: QAT fusion warnings
**Solution:**
- Check model architecture matches expected patterns
- Verify Conv+BN+ReLU sequences exist
- Fusion is soft-fail (warnings only)

### Issue: Validation accuracy drop after quantization
**Solution:**
- Increase calibration data diversity
- Run QAT for more epochs
- Check per-channel quantization is enabled (qnnpack)

## Next Steps

1. **Export to TorchScript**: `ml/training/export.py`
2. **Export to ExecuTorch**: iOS deployment
3. **iOS Integration**: Load `.pte` file in Swift
4. **Real-world Testing**: Validate on device

## References

- `ml/training/train_loop.py` - Production training loop
- `ml/training/losses.py` - Multi-head loss function
- `ml/training/quantization.py` - PTQ quantization
- `tools/quantization/qat_finetune.py` - QAT fine-tuning
- `tools/quantization/validate_and_bench.py` - Validation & benchmarking
