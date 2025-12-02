# MaxSight CNN - Complete Quantization & Training Pipeline

## 🎯 What You Have Now

You now have a **complete, production-grade training and quantization pipeline** for MaxSight CNN:

### ✅ Core Components

1. **Production Training Loop** (`ml/training/train_loop.py`)
   - Matches your pseudo-code specification exactly
   - Multi-head loss support
   - Mixed precision training
   - Automatic checkpointing
   - Best model tracking

2. **QAT Fine-tuning** (`tools/quantization/qat_finetune.py`)
   - MaxSight-specific fusion patterns
   - Per-channel weight quantization
   - Warmup + full QAT phases
   - Automatic INT8 conversion

3. **Validation & Benchmarking** (`tools/quantization/validate_and_bench.py`)
   - Per-head metrics (classification, bbox, embedding, urgency)
   - Detailed error analysis
   - Latency benchmarking
   - JSON export for CI/CD

4. **Training Script** (`scripts/train_maxsight.py`)
   - Ready-to-use CLI
   - Full argument parsing
   - Integrated with MaxSightDataset

5. **Model Compatibility** (`ml/models/maxsight_cnn.py`)
   - `build_model()` function for quantization scripts
   - `create_model()` for general use
   - Full MaxSight architecture

---

## 🚀 Quick Start

### Sprint 1: Train FP32 Model

```bash
# 1. Prepare your data
datasets/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json

# 2. Train
python scripts/train_maxsight.py \
    --data-dir datasets/ \
    --epochs 100 \
    --batch-size 32 \
    --device cuda \
    --checkpoint-dir checkpoints

# 3. Check results
cat checkpoints/training_history.json
```

### Sprint 2: Quantize Model

```bash
# 1. Try PTQ first
python -c "
from ml.models.maxsight_cnn import create_model
from ml.training.quantization import quantize_model_int8
import torch

model = create_model(num_classes=48)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Quantize with calibration data
model_int8 = quantize_model_int8(
    model=model,
    calibration_data=calibration_loader,
    backend='qnnpack'
)
torch.save(model_int8.state_dict(), 'artifacts/ptq/model_int8.pt')
"

# 2. Validate PTQ
python tools/quantization/validate_and_bench.py \
    --fp32-model checkpoints/best_model.pt \
    --int8-model artifacts/ptq/model_int8.pt \
    --model-file ml/models/maxsight_cnn.py \
    --data-dir datasets/val \
    --benchmark

# 3. If accuracy drop > 1%, run QAT
python tools/quantization/qat_finetune.py \
    --model-file ml/models/maxsight_cnn.py \
    --data-dir datasets/ \
    --epochs 5 \
    --lr 1e-5 \
    --device cuda \
    --backend qnnpack \
    --output-dir artifacts/qat
```

---

## 📋 Training Loop (Pseudo-Code → Real Code)

Your pseudo-code:

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = move_targets_to_device(labels, device)
        
        with autocast():
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

**Implemented in:** `ml/training/train_loop.py` - **ProductionTrainLoop class**

**Usage:**
```python
from ml.training.train_loop import ProductionTrainLoop

trainer = ProductionTrainLoop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=MaxSightLoss(num_classes=48),
    device='cuda',
    num_epochs=100
)

results = trainer.train()
```

---

## 🔧 Integration Points

### 1. Model Creation

**For training:**
```python
from ml.models.maxsight_cnn import create_model
model = create_model(num_classes=48)
```

**For quantization scripts:**
```python
# Uses build_model() internally
from ml.models.maxsight_cnn import build_model
model = build_model(num_classes=48)
```

### 2. Loss Function

**MaxSightLoss** handles all heads:
- Classification (focal loss)
- Bounding box (IoU loss)
- Objectness (BCE loss)
- Urgency (CE loss)
- Distance (disabled for now)

```python
from ml.training.losses import MaxSightLoss
loss_fn = MaxSightLoss(num_classes=48)
```

### 3. Data Loading

**MaxSightDataset** returns:
```python
{
    'images': torch.Tensor,      # [B, 3, 224, 224]
    'labels': torch.Tensor,      # [B, max_objects]
    'boxes': torch.Tensor,        # [B, max_objects, 4]
    'urgency': torch.Tensor,      # [B]
    'num_objects': torch.Tensor   # [B]
}
```

---

## 📊 Validation Metrics

The validation script checks:

### Classification Head
- Prediction agreement (FP32 vs INT8)
- Logits error (MSE, MAE, SNR)

### Bounding Box Head
- BBox error (MSE, MAE)
- IoU drop (if GT available)

### Scene Embedding
- Cosine similarity (should be > 0.99)
- L2 distance

### Urgency Head
- Prediction agreement
- Accuracy drop (if GT available)

### Objectness Head
- Error metrics

### Benchmarking
- Mean latency (ms)
- P95/P99 percentiles
- Speedup vs FP32

---

## 🎯 Acceptance Criteria

### Sprint 1 (FP32 Training)
- [x] Model architecture complete
- [x] Training loop implemented
- [x] Loss function integrated
- [ ] Model trained to convergence
- [ ] Validation mAP > 0.30

### Sprint 2 (Quantization)
- [x] PTQ quantization implemented
- [x] QAT fine-tuning implemented
- [x] Validation script complete
- [ ] INT8 model < 50 MB
- [ ] Accuracy drop < 1%
- [ ] Speedup > 2x

---

## 📁 File Structure

```
2026-Prototype/
├── ml/
│   ├── models/
│   │   └── maxsight_cnn.py          # Model + build_model()
│   ├── training/
│   │   ├── train_loop.py            # Production training loop
│   │   ├── losses.py                # MaxSightLoss
│   │   └── quantization.py          # PTQ quantization
│   └── data/
│       └── dataset.py                # MaxSightDataset
├── tools/
│   └── quantization/
│       ├── qat_finetune.py          # QAT training
│       └── validate_and_bench.py   # Validation
├── scripts/
│   └── train_maxsight.py             # Training CLI
├── docs/
│   ├── production_training_guide.md # Full guide
│   └── sprint_roadmap.md            # Sprint plan
└── checkpoints/                      # Model checkpoints
```

---

## 🐛 Common Issues

### Issue: "build_model() not found"
**Solution:** Use `create_model()` or ensure `build_model()` is imported from `maxsight_cnn.py`

### Issue: QAT fusion warnings
**Solution:** Normal - fusion is soft-fail. Check that patterns match your architecture.

### Issue: Validation script can't find model outputs
**Solution:** Ensure model returns dict with keys: `classifications`, `boxes`, `objectness`, `urgency_scores`, `scene_embedding`

### Issue: Training loss not decreasing
**Solution:**
- Check learning rate (try 1e-4)
- Verify data loading
- Print individual loss components

---

## 📚 Documentation

- **Full Training Guide**: `docs/production_training_guide.md`
- **Sprint Roadmap**: `docs/sprint_roadmap.md`
- **Model Architecture**: `ml/models/maxsight_cnn.py` (docstrings)

---

## 🎉 What's Next?

1. **Train your FP32 model** using `scripts/train_maxsight.py`
2. **Run PTQ** and validate results
3. **If needed, run QAT** for better accuracy
4. **Export to TorchScript/ExecuTorch** for iOS
5. **Integrate into iOS app**

Everything is ready. Just add your data and start training!

