# MaxSight CNN - Sprint Roadmap

## Sprint 1: Foundation (Week 1)

### Goals
- Build MaxSight CNN architecture
- Implement training pipeline
- Train FP32 baseline model
- Validate on test set

### Tasks

#### Day 1-2: Architecture & Setup
- [x] Create MaxSightCNN model (`ml/models/maxsight_cnn.py`)
- [x] Implement FPN neck
- [x] Add multi-head detection (classification, bbox, objectness)
- [x] Add scene-level heads (urgency, distance, embedding)
- [x] Add audio branch (optional)
- [x] Create `build_model()` function

#### Day 3-4: Training Infrastructure
- [x] Create production training loop (`ml/training/train_loop.py`)
- [x] Implement MaxSightLoss (`ml/training/losses.py`)
- [x] Add mixed precision support
- [x] Add checkpointing
- [x] Create training script (`scripts/train_maxsight.py`)

#### Day 5-6: Data & Training
- [ ] Prepare datasets (COCO + accessibility classes)
- [ ] Create data loaders
- [ ] Train FP32 model (100 epochs)
- [ ] Validate on test set
- [ ] Save best checkpoint

#### Day 7: Validation & Export Prep
- [ ] Evaluate model metrics (mAP, accuracy)
- [ ] Test inference pipeline
- [ ] Prepare for Sprint 2 (quantization)

### Deliverables
- FP32 trained model (`checkpoints/best_model.pt`)
- Training history (`checkpoints/training_history.json`)
- Model evaluation metrics
- Documentation

---

## Sprint 2: Quantization & Mobile Deployment (Week 2)

### Goals
- Quantize model to INT8
- Validate quantization quality
- Export for iOS deployment
- Prepare for device testing

### Tasks

#### Day 8-9: Post-Training Quantization (PTQ)
- [x] Implement PTQ quantization (`ml/training/quantization.py`)
- [x] Create calibration data loader
- [ ] Run PTQ on FP32 model
- [ ] Validate PTQ results
- [ ] Check accuracy drop

**Decision Point:**
- If accuracy drop < 1% → Ship PTQ model
- If accuracy drop > 1% → Continue to QAT

#### Day 10-11: Quantization-Aware Training (QAT)
- [x] Create QAT training script (`tools/quantization/qat_finetune.py`)
- [x] Implement MaxSight-specific fusion patterns
- [ ] Run QAT fine-tuning (5 epochs)
- [ ] Validate QAT results
- [ ] Compare PTQ vs QAT

#### Day 12: Validation & Benchmarking
- [x] Create validation script (`tools/quantization/validate_and_bench.py`)
- [ ] Run comprehensive validation
- [ ] Benchmark latency (FP32 vs INT8)
- [ ] Generate validation report

#### Day 13-14: Export & iOS Integration
- [ ] Export to TorchScript
- [ ] Export to ExecuTorch (`.pte` file)
- [ ] Test model loading in iOS
- [ ] Validate preprocessing pipeline
- [ ] Device profiling

### Deliverables
- INT8 quantized model (`artifacts/qat/model_int8_from_qat.pt`)
- Validation report (`results/qat_validation.json`)
- TorchScript model (`artifacts/export/model_int8.pt`)
- ExecuTorch model (`artifacts/export/model_int8.pte`)
- iOS integration guide

---

## Sprint 3: iOS App Integration (Week 3)

### Goals
- Integrate model into iOS app
- Implement real-time inference
- Add TTS output
- Test on device

### Tasks

#### Day 15-16: iOS Integration
- [ ] Load ExecuTorch model in Swift
- [ ] Implement preprocessing pipeline
- [ ] Match Python preprocessing exactly
- [ ] Test model inference
- [ ] Handle model errors gracefully

#### Day 17-18: Real-time Inference
- [ ] Implement camera capture
- [ ] Frame-by-frame processing
- [ ] Optimize inference speed
- [ ] Add frame skipping for performance
- [ ] Test on multiple devices

#### Day 19-20: TTS & Output
- [ ] Integrate TTS engine
- [ ] Generate natural language descriptions
- [ ] Add urgency alerts
- [ ] Implement vibration patterns
- [ ] User testing

### Deliverables
- Working iOS app
- Real-time inference pipeline
- TTS integration
- Device performance metrics

---

## Sprint 4: Advanced Features (Week 4+)

### Goals
- Add OCR integration
- Implement audio event detection
- Add personalization
- Improve accuracy

### Tasks

#### OCR Integration
- [ ] Integrate text detection head
- [ ] Add OCR post-processing
- [ ] Test on real-world text
- [ ] Optimize for mobile

#### Audio Event Detection
- [ ] Integrate YAMNet or custom audio model
- [ ] Detect alarms, sirens, etc.
- [ ] Combine with vision predictions
- [ ] Test multimodal fusion

#### Personalization
- [ ] User-defined labels
- [ ] Custom alert mappings
- [ ] Adjustable verbosity
- [ ] Save user preferences

#### Accuracy Improvements
- [ ] Fine-tune on accessibility datasets
- [ ] Add data augmentation
- [ ] Implement active learning
- [ ] Collect user feedback

---

## File Structure

```
maxsight/
├── ml/
│   ├── models/
│   │   └── maxsight_cnn.py          # Model architecture
│   ├── training/
│   │   ├── train_loop.py            # Production training loop
│   │   ├── losses.py                # Multi-head loss
│   │   ├── quantization.py         # PTQ quantization
│   │   └── export.py                # Model export
│   └── data/
│       └── dataset.py                # Data loading
├── tools/
│   └── quantization/
│       ├── qat_finetune.py          # QAT training
│       └── validate_and_bench.py    # Validation
├── scripts/
│   └── train_maxsight.py             # Training script
├── checkpoints/                      # Model checkpoints
├── artifacts/                        # Quantized models
└── docs/                             # Documentation
```

---

## Key Metrics

### Training Metrics
- **mAP@0.5**: > 0.30 (target: 0.40)
- **Classification Accuracy**: > 0.85
- **Urgency Accuracy**: > 0.90
- **Training Time**: < 24 hours (100 epochs)

### Quantization Metrics
- **Model Size**: < 50 MB (INT8)
- **Accuracy Drop**: < 1%
- **Speedup**: > 2x (vs FP32)
- **Latency**: < 200ms (iPhone 12+)

### Deployment Metrics
- **Frame Rate**: > 5 FPS
- **Memory Usage**: < 500 MB
- **Battery Impact**: < 10% per hour
- **Crash Rate**: < 0.1%

---

## Next Steps After Sprint 4

1. **Production Deployment**
   - App Store submission
   - Beta testing program
   - User feedback collection

2. **Model Improvements**
   - Larger training datasets
   - Advanced architectures (Vision Transformers)
   - Multi-modal fusion improvements

3. **Feature Expansion**
   - Real-time navigation
   - Social features
   - Integration with assistive technologies

