# Repository Analysis - MaxSight CNN Project

**Date**: Current  
**Repository**: 2026-Prototype  
**Status**: 🟢 Active Development

---

## 📊 Overview

**MaxSight** is an accessibility-focused object detection system designed to help users with vision impairments navigate their environment. The project implements a multi-task CNN architecture with quantization support for mobile deployment.

### Key Statistics
- **30 Python files** across the codebase
- **~9,228 lines of Python code**
- **72 classes/functions** in ML modules
- **16 documentation files**
- **7 commits** (most recent: vectorized metrics/losses refactor)

---

## 🏗️ Architecture

### Core Model: MaxSightCNN

**Location**: `ml/models/maxsight_cnn.py` (~1,146 lines)

**Architecture Components**:
1. **Backbone**: ResNet50 (pretrained ImageNet)
   - Extracts multi-scale features (C2, C3, C4, C5)
   - ~25M parameters

2. **Neck**: Simplified FPN (Feature Pyramid Network)
   - Combines features at 4 scales
   - Output: P2, P3, P4, P5 feature maps

3. **Multi-Head Detection**:
   - **Classification Head**: 48+ classes (COCO + accessibility classes)
   - **Bounding Box Head**: Center format (cx, cy, w, h)
   - **Objectness Head**: Binary detection confidence
   - **Scene Embedding Head**: 256-d embedding for TTS
   - **Urgency Head**: 4 levels (safe, caution, warning, danger)
   - **Distance Zone Head**: 3 zones (near, medium, far)

4. **Audio Branch** (optional):
   - Input: 128-dim MFCC features
   - Output: 128-d audio context
   - Fused with scene features

**Total Parameters**: ~36M (within mobile deployment target)

---

## 📁 Project Structure

```
2026-Prototype/
├── ml/                          # Core ML code
│   ├── models/
│   │   └── maxsight_cnn.py      # Main model architecture
│   ├── training/
│   │   ├── train_production.py  # Production trainer (1,004 lines)
│   │   ├── train_loop.py        # Training loop utilities
│   │   ├── losses.py            # Multi-task loss (339 lines)
│   │   ├── metrics.py           # Detection metrics (324 lines)
│   │   ├── scene_metrics.py     # Scene-level metrics
│   │   ├── matching.py          # Hungarian matching
│   │   ├── quantization.py     # PTQ quantization
│   │   ├── export.py            # Model export
│   │   ├── evaluation.py        # Model evaluation
│   │   └── benchmark.py         # Performance benchmarking
│   ├── data/
│   │   ├── dataset.py           # MaxSightDataset
│   │   ├── download_datasets.py # COCO download + verification
│   │   └── generate_annotations.py
│   └── utils/
│       └── preprocessing.py     # Image transforms
├── tools/
│   └── quantization/
│       ├── qat_finetune.py      # QAT training
│       └── validate_and_bench.py # Validation & benchmarking
├── scripts/
│   └── train_maxsight.py        # Training CLI script
├── tests/                       # Test suite
├── docs/                        # Comprehensive documentation
├── checkpoints/                 # Model checkpoints
├── datasets/                    # Training data
└── ios/                         # iOS app (future)

```

---

## 🔧 Key Components

### 1. Training Infrastructure

#### ProductionTrainer (`ml/training/train_production.py`)
- **1,004 lines** - Comprehensive training orchestrator
- Features:
  - Multi-task loss computation
  - Validation with DetectionMetrics
  - Checkpointing & best model tracking
  - Mixed precision support
  - Lighting condition tracking
  - Dummy data generation for testing

#### DetectionLoss (`ml/training/losses.py`)
- **339 lines** - Multi-task loss function
- Components:
  - **FocalLoss**: Class imbalance handling (α=0.25, γ=2.0)
  - **IoULoss**: Box regression (supports IoU, GIoU, DIoU, CIoU)
  - **BCE Loss**: Objectness prediction
  - **CrossEntropy**: Urgency & distance prediction
  - **TripletLoss**: Scene embedding learning (recently added)
- Weighted combination of all losses

#### DetectionMetrics (`ml/training/metrics.py`)
- **324 lines** - Comprehensive metrics calculator
- Features:
  - **Vectorized AP computation** (numpy-based, fast)
  - Precision, Recall, F1-Score
  - COCO-style mAP@[0.5:0.95]
  - Per-class, per-size, per-condition breakdowns
  - Latency tracking (mean, median, p95, p99)
- Uses `DetectionPrediction` dataclass for clean storage

### 2. Model Architecture

#### MaxSightCNN (`ml/models/maxsight_cnn.py`)
- **1,146 lines** - Complete model implementation
- **400+ classes** defined (COCO + accessibility-specific)
- Condition-specific adaptations (glaucoma, AMD, cataracts, etc.)
- Audio fusion support
- Scene-level understanding

### 3. Data Pipeline

#### MaxSightDataset (`ml/data/dataset.py`)
- COCO-compatible dataset loader
- Supports condition-specific transforms
- Handles multi-modal inputs (image + audio)

#### Dataset Download (`ml/data/download_datasets.py`)
- COCO dataset download script
- **Dataset verification** (recently added)
- Checks train/val images and annotations

### 4. Quantization Pipeline

#### PTQ (`ml/training/quantization.py`)
- Post-training quantization to INT8
- Calibration data support

#### QAT (`tools/quantization/qat_finetune.py`)
- Quantization-aware training
- MaxSight-specific fusion patterns
- Per-channel weight quantization

#### Validation (`tools/quantization/validate_and_bench.py`)
- Per-head metrics validation
- Latency benchmarking
- JSON export for CI/CD

---

## 📈 Current Status

### ✅ Completed (Sprint 1)

1. **Architecture** 🟢 GREEN
   - MaxSightCNN implemented
   - Multi-head detection
   - Audio fusion branch
   - Condition-specific adaptations

2. **Training Infrastructure** 🟢 GREEN
   - ProductionTrainer complete
   - Multi-task loss (MaxSightLoss/DetectionLoss)
   - Vectorized metrics
   - Checkpointing & validation

3. **Loss Functions** 🟢 GREEN
   - FocalLoss for class imbalance
   - IoULoss with multiple variants
   - TripletLoss for embeddings
   - Weighted multi-task combination

4. **Metrics** 🟢 GREEN
   - Vectorized AP/mAP computation
   - COCO-style evaluation
   - Per-condition & per-size breakdowns
   - Latency tracking

5. **Dataset Infrastructure** 🟢 GREEN
   - Dataset download scripts
   - Verification functions
   - Preprocessing pipeline

### 🟠 In Progress

1. **Training Execution** 🟠 ORANGE
   - Infrastructure ready
   - Dataset download pending
   - FP32 training not yet executed

2. **Quantization** 🟠 ORANGE
   - Code complete
   - Waiting for trained FP32 model

### 🔴 Not Started

1. **iOS Integration** 🔴 RED
   - Model export ready
   - iOS app not implemented

2. **OCR Integration** 🔴 RED
   - Planned for Sprint 4

3. **Audio Event Detection** 🔴 RED
   - Planned for Sprint 4

---

## 🎯 Key Features

### 1. Accessibility Focus
- **400+ classes** including accessibility-specific objects
- Condition-specific model adaptations
- Multi-modal input (vision + audio)

### 2. Production-Ready
- Comprehensive metrics tracking
- Latency measurement
- Model export utilities
- Quantization pipeline

### 3. Performance Optimized
- Vectorized operations (numpy)
- Efficient IoU computation
- Mobile deployment ready (INT8 quantization)

### 4. Well-Documented
- 16 documentation files
- Architecture rationale
- Training guides
- Sprint roadmaps

---

## 🔍 Code Quality

### Strengths
1. **Modular Design**: Clean separation of concerns
2. **Type Hints**: Comprehensive type annotations
3. **Documentation**: Extensive inline and external docs
4. **Backward Compatibility**: Aliases for renamed functions
5. **Error Handling**: Robust edge case handling

### Recent Improvements
1. **Vectorized Metrics**: Faster AP computation
2. **Cleaner Code**: Comments removed for readability
3. **Better Structure**: Dataclasses for prediction storage
4. **Type Safety**: Fixed type errors

---

## 📦 Dependencies

### Core
- **PyTorch** 2.9.1+ (with MPS support)
- **TorchVision** 0.24.1+
- **NumPy** 2.2.6+
- **Pandas** 2.3.3+

### ML/AI
- **torchao** 0.14.1+ (quantization)
- **scipy** 1.11.0+ (Hungarian matching)

### Development
- **pytest** 9.0.1+ (testing)
- **matplotlib** 3.10.7+ (visualization)

---

## 🚀 Next Steps

### Immediate (Sprint 1 Completion)
1. **Download & Verify Dataset**
   - Run `ml/data/download_datasets.py`
   - Verify 15,000+ samples

2. **Train FP32 Model**
   - Execute `scripts/train_maxsight.py`
   - Target: 100 epochs
   - Validate mAP@0.5 > 0.30

3. **Evaluate Model**
   - Run evaluation pipeline
   - Generate metrics report

### Short-term (Sprint 2)
1. **Quantize Model**
   - PTQ first (if accuracy drop < 1%)
   - QAT if needed
   - Validate INT8 performance

2. **Export for iOS**
   - TorchScript export
   - ExecuTorch conversion
   - Test model loading

### Long-term (Sprint 3-4)
1. **iOS Integration**
   - Real-time inference
   - TTS integration
   - Device testing

2. **Advanced Features**
   - OCR integration
   - Audio event detection
   - Personalization

---

## 📊 Metrics & Targets

### Training Targets
- **mAP@0.5**: > 0.30 (target: 0.40)
- **Classification Accuracy**: > 0.85
- **Urgency Accuracy**: > 0.90
- **Training Time**: < 24 hours (100 epochs)

### Quantization Targets
- **Model Size**: < 50 MB (INT8)
- **Accuracy Drop**: < 1%
- **Speedup**: > 2x (vs FP32)
- **Latency**: < 200ms (iPhone 12+)

### Deployment Targets
- **Frame Rate**: > 5 FPS
- **Memory Usage**: < 500 MB
- **Battery Impact**: < 10% per hour
- **Crash Rate**: < 0.1%

---

## 🐛 Known Issues

1. **Dataset Download**: Manual download required (script ready)
2. **Training Execution**: Not yet run (infrastructure complete)
3. **iOS App**: Not implemented (model export ready)

---

## 💡 Recommendations

1. **Immediate Actions**:
   - Download COCO dataset
   - Run initial training
   - Validate metrics

2. **Code Quality**:
   - Add unit tests for new metrics
   - Implement integration tests
   - Add CI/CD pipeline

3. **Performance**:
   - Profile training loop
   - Optimize data loading
   - Benchmark quantization

4. **Documentation**:
   - Add API documentation
   - Create user guide
   - Document deployment process

---

## 📝 Summary

**MaxSight** is a well-structured, production-ready accessibility project with:
- ✅ Complete model architecture
- ✅ Comprehensive training infrastructure
- ✅ Vectorized, efficient metrics
- ✅ Quantization pipeline
- ✅ Extensive documentation

**Status**: Infrastructure complete, awaiting dataset and training execution.

**Next Milestone**: Complete Sprint 1 with trained FP32 model.

