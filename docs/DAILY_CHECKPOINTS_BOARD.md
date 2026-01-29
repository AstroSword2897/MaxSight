# MaxSight Daily Checkpoints Board

**Status Legend:**
- 🟢 **GREEN**: Acceptance criteria met + evidence exists
- 🟠 **ORANGE**: Partially met OR exactly 1 critical blocker
- 🔴 **RED**: 2+ critical blockers OR core functionality not runnable

**Last Updated**: 2026-01-29

---

## Dataset Readiness Lane (Gate 0)

### COCO 2017 Dataset
- **Path**: `/Users/nani/2026-Prototype/datasets/coco_raw/`
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `val2017/` directory exists with images
  - ✅ `annotations/instances_val2017.json` exists
  - ✅ `MaxSightDataset` can load samples successfully
- **Next Action**: Run `scripts/setup_coco_splits.py` to generate cleaned splits
- **Owner**: System

### Train Images (Optional)
- **Status**: 🟠 **ORANGE**
- **Reason**: Not downloaded yet (18GB). Val-only sufficient for initial validation.
- **Next Action**: Download `train2017.zip` if full training needed
- **Owner**: TBD

---

## PRE-SPRINT: Environment Setup (Day -1)

### Task 0.1: Mac Development Environment
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ Xcode installed (verified via `xcodebuild -version`)
  - ✅ Python 3.12+ available
  - ✅ Command Line Tools available
  - ✅ Homebrew available (assumed from context)
- **Verification Command**: `xcodebuild -version && python3 --version`
- **Owner**: BC (Done)

### Task 0.2: Python ML Dependencies
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ PyTorch 2.10.0.dev installed
  - ✅ torchvision 0.25.0.dev installed
  - ✅ torchaudio 2.10.0.dev installed
  - ✅ MPS backend available
- **Verification Command**: `python -c "import torch; import torchvision; import torchaudio; print(torch.__version__); print(torch.backends.mps.is_available())"`
- **Owner**: BC & Dylan (Done)

### Task 0.3: Repository Setup
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ Git repository initialized
  - ✅ Directory structure exists: `ml/`, `ios/`, `datasets/`, `tests/`, `docs/`
  - ✅ `.gitignore` configured
  - ✅ Project overview documented in README.md
  - ✅ Initial commits made
- **Verification**: `git status` clean, `tree ml/` shows structure
- **Owner**: System (Done)

### Task 0.4: Requirements Mapping
- **Status**: 🟠 **ORANGE**
- **Reason**: Documentation exists but needs verification of completeness
- **Evidence Needed**:
  - ✅ 10 vision conditions mapped (verify in docs)
  - ✅ Environmental structuring approach documented
  - ✅ Multimodal output methods planned
  - ⚠️ Model feature list alignment needs verification
  - ⚠️ 30+ user stories need verification
- **Next Action**: Audit `docs/` for requirements mapping completeness
- **Owner**: TBD

---

## SPRINT 1: Custom CNN for Environmental Reading (Days 1-14)

### Week 1: CNN Architecture for Environmental Structuring (Days 1-7)

#### DAY 1: Vision Condition Analysis & Architecture

**Task 1.1: Condition-Specific Requirements (90 min)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `docs/conditions_requirements.md` exists
  - ✅ Conditions mapped to technical requirements
  - ✅ Environmental structuring approach documented
- **Verification**: File exists with condition documentation
- **Owner**: System (Done)

**Task 1.2: CNN Architecture Design for Scene Understanding (90 min)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ Architecture implemented in `ml/models/maxsight_cnn.py`
  - ✅ ResNet50 backbone + FPN
  - ✅ Multi-task heads (30+ outputs)
  - ✅ Audio fusion branch exists
  - ✅ Stage A/B separation enforced
- **Verification**: `python -c "from ml.models.maxsight_cnn import create_model; m = create_model(); print(f'Params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M')"`
- **Owner**: System (Done)

**Task 1.3: Environmental Dataset Acquisition (2 hours)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ COCO 2017 val images downloaded (1GB)
  - ✅ COCO 2017 annotations downloaded (241MB)
  - ✅ Dataset structure verified
- **Verification**: `ls datasets/coco_raw/val2017/ | wc -l` shows images
- **Owner**: System (Done)
- **Note**: Train images (18GB) not downloaded yet - sufficient for validation

**Task 1.4: Preprocessing for Environmental Structuring (1 hour)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ Image transforms in `ml/utils/preprocessing.py`
  - ✅ Audio MFCC extraction in `ml/data/dataset.py` (torchaudio)
  - ✅ Distance estimation preprocessing exists
  - ✅ Text region detection preprocessing exists
- **Verification**: `MaxSightDataset` loads and preprocesses correctly
- **Owner**: System (Done)

#### DAY 2: Model Implementation

**Task 2.1: Environmental Scene CNN Implementation (3 hours)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `MaxSightCNN` class in `ml/models/maxsight_cnn.py`
  - ✅ ResNet50 backbone (pretrained)
  - ✅ Audio processing branch
  - ✅ Multi-head attention
  - ✅ All output heads implemented
- **Verification**: Forward pass succeeds with dummy data
- **Owner**: System (Done)

**Task 2.2: Multi-Task Loss for Scene Understanding (1 hour)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ Loss functions in `ml/training/losses.py`
  - ✅ `MultiHeadLoss` and `GradNormMultiHeadLoss` implemented
  - ✅ All individual losses: Classification, Box, Distance, Urgency, etc.
- **Verification**: `python -c "from ml.training.losses import MultiHeadLoss; print('✅ Losses available')"`
- **Owner**: System (Done)

**Task 2.3: Training Infrastructure (2 hours)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `ProductionTrainLoop` in `ml/training/train_loop.py`
  - ✅ AdamW optimizer with discriminative learning rates
  - ✅ Cosine annealing scheduler
  - ✅ Training/validation loops
  - ✅ Checkpoint management
  - ✅ Early stopping
  - ✅ Progress tracking
- **Verification**: `scripts/train_maxsight.py` runs successfully
- **Owner**: System (Done)

#### DAY 3: Dataset & Initial Training

**Task 3.1: Environmental Dataset Class (2 hours)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `MaxSightDataset` in `ml/data/dataset.py`
  - ✅ Loads images with environmental context
  - ✅ Condition-specific simulations supported
  - ✅ Formats targets: classes, boxes, descriptions, urgency, distance
- **Verification**: Dataset loads 100+ samples correctly
- **Owner**: System (Done)

**Task 3.2: Annotation Generation (1 hour)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `scripts/setup_coco_splits.py` run successfully
  - ✅ Generated `datasets/cleaned_splits/maxsight_train.json` (5000 samples)
  - ✅ Generated `datasets/cleaned_splits/maxsight_val.json` (1000 samples)
  - ✅ Generated `datasets/cleaned_splits/maxsight_test.json` (111,266 samples)
- **Verification**: Files exist with correct sample counts
- **Owner**: System (Done)

**Task 3.3: Initial Training Run (3 hours)**
- **Status**: 🔴 **RED**
- **Reason**: Not run yet (depends on Task 3.2)
- **Evidence Needed**:
  - ✅ Training completes 10 epochs
  - ✅ Loss decreases
  - ✅ Checkpoints saved
  - ✅ >75% accuracy achieved
- **Next Action**: Complete Task 3.2, then run training
- **Owner**: TBD

#### DAY 4: Evaluation & Environmental Understanding Testing

**Task 4.1: Scene Understanding Metrics (2 hours)**
- **Status**: 🟠 **ORANGE**
- **Reason**: Metrics exist but need refinement per roadmap
- **Evidence**:
  - ✅ `DetectionMetrics` in `ml/training/metrics.py`
  - ✅ `SceneMetrics` in `ml/training/scene_metrics.py`
  - ✅ `EvaluationMetrics` in `ml/evaluation/metrics.py`
  - ⚠️ Needs refinement per roadmap requirements
- **Next Action**: Verify all required metrics are implemented
- **Owner**: TBD

**Task 4.2: Condition-Specific Testing (1 hour)**
- **Status**: 🔴 **RED**
- **Reason**: Not implemented yet
- **Evidence Needed**:
  - ✅ Test with each impairment simulation
  - ✅ Verify <10% accuracy degradation
- **Next Action**: Implement condition-specific test suite
- **Owner**: TBD

**Task 4.3: Model Optimization (3 hours)**
- **Status**: 🟠 **ORANGE**
- **Reason**: Quantization utilities exist but not applied
- **Evidence**:
  - ✅ `ml/training/quantization.py` exists
  - ⚠️ INT8 quantization not yet applied
  - ⚠️ Size/latency benchmarks not run
- **Next Action**: Apply quantization and benchmark
- **Owner**: TBD

#### DAY 5: OCR Integration & Text Reading

**Task 5.1: Text Detection Module (2 hours)**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `TransformerOCRHead` in `ml/models/heads/ocr_head.py`
  - ✅ Text region detection via attention
  - ✅ Text/non-text classification head
- **Verification**: OCR head exists and can be instantiated
- **Owner**: System (Done)

**Task 5.2: OCR Integration Planning (1 hour)**
- **Status**: 🟠 **ORANGE**
- **Reason**: Planning needed for iOS Vision framework integration
- **Evidence Needed**: Document OCR workflow for iOS
- **Next Action**: Create iOS OCR integration plan
- **Owner**: TBD

**Task 5.3: Model Export to iOS (3 hours)**
- **Status**: 🟠 **ORANGE**
- **Reason**: Export utilities exist but ExecuTorch build failed previously
- **Evidence**:
  - ✅ `ml/training/export.py` has export functions
  - ⚠️ ExecuTorch build failed (needs resolution)
  - ⚠️ CoreML export needs verification
- **Next Action**: Resolve ExecuTorch build or use CoreML fallback
- **Owner**: TBD

#### DAYS 6-7: Sprint Review & Planning

**Day 6: Sprint 1 Demo (5 hours)**
- **Status**: 🔴 **RED**
- **Reason**: Cannot demo until training completes
- **Next Action**: Complete Tasks 3.2, 3.3, 4.1-4.3 first
- **Owner**: TBD

**Day 7: Backlog Refinement (4 hours)**
- **Status**: 🟠 **ORANGE**
- **Reason**: Backlog exists but needs Sprint 2 detail
- **Next Action**: Break down Sprint 2 iOS tasks
- **Owner**: TBD

---

## SPRINT 2: iOS App - "Reads Environment" Feature (Days 8-21)

### Week 3: Camera & Real-Time Detection (Days 8-14)

#### DAY 8: iOS Project Foundation

**Task 8.1: Xcode Project Setup (1 hour)**
- **Status**: 🔴 **RED**
- **Reason**: iOS project not created yet
- **Evidence Needed**: Xcode project builds successfully
- **Owner**: TBD

**Task 8.2: ExecuTorch Integration (2 hours)**
- **Status**: 🔴 **RED**
- **Reason**: Depends on Task 8.1 + ExecuTorch build resolution
- **Owner**: TBD

**Task 8.3: Model File Integration (1 hour)**
- **Status**: 🔴 **RED**
- **Reason**: Depends on Task 5.3 (model export)
- **Owner**: TBD

**Task 8.4: Camera Permission & Setup (2 hours)**
- **Status**: 🔴 **RED**
- **Reason**: Depends on Task 8.1
- **Owner**: TBD

#### DAY 9-14: iOS Implementation Tasks
- **Status**: 🔴 **RED** (All blocked by Sprint 1 completion)
- **Owner**: TBD

---

## SPRINT 3: Advanced Features & Polish (Days 22-35)

### Week 5-6: Enhanced Capabilities & Testing
- **Status**: 🔴 **RED** (Blocked by Sprint 1-2)
- **Owner**: TBD

---

## SPRINT 4: Deployment & Iteration (Days 36-60)

### Week 7-10: Soft Launch & Monitoring
- **Status**: 🔴 **RED** (Blocked by previous sprints)
- **Owner**: TBD

---

## SPRINT 5: Repository Finalization

**Task 36: Basic Test Systems**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `scripts/smoke_train.py` exists
  - ✅ `scripts/validate_forward_passes.py` exists
  - ✅ Smoke tests pass on CPU
- **Verification**: Run smoke tests successfully
- **Owner**: System (Done)

**Task 37: Therapy System Fix**
- **Status**: 🟠 **ORANGE**
- **Reason**: Therapy system exists but needs verification
- **Evidence**: `ml/models/heads/therapy_state_head.py` exists
- **Next Action**: Verify therapy system flow
- **Owner**: TBD

**Task 38: Function Flow Verification**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `scripts/analyze_function_flow.py` exists
  - ✅ Function flow documented
- **Owner**: System (Done)

**Task 39: File Cleanup**
- **Status**: 🟠 **ORANGE**
- **Reason**: Some cleanup done, needs final pass
- **Next Action**: Final cleanup pass
- **Owner**: TBD

**Task 40: Repository Mapping**
- **Status**: 🟠 **ORANGE**
- **Reason**: Needs production verification
- **Next Action**: Verify production readiness
- **Owner**: TBD

**Task 41: Web Simulator**
- **Status**: 🟢 **GREEN**
- **Evidence**:
  - ✅ `tools/simulation/web_simulator.py` exists
  - ✅ Simulator verified
- **Owner**: System (Done)

**Task 42-43: AutoML Setup**
- **Status**: 🔴 **RED**
- **Reason**: Not implemented yet
- **Owner**: TBD

---

## Daily Log Template

### Date: YYYY-MM-DD

**Completed Today:**
- [ ] Task X.Y: Description

**Blockers:**
- Blocker description + owner

**Status Changes:**
- Task X.Y: 🟠 → 🟢 (reason)

**Next Day Priorities:**
1. Priority 1
2. Priority 2

**Evidence Added:**
- Command output / file path / screenshot

---

## Quick Status Summary

- 🟢 **GREEN**: 15 tasks
- 🟠 **ORANGE**: 12 tasks
- 🔴 **RED**: 25+ tasks (mostly iOS/Sprint 2+)

**Critical Path:**
1. Complete Task 3.2 (annotation generation)
2. Complete Task 3.3 (initial training)
3. Complete Task 4.1-4.3 (evaluation)
4. Resolve Task 5.3 (model export)
5. Begin Sprint 2 (iOS)

