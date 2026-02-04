# MaxSight 3.0 - Complete Repository Walkthrough

**Step-by-step explanation of every file, directory, and component in the codebase**

---

## Table of Contents

1. [Root Directory Structure](#1-root-directory-structure)
2. [Core ML Module (`ml/`)](#2-core-ml-module-ml)
3. [Model Components (`ml/models/`)](#3-model-components-mlmodels)
4. [Training Infrastructure (`ml/training/`)](#4-training-infrastructure-mltraining)
5. [Data Pipeline (`ml/data/`)](#5-data-pipeline-mldata)
6. [Retrieval System (`ml/retrieval/`)](#6-retrieval-system-mlretrieval)
7. [Utilities (`ml/utils/`)](#7-utilities-mlutils)
8. [Scripts (`scripts/`)](#8-scripts-scripts)
9. [Tests (`tests/`)](#9-tests-tests)
10. [Application Layer (`app/`)](#10-application-layer-app)
11. [Tools (`tools/`)](#11-tools-tools)
12. [Configuration Files](#12-configuration-files)

---

## 1. Root Directory Structure

```
2026-Prototype/
├── ml/                    # Core ML code (main codebase)
├── scripts/               # Training & validation scripts
├── tests/                 # Test suite
├── app/                   # Application layer (UI, overlays)
├── tools/                 # Development tools (simulator, etc.)
├── docs/                  # Documentation
├── checkpoints/           # Saved model checkpoints
├── datasets/              # Training data
├── exports/               # Exported models (CoreML, ONNX, etc.)
├── test_images/           # Test images for validation
├── logs/                  # Training and runtime logs
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── README.md              # Main documentation
└── .gitignore            # Git ignore rules
```

---

## 2. Core ML Module (`ml/`)

### `ml/__init__.py`
**Purpose**: Module initialization  
**What it does**: Makes `ml` a Python package, can export common classes/functions

### `ml/config.py`
**Purpose**: Global configuration settings  
**What it contains**:
- Model hyperparameters
- Training settings
- Path configurations
- Feature flags

---

## 3. Model Components (`ml/models/`)

### `ml/models/maxsight_cnn.py` ⭐ **CORE FILE**
**Purpose**: Main MaxSight CNN model (210M parameters)  
**What it contains**:
- `MaxSightCNN` class: Main model architecture
- `SimplifiedFPN`: Feature Pyramid Network
- `CapabilityTier`: Enum for tiers (T0-T5)
- `TierConfig`: Configuration for each tier
- `TierManager`: Runtime tier management
- `create_model()`: Factory function to create models

**Key Methods**:
- `__init__()`: Initializes ResNet50, FPN, all heads
- `forward()`: Complete forward pass (Stage A + Stage B)
- `_forward_stage_a_backbone()`: Stage A backbone (ResNet50+FPN)
- `_forward_stage_b_backbone()`: Stage B backbone (Hybrid/Temporal)
- `get_detections()`: Post-processes outputs to get final detections

**Lines of Code**: ~2,680 lines  
**Status**: ✅ Active, production-ready

---

### `ml/models/backbone/` - Backbone Architectures

#### `ml/models/backbone/hybrid_backbone.py`
**Purpose**: Hybrid CNN-ViT backbone for Stage B  
**What it does**:
- Combines ResNet50 (CNN) + Vision Transformer (ViT)
- Cross-layer connections between CNN and ViT
- Fusion methods: weighted, cross-attention
- Used in T2+ tiers

**Key Classes**:
- `HybridCNNViTBackbone`: Main hybrid backbone
- `SpatialAttentionPooling`: Attention-based pooling

#### `ml/models/backbone/vit_backbone.py`
**Purpose**: Vision Transformer components  
**What it contains**:
- `VisionTransformerBackbone`: Full ViT implementation
- `TransformerBlock`: Self-attention blocks
- `PatchEmbedding`: Image patch embedding
- Positional embeddings

**Used by**: Hybrid backbone (ViT path)

#### `ml/models/backbone/dynamic_conv.py`
**Purpose**: Dynamic convolution (adaptive filters)  
**What it does**:
- Adapts convolution kernels based on input
- Used in T2+ tiers for adaptive feature extraction

---

### `ml/models/heads/` - Task-Specific Output Heads

#### `ml/models/heads/therapy_state_head.py` ⭐
**Purpose**: Unified head for therapy tasks  
**What it outputs**:
- Fatigue score
- Blink rate
- Fixation stability
- Depth map
- Uncertainty
- Distance zones
- Contrast map
- Edge map

**Status**: ✅ Active, replaces separate Fatigue/Depth/Contrast heads

#### `ml/models/heads/motion_head.py`
**Purpose**: Motion tracking and optical flow  
**What it outputs**:
- Motion vectors `[B, 2, H, W]` (x, y displacement)
- Temporal features

**Used by**: Therapy state head, temporal processing

#### `ml/models/heads/ocr_head.py`
**Purpose**: Text detection and recognition  
**What it does**:
- Detects text regions
- Recognizes text content
- Uses transformer-based OCR

**Status**: ✅ Active, Tier 3

#### `ml/models/heads/scene_description_head.py`
**Purpose**: Natural language scene descriptions  
**What it outputs**:
- Text description of the scene
- Uses transformer-based generation

**Status**: ✅ Active, Tier 3

#### `ml/models/heads/sound_event_head.py`
**Purpose**: Audio event classification  
**What it outputs**:
- Sound event classes
- Directional audio information
- Spatial sound mapping

**Status**: ✅ Active, requires audio input

#### `ml/models/heads/personalization_head.py`
**Purpose**: User-specific adaptations  
**What it does**:
- Adapts model behavior to user preferences
- Learns user-specific patterns

**Status**: ✅ Active, Tier 3

#### `ml/models/heads/predictive_alert_head.py`
**Purpose**: Hazard anticipation  
**What it outputs**:
- Predictive alerts for hazards
- Navigation guidance

**Status**: ⚠️ Defined but not called in forward pass (future integration)

#### `ml/models/heads/uncertainty_head.py`
**Purpose**: Model confidence estimation  
**What it outputs**:
- Global uncertainty score
- Per-detection uncertainty

**Status**: ✅ Active, Tier 1 (safety-critical)

#### `ml/models/heads/roi_priority_head.py`
**Purpose**: Region of interest prioritization  
**What it outputs**:
- Priority scores for image regions
- Used for attention/focus

**Status**: ✅ Active, Tier 2

#### `ml/models/heads/depth_head.py`
**Purpose**: Depth estimation (legacy, replaced by therapy_state_head)  
**Status**: ⚠️ Legacy, use `therapy_state_head` instead

#### `ml/models/heads/fatigue_head.py`
**Purpose**: Fatigue detection (legacy, replaced by therapy_state_head)  
**Status**: ⚠️ Legacy, use `therapy_state_head` instead

#### `ml/models/heads/contrast_head.py`
**Purpose**: Contrast mapping (legacy, replaced by therapy_state_head)  
**Status**: ⚠️ Legacy, use `therapy_state_head` instead

---

### `ml/models/attention/` - Attention Mechanisms

#### `ml/models/attention/attention.py`
**Purpose**: Base attention implementations  
**What it contains**:
- Multi-head attention
- Self-attention
- Cross-attention

#### `ml/models/attention/cbam_attention.py`
**Purpose**: CBAM (Convolutional Block Attention Module)  
**What it does**:
- Channel attention + spatial attention
- Used in T1+ tiers

#### `ml/models/attention/cross_modal_attention.py`
**Purpose**: Cross-modal attention (vision + audio)  
**What it does**:
- Attends across vision and audio modalities
- Used in T4+ tiers

#### `ml/models/attention/cross_task_attention.py`
**Purpose**: Cross-task attention  
**What it does**:
- Shares information across different tasks
- Used in T3+ tiers

---

### `ml/models/temporal/` - Temporal Processing

#### `ml/models/temporal/temporal_encoder.py`
**Purpose**: Temporal feature encoding  
**What it contains**:
- `TemporalEncoder`: Integrates ConvLSTM + TimeSformer
- `TemporalBuffer`: Sliding window buffer
- Motion head integration
- Consistency and flicker detection

**Used by**: T5 tier (temporal modeling)

#### `ml/models/temporal/conv_lstm.py`
**Purpose**: Convolutional LSTM for temporal sequences  
**What it contains**:
- `ConvLSTMCell`: Single LSTM cell with convolutions
- `ConvLSTM`: Multi-layer ConvLSTM
- `TimeSformer`: Transformer-based temporal modeling

**Status**: ✅ Active, T5 tier

---

### `ml/models/fusion/` - Multi-Modal Fusion

#### `ml/models/fusion/multimodal_fusion.py`
**Purpose**: Multi-modal sensor fusion  
**What it contains**:
- `EnhancedAudioEncoder`: Audio feature extraction
- `SpatialSoundMapping`: Directional audio
- `HapticEmbedding`: Haptic feedback encoding
- `MultimodalFusion`: Combines vision + audio + haptic

**Status**: ✅ Active, T4+ tiers

---

### `ml/models/scene_graph/` - Scene Graph Encoding

#### `ml/models/scene_graph/scene_graph_encoder_new.py`
**Purpose**: Scene graph encoding (spatial/semantic relations)  
**What it does**:
- Extracts spatial relations (left, right, above, below)
- Extracts semantic relations (object-object relationships)
- Builds graph structure
- GNN encoding for graph embeddings

**Status**: ✅ Active, T3+ tiers

**Note**: There was a `scene_graph_encoder.py` that was deleted, now using `scene_graph_encoder_new.py`

---

### `ml/models/eye_model/` - Eye Tracking

#### `ml/models/eye_model/eye_model.py`
**Purpose**: Eye tracking and fatigue detection  
**What it does**:
- Blink detection
- Fixation tracking
- Pupil size estimation

**Status**: ⚠️ Stub implementation, not integrated into forward pass

---

### `ml/models/retrieval_heads_production.py`
**Purpose**: Production-ready multi-vector retrieval heads  
**What it does**:
- Global encoder (CLIP-based)
- Region extractor
- Patch extractor
- Depth extractor
- OCR encoder
- Audio encoder
- Scene graph encoder
- Projects all to common embedding space (256 dim)

**Status**: ✅ Active, T4+ tiers

### `ml/models/retrieval_heads.py`
**Purpose**: Legacy retrieval heads  
**Status**: ⚠️ Legacy, use `retrieval_heads_production.py` instead

---

## 4. Training Infrastructure (`ml/training/`)

### `ml/training/losses.py`
**Purpose**: Per-head loss functions  
**What it contains**:
- `ObjectDetectionLoss`: Detection loss (Focal Loss)
- `OCRLoss`: OCR loss
- `SceneDescriptionLoss`: Scene description loss
- `SoundEventLoss`: Audio event loss
- `PersonalizationLoss`: Personalization loss
- `PredictiveAlertLoss`: Alert loss
- `DepthLoss`: Depth estimation loss
- `UrgencyLoss`: Urgency classification loss
- `ContrastLoss`: Contrast mapping loss
- `FatigueLoss`: Fatigue detection loss
- `FixationStabilityLoss`: Fixation loss
- `MultiHeadLoss`: Combines all losses

**Status**: ✅ Active

### `ml/training/metrics.py`
**Purpose**: Evaluation metrics  
**What it contains**:
- mAP (mean Average Precision)
- Precision, Recall, F1
- Per-class metrics
- Accessibility-specific metrics

**Status**: ✅ Active

### `ml/training/task_balancing.py`
**Purpose**: Multi-task loss balancing  
**What it contains**:
- `GradNormMultiHeadLoss`: GradNorm algorithm
- `PCGrad`: PCGrad algorithm
- Adaptive loss weighting

**Status**: ✅ Active

### `ml/training/train_loop.py`
**Purpose**: Training loop implementation  
**What it contains**:
- `ProductionTrainLoop`: Main training loop
- Epoch management
- Checkpointing
- Logging
- Validation

**Status**: ✅ Active

### `ml/training/export.py`
**Purpose**: Model export for deployment  
**What it contains**:
- `export_to_coreml()`: CoreML export (iOS)
- `export_to_executorch()`: ExecuTorch export (mobile)
- `export_to_onnx()`: ONNX export (cross-platform)
- `export_to_jit()`: PyTorch JIT export
- `ExecutorchWrapper`: Wraps dict outputs for ExecuTorch
- `FlattenedModel`: Flattens outputs for CoreML

**Status**: ✅ Active

### `ml/training/head_losses.py`
**Purpose**: Legacy head loss functions  
**Status**: ⚠️ Legacy, use `losses.py` instead

### `ml/training/regularization.py`
**Purpose**: Regularization techniques  
**What it contains**:
- Transfer learning configs
- Freezing/unfreezing strategies
- Weight decay
- Dropout schedules

**Status**: ✅ Active

### `ml/training/self_supervised_pretrain.py`
**Purpose**: Self-supervised pretraining  
**What it contains**:
- MAE (Masked Autoencoder)
- SimCLR contrastive learning
- Knowledge distillation

**Status**: ✅ Active, Phase 5

### `ml/training/quantization.py`
**Purpose**: Model quantization  
**What it contains**:
- INT8 quantization
- Post-training quantization
- Quantization-aware training

**Status**: ✅ Active

### `ml/training/evaluation.py`
**Purpose**: Model evaluation  
**What it contains**:
- Evaluation loop
- Metric computation
- Result aggregation

**Status**: ✅ Active

### `ml/training/validation.py`
**Purpose**: Validation utilities  
**Status**: ✅ Active

### `ml/training/matching.py`
**Purpose**: Hungarian matching for detection  
**What it does**:
- Matches predictions to ground truth
- Used in loss computation

**Status**: ✅ Active

### `ml/training/scene_metrics.py`
**Purpose**: Scene-level metrics  
**Status**: ✅ Active

### `ml/training/personalization_loss.py`
**Purpose**: Personalization-specific loss  
**Status**: ✅ Active

### `ml/training/stress_tests.py`
**Purpose**: Stress testing utilities  
**Status**: ✅ Active

### `ml/training/benchmark.py`
**Purpose**: Performance benchmarking  
**Status**: ✅ Active

---

## 5. Data Pipeline (`ml/data/`)

### `ml/data/dataset.py`
**Purpose**: Main dataset loader  
**What it contains**:
- `MaxSightDataset`: PyTorch Dataset for MaxSight
- Loads COCO + accessibility data
- Condition-specific augmentations
- Multi-modal support (image + audio)

**Status**: ✅ Active

### `ml/data/create_accessibility_dataset.py`
**Purpose**: Accessibility-specific dataset creation  
**What it does**:
- Generates accessibility annotations
- Therapy-oriented labels
- Contrast sensitivity, glare risk, findability

**Status**: ✅ Active

### `ml/data/advanced_augmentation.py`
**Purpose**: Advanced data augmentation  
**What it contains**:
- Condition-specific augmentations (glaucoma, AMD, etc.)
- Multi-modal augmentation
- Synthetic impairment simulation

**Status**: ✅ Active

### `ml/data/multi_modal_augment.py`
**Purpose**: Multi-modal augmentation  
**What it does**:
- Vision + audio augmentation
- Synchronized augmentation

**Status**: ✅ Active

### `ml/data/coco_dataset_splitter.py`
**Purpose**: COCO dataset splitting  
**What it does**:
- Splits COCO into train/val/test
- Handles annotations
- Creates proper directory structure

**Status**: ✅ Active

### `ml/data/download_datasets.py`
**Purpose**: Dataset download utilities  
**What it does**:
- Downloads COCO dataset
- Downloads AudioSet
- Downloads other datasets

**Status**: ✅ Active

### `ml/data/generate_annotations.py`
**Purpose**: Annotation generation  
**What it does**:
- Generates annotations from COCO
- Creates accessibility-specific annotations

**Status**: ✅ Active

### `ml/data/inference_datasets.py`
**Purpose**: Inference dataset loaders  
**What it contains**:
- `OpenImagesV6Dataset`
- `BDD100KDataset`
- `ADE20KDataset`

**Status**: ✅ Active

### `ml/data/synthetic_scene_generator.py`
**Purpose**: Synthetic scene generation  
**What it does**:
- Generates synthetic scenes for testing
- Creates ground truth annotations

**Status**: ✅ Active

---

## 6. Retrieval System (`ml/retrieval/`)

### `ml/retrieval/encoders/` - Feature Encoders

#### `ml/retrieval/encoders/global_encoder.py`
**Purpose**: Global scene encoder (CLIP-based)  
**What it does**:
- Encodes entire image to global embedding
- Uses CLIP or fallback encoder

**Status**: ✅ Active

#### `ml/retrieval/encoders/region_extractor.py`
**Purpose**: Region-based feature extraction  
**What it does**:
- Extracts features from image regions
- Returns region embeddings + bounding boxes

**Status**: ✅ Active

#### `ml/retrieval/encoders/patch_extractor.py`
**Purpose**: Patch-level feature extraction  
**What it does**:
- Extracts features from image patches
- Fine-grained local features

**Status**: ✅ Active

#### `ml/retrieval/encoders/depth_extractor.py`
**Purpose**: Depth-based feature extraction  
**What it does**:
- Extracts depth-aware features
- Geometry-aware embeddings

**Status**: ✅ Active

#### `ml/retrieval/encoders/ocr_encoder.py`
**Purpose**: OCR text encoding  
**What it does**:
- Encodes detected text
- Uses sentence-transformers or fallback

**Status**: ✅ Active

#### `ml/retrieval/encoders/audio_encoder.py`
**Purpose**: Audio feature encoding  
**What it does**:
- Encodes audio features
- Spatial sound mapping

**Status**: ✅ Active

#### `ml/retrieval/encoders/scene_graph_encoder.py`
**Purpose**: Scene graph encoding for retrieval  
**What it does**:
- Encodes scene graph structure
- Graph embeddings

**Status**: ✅ Active

---

### `ml/retrieval/indexing/` - FAISS Indexing

#### `ml/retrieval/indexing/neural_index_builder.py`
**Purpose**: Builds FAISS indices  
**What it does**:
- Creates HNSW indices
- Creates IVF-PQ indices
- GPU/CPU support

**Status**: ✅ Active

#### `ml/retrieval/indexing/index_manager.py`
**Purpose**: Index management  
**What it does**:
- Manages multiple indices
- Index updates
- Index persistence

**Status**: ✅ Active

---

### `ml/retrieval/retrieval/` - Retrieval Algorithms

#### `ml/retrieval/retrieval/stage1_ann.py`
**Purpose**: Stage 1 approximate nearest neighbor search  
**What it does**:
- FAISS-based ANN search
- Fast retrieval

**Status**: ✅ Active

#### `ml/retrieval/retrieval/stage2_rerank.py`
**Purpose**: Stage 2 reranking  
**What it does**:
- Multi-vector reranking
- Learned reranking model

**Status**: ✅ Active

#### `ml/retrieval/retrieval/knowledge_augment.py`
**Purpose**: Knowledge-augmented retrieval  
**What it does**:
- GNN-based knowledge graph retrieval
- Semantic similarity

**Status**: ✅ Active

#### `ml/retrieval/retrieval/async_retrieval.py`
**Purpose**: Asynchronous retrieval worker  
**What it does**:
- Non-blocking retrieval
- Background thread
- Caching

**Status**: ✅ Active

#### `ml/retrieval/retrieval/concept_retrieval.py`
**Purpose**: Concept-based retrieval  
**Status**: ✅ Active

---

### `ml/retrieval/fusion/` - Retrieval Fusion

#### `ml/retrieval/fusion/attention_fusion.py`
**Purpose**: Attention-based fusion of retrieval results  
**Status**: ✅ Active

#### `ml/retrieval/fusion/meta_fusion.py`
**Purpose**: Meta-learning fusion weights  
**What it does**:
- Learns optimal fusion weights
- User-adaptive fusion

**Status**: ✅ Active, Phase 6

#### `ml/retrieval/fusion/fusion_train.py`
**Purpose**: Training fusion models  
**Status**: ✅ Active

---

### `ml/retrieval/cross_view/` - Cross-View Training

#### `ml/retrieval/cross_view/cv_training.py`
**Purpose**: Cross-view contrastive learning  
**What it does**:
- Multi-view training
- Contrastive learning

**Status**: ✅ Active, Phase 5

---

## 7. Utilities (`ml/utils/`)

### `ml/utils/preprocessing.py`
**Purpose**: Image preprocessing  
**What it contains**:
- `ImagePreprocessor`: Condition-specific preprocessing
- Normalization
- Resizing
- Augmentation

**Status**: ✅ Active

### `ml/utils/output_scheduler.py`
**Purpose**: Output scheduling and prioritization  
**What it does**:
- Prioritizes outputs by tier
- Rate limiting
- Output queuing

**Status**: ✅ Active

### `ml/utils/description_generator.py`
**Purpose**: Natural language description generation  
**Status**: ✅ Active

### `ml/utils/ocr_integration.py`
**Purpose**: OCR integration utilities  
**Status**: ✅ Active

### `ml/utils/spatial_memory.py`
**Purpose**: Spatial memory for object tracking  
**Status**: ✅ Active

### `ml/utils/path_planning.py`
**Purpose**: Navigation path planning  
**Status**: ✅ Active

### `ml/utils/user_preferences.py`
**Purpose**: User preference management  
**Status**: ✅ Active

### `ml/utils/adaptive_assistance.py`
**Purpose**: Adaptive assistance logic  
**Status**: ✅ Active

### `ml/utils/sound_processing.py`
**Purpose**: Audio processing utilities  
**Status**: ✅ Active

### `ml/utils/semantic_grouping.py`
**Purpose**: Semantic grouping of objects  
**Status**: ✅ Active

### `ml/utils/logging_config.py`
**Purpose**: Logging configuration  
**Status**: ✅ Active

### `ml/utils/error_handling.py`
**Purpose**: Error handling utilities  
**Status**: ✅ Active

### `ml/utils/monitoring.py`
**Purpose**: Performance monitoring  
**Status**: ✅ Active

### `ml/utils/schema_validator.py`
**Purpose**: Schema validation  
**Status**: ✅ Active

### `ml/utils/stress_testing.py`
**Purpose**: Stress testing utilities  
**Status**: ✅ Active

### `ml/utils/per_class_metrics.py`
**Purpose**: Per-class metric computation  
**Status**: ✅ Active

### `ml/utils/performance.py`
**Purpose**: Performance utilities  
**Status**: ✅ Active

### `ml/utils/multihead_benchmark.py`
**Purpose**: Multi-head benchmarking  
**Status**: ✅ Active

---

## 8. Scripts (`scripts/`)

### `scripts/train_maxsight.py` ⭐
**Purpose**: Main training script  
**What it does**:
- Loads dataset
- Creates model
- Sets up optimizer
- Runs training loop
- Saves checkpoints
- Evaluates model

**Usage**:
```bash
python scripts/train_maxsight.py --data-dir datasets/coco --epochs 100 --device cuda
```

**Status**: ✅ Active

### `scripts/smoke_train.py`
**Purpose**: Smoke training (proof of life)  
**What it does**:
- Minimal training (1-2 epochs)
- Tiny synthetic dataset
- Verifies gradients work
- Checks for NaNs

**Usage**:
```bash
python scripts/smoke_train.py --tier T2_HYBRID_VIT --epochs 2 --batches 5
```

**Status**: ✅ Active

### `scripts/validate_forward_passes.py`
**Purpose**: Forward pass validation  
**What it does**:
- Tests all tiers (T0-T5)
- Validates forward passes
- Measures latency
- Checks memory usage

**Usage**:
```bash
python scripts/validate_forward_passes.py
```

**Status**: ✅ Active

### `scripts/analyze_function_flow.py`
**Purpose**: Function flow analysis  
**What it does**:
- Traces forward pass
- Documents data flow
- Identifies decision points

**Status**: ✅ Active

### `scripts/benchmark_tiers.py`
**Purpose**: Performance benchmarking  
**What it does**:
- Benchmarks all tiers
- Measures latency, memory, throughput
- Compares tiers

**Status**: ✅ Active

### `scripts/analyze_forward_passes.py`
**Purpose**: Forward pass analysis (legacy)  
**Status**: ⚠️ Legacy, use `validate_forward_passes.py`

### `scripts/analyze_forward_paths.py`
**Purpose**: Forward path mapping  
**Status**: ✅ Active

### `scripts/throughput_benchmark.py`
**Purpose**: Throughput benchmarking  
**Status**: ✅ Active

### `scripts/run_stress_tests.py`
**Purpose**: Stress testing  
**Status**: ✅ Active

### `scripts/profile_integration.py`
**Purpose**: Integration profiling  
**Status**: ✅ Active

### `scripts/setup_coco_splits.py`
**Purpose**: COCO dataset splitting  
**Status**: ✅ Active

### `scripts/fix_dataset_splits.py`
**Purpose**: Fix dataset splits  
**Status**: ✅ Active

### `scripts/generate_maxsight_dataset.py`
**Purpose**: Generate MaxSight dataset  
**Status**: ✅ Active

### `scripts/generate_class_weights.py`
**Purpose**: Generate class weights  
**Status**: ✅ Active

---

## 9. Tests (`tests/`)

### `tests/test_phase0_backbone.py`
**Purpose**: Phase 0 backbone tests  
**Tests**:
- ResNet50
- FPN
- Hybrid backbone
- ViT
- Dynamic convolution
- Temporal modules

**Status**: ✅ Active

### `tests/test_phase1_fusion.py`
**Purpose**: Phase 1 fusion tests  
**Tests**:
- Audio encoder
- Spatial sound mapping
- Haptic embedding
- Multimodal fusion

**Status**: ✅ Active

### `tests/test_phase2_heads.py`
**Purpose**: Phase 2 head tests  
**Tests**:
- OCR head
- Scene description head
- Sound event head
- Personalization head
- Predictive alert head

**Status**: ✅ Active

### `tests/test_phase3_retrieval.py`
**Purpose**: Phase 3 retrieval tests  
**Tests**:
- FAISS indexing
- Two-stage retrieval
- Retrieval encoders

**Status**: ✅ Active

### `tests/test_phase4_knowledge.py`
**Purpose**: Phase 4 knowledge tests  
**Tests**:
- Scene graph encoder
- GNN encoder
- Knowledge-augmented retrieval

**Status**: ✅ Active

### `tests/test_phase5_training.py`
**Purpose**: Phase 5 training tests  
**Tests**:
- Self-supervised pretraining
- Knowledge distillation
- Data augmentation
- Continual learning
- Cross-view training

**Status**: ✅ Active

### `tests/test_all_phases.py`
**Purpose**: Master test runner  
**What it does**:
- Runs all phase tests
- Aggregates results

**Status**: ✅ Active

### `tests/test_model.py`
**Purpose**: Model integration tests  
**Status**: ✅ Active

### `tests/test_integration_constraints.py`
**Purpose**: Integration constraint tests  
**Status**: ✅ Active

### `tests/test_integration_structure.py`
**Purpose**: Integration structure tests  
**Status**: ✅ Active

### `tests/test_timing_enforcement.py`
**Purpose**: Timing enforcement tests  
**Status**: ✅ Active

### `tests/test_critical_fixes.py`
**Purpose**: Critical fix tests  
**Status**: ✅ Active

### `tests/test_error_handling.py`
**Purpose**: Error handling tests  
**Status**: ✅ Active

### `tests/test_edge_cases.py`
**Purpose**: Edge case tests  
**Status**: ✅ Active

### `tests/test_condition_specific.py`
**Purpose**: Condition-specific tests  
**Status**: ✅ Active

### `tests/test_comprehensive_system.py`
**Purpose**: Comprehensive system tests  
**Status**: ✅ Active

### `tests/test_export_validation.py`
**Purpose**: Export validation tests  
**Status**: ✅ Active

### `tests/test_gradnorm_integration.py`
**Purpose**: GradNorm integration tests  
**Status**: ✅ Active

### `tests/test_performance.py`
**Purpose**: Performance tests  
**Status**: ✅ Active

### `tests/test_multihead_benchmark.py`
**Purpose**: Multi-head benchmark tests  
**Status**: ✅ Active

### `tests/test_training_pipeline.py`
**Purpose**: Training pipeline tests  
**Status**: ✅ Active

### `tests/test_scene_graph_consistency.py`
**Purpose**: Scene graph consistency tests  
**Status**: ✅ Active

---

## 10. Application Layer (`app/`)

### `app/personal_mode.py`
**Purpose**: Personal mode manager  
**What it does**:
- Adapts behavior to user preferences
- Active scene exploration
- Predictive navigation guidance

**Status**: ✅ Active, Phase 6

### `app/ui/voice_feedback.py`
**Purpose**: Voice feedback (TTS)  
**Status**: ✅ Active

### `app/ui/haptic_feedback.py`
**Purpose**: Haptic feedback  
**Status**: ✅ Active

### `app/overlays/`
**Purpose**: Visual overlays  
**Status**: ✅ Active

---

## 11. Tools (`tools/`)

### `tools/simulation/` - Simulator

#### `tools/simulation/web_simulator.py`
**Purpose**: Web-based simulator  
**What it does**:
- Runs MaxSight in browser
- Tests all components
- Visual feedback

**Status**: ✅ Active

#### `tools/simulation/comprehensive_simulator.py`
**Purpose**: Comprehensive simulator  
**Status**: ✅ Active

#### `tools/simulation/retrieval_integration.py`
**Purpose**: Retrieval integration for simulator  
**Status**: ✅ Active, Phase 8

#### `tools/simulation/simulator/`
**Purpose**: Simulator components  
**Status**: ✅ Active

---

## 12. Configuration Files

### `requirements.txt`
**Purpose**: Python dependencies  
**What it contains**:
- torch, torchvision, torchaudio
- numpy, pandas, pillow
- opencv-python
- pytest
- flask, flask-cors
- scipy, scikit-learn
- matplotlib, tqdm

**Status**: ✅ Active

### `pyproject.toml`
**Purpose**: Project configuration  
**What it contains**:
- Pyright configuration
- Pytest configuration
- Python version

**Status**: ✅ Active

### `.gitignore`
**Purpose**: Git ignore rules  
**What it ignores**:
- `__pycache__/`
- `*.pyc`, `*.pyo`
- `venv/`, `env/`
- `checkpoints/*.pt`
- `datasets/`
- `docs/*.md`
- `*.log`
- `exports/`

**Status**: ✅ Active

### `README.md`
**Purpose**: Main documentation  
**What it contains**:
- Project overview
- Quick start
- Architecture
- Components
- Training guide
- Testing guide

**Status**: ✅ Active

---

## Summary Statistics

**Total Files**: ~186 Python files  
**Core Model**: `ml/models/maxsight_cnn.py` (2,680 lines)  
**Total Components**: 100+  
**Active Components**: 95+  
**Stub/Placeholder**: 5+  

**Key Directories**:
- `ml/models/`: 33 files (model architectures)
- `ml/training/`: 15 files (training infrastructure)
- `ml/data/`: 8 files (data pipeline)
- `ml/retrieval/`: 24 files (retrieval system)
- `ml/utils/`: 18 files (utilities)
- `scripts/`: 13 files (training/validation scripts)
- `tests/`: 22 files (test suite)

---

**Next Steps**:
- See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for code execution flow
- See [FUNCTION_FLOW_ANALYSIS.md](FUNCTION_FLOW_ANALYSIS.md) for forward pass details
- See [COMPONENTS_LOG.md](../COMPONENTS_LOG.md) for component inventory

