# MaxSight 3.0 Components Log

**Generated**: $(date)  
**Purpose**: Complete inventory of all components in the MaxSight 3.0 codebase

---

## Core Model Components

### Main Model
- **MaxSightCNN** (`ml/models/maxsight_cnn.py`)
  - Parameters: ~210M (T2_HYBRID_VIT tier)
  - Two-stage inference: Stage A (ResNet50+FPN) + Stage B (Hybrid/Temporal)
  - 20+ task-specific heads organized by criticality tiers

### Backbone Components
- **ResNet50+FPN** (`ml/models/backbone/`)
  - Stage A backbone (always used)
  - Multi-scale feature extraction
- **Hybrid CNN-ViT** (`ml/models/backbone/hybrid_backbone.py`)
  - Stage B enhancement (T2+)
  - Combines CNN efficiency with ViT attention
- **Vision Transformer** (`ml/models/backbone/vit_backbone.py`)
  - ViT components for hybrid backbone
- **Dynamic Convolution** (`ml/models/backbone/dynamic_conv.py`)
  - Adaptive convolution for condition-specific processing

### Head Components (20+)

#### Tier 1: Safety-Critical (Never Disabled)
- **Objectness Head** (built into MaxSightCNN)
- **Classification Head** (built into MaxSightCNN)
- **Box Regression Head** (built into MaxSightCNN)
- **Distance Zone Head** (built into MaxSightCNN)
- **Urgency Head** (built into MaxSightCNN)
- **Uncertainty Head** (`ml/models/heads/uncertainty_head.py`)
  - Global Confidence Aggregator
  - System-level uncertainty aggregation

#### Tier 2: Navigation & Context
- **Motion Head** (`ml/models/heads/motion_head.py`)
  - Temporal motion tracking
  - Optical flow estimation
- **Therapy State Head** (`ml/models/heads/therapy_state_head.py`)
  - Unified head for fatigue, depth, contrast
  - Combines FatigueHead, DepthHead, ContrastMapHead
- **ROI Priority Head** (`ml/models/heads/roi_priority_head.py`)
  - Region-of-interest prioritization
- **Findability Head** (built into MaxSightCNN)
  - Object findability scores
- **Navigation Difficulty Head** (built into MaxSightCNN)
  - Scene complexity assessment

#### Tier 3: Enhancement & Therapy
- **OCR Head** (`ml/models/heads/ocr_head.py`)
  - Text detection and recognition
  - Transformer-based OCR
- **Scene Description Head** (`ml/models/heads/scene_description_head.py`)
  - Natural language scene descriptions
  - Transformer-based generation
- **Sound Event Head** (`ml/models/heads/sound_event_head.py`)
  - Audio classification
  - Directional sound detection
- **Personalization Head** (`ml/models/heads/personalization_head.py`)
  - User-specific adaptations
  - User embedding learning
- **Predictive Alert Head** (`ml/models/heads/predictive_alert_head.py`)
  - Hazard anticipation
  - Navigation guidance
  - **Status**: Defined but not called in forward pass (future integration)
- **Depth Head** (`ml/models/heads/depth_head.py`)
  - Depth estimation with uncertainty
  - Integrated into TherapyStateHead
- **Fatigue Head** (`ml/models/heads/fatigue_head.py`)
  - User fatigue detection
  - Integrated into TherapyStateHead
- **Contrast Head** (`ml/models/heads/contrast_head.py`)
  - Contrast mapping
  - Integrated into TherapyStateHead

### Temporal Processing
- **Temporal Encoder** (`ml/models/temporal/temporal_encoder.py`)
  - ConvLSTM + TimeSformer integration
  - Motion features, temporal consistency, flicker detection
- **ConvLSTM** (`ml/models/temporal/conv_lstm.py`)
  - Multi-layer temporal processing
  - Fixed stacking bug
- **TimeSformer** (`ml/models/temporal/temporal_encoder.py`)
  - Long-range temporal dependencies
  - Divided space-time attention

### Scene Graph & Retrieval
- **Scene Graph Encoder** (`ml/models/scene_graph/scene_graph_encoder.py`)
  - Batched spatial/semantic relations
  - Vectorized pairwise operations
  - MPS-stable mode
- **GNN Encoder** (`ml/models/scene_graph/scene_graph_encoder.py`)
  - Graph neural network encoding
  - Message passing with edge attributes
- **Retrieval Heads** (`ml/models/retrieval_heads_production.py`)
  - Multi-vector retrieval
  - Common embedding space (256-dim)
- **Async Retrieval** (`ml/retrieval/retrieval/async_retrieval.py`)
  - Non-blocking retrieval worker
  - Thread-based async processing

### Multi-Modal Fusion
- **Enhanced Audio Encoder** (`ml/models/fusion/multimodal_fusion.py`)
  - Audio feature extraction
  - Spatial sound mapping
- **Spatial Sound Mapping** (`ml/models/fusion/multimodal_fusion.py`)
  - Directional audio attention
- **Haptic Embedding** (`ml/models/fusion/multimodal_fusion.py`)
  - Haptic feedback encoding
- **Multimodal Fusion** (`ml/models/fusion/multimodal_fusion.py`)
  - Cross-modal attention fusion

### Attention Modules
- **CBAM Attention** (`ml/models/attention/cbam_attention.py`)
  - Channel and spatial attention
- **Cross-Modal Attention** (`ml/models/attention/cross_modal_attention.py`)
  - Vision-audio attention
- **Cross-Task Attention** (`ml/models/attention/cross_task_attention.py`)
  - Task-specific attention

---

## Training Infrastructure

### Loss Functions
- **ObjectDetectionLoss** (`ml/training/losses.py`)
- **OCRLoss** (`ml/training/losses.py`)
- **SceneDescriptionLoss** (`ml/training/losses.py`)
- **SoundEventLoss** (`ml/training/losses.py`)
- **PersonalizationLoss** (`ml/training/losses.py`)
- **PredictiveAlertLoss** (`ml/training/losses.py`)
- **DepthLoss** (`ml/training/losses.py`)
- **UrgencyLoss** (`ml/training/losses.py`)
- **ContrastLoss** (`ml/training/losses.py`)
- **FatigueLoss** (`ml/training/losses.py`)
- **FixationStabilityLoss** (`ml/training/losses.py`)

### Task Balancing
- **GradNormMultiHeadLoss** (`ml/training/task_balancing.py`)
  - Adaptive loss balancing
  - Prevents gradient warfare
- **PCGrad** (`ml/training/task_balancing.py`)
  - Projected conflicting gradients

### Metrics
- **DetectionMetrics** (`ml/training/metrics.py`)
  - mAP, precision, recall
- **EvaluationMetrics** (`ml/evaluation/metrics.py`)
  - Multi-modal metrics
  - Accessibility-specific metrics
  - Robustness evaluation

### Export
- **Export Functions** (`ml/training/export.py`)
  - CoreML export
  - ExecuTorch export
  - ONNX export
  - JIT export

---

## Data & Augmentation

### Datasets
- **MaxSightDataset** (`ml/data/dataset.py`)
  - COCO + accessibility data
  - Multi-modal loading
- **Advanced Augmentation** (`ml/data/advanced_augmentation.py`)
  - Multi-modal augmentation
  - Condition-specific augmentations
- **Multi-Modal Augment** (`ml/data/multi_modal_augment.py`)
  - Vision + audio augmentation

### Data Utilities
- **COCO Dataset Splitter** (`ml/data/coco_dataset_splitter.py`)
- **Download Datasets** (`ml/data/download_datasets.py`)
- **Generate Annotations** (`ml/data/generate_annotations.py`)
- **Synthetic Scene Generator** (`ml/data/synthetic_scene_generator.py`)
- **Inference Datasets** (`ml/data/inference_datasets.py`)

---

## Retrieval System

### Encoders
- **Global Encoder** (`ml/retrieval/encoders/global_encoder.py`)
  - CLIP/DINOv2 for scene-level embeddings
- **Region Extractor** (`ml/retrieval/encoders/region_extractor.py`)
  - Object-centric embeddings
- **Patch Extractor** (`ml/retrieval/encoders/patch_extractor.py`)
  - Fine-grained local embeddings
- **Depth Extractor** (`ml/retrieval/encoders/depth_extractor.py`)
  - Geometry-aware embeddings
- **OCR Encoder** (`ml/retrieval/encoders/ocr_encoder.py`)
  - Text-based embeddings
- **Audio Encoder** (`ml/retrieval/encoders/audio_encoder.py`)
  - Audio embeddings
- **Scene Graph Encoder** (`ml/retrieval/encoders/scene_graph_encoder.py`)
  - Graph-based embeddings

### Indexing
- **Neural Index Builder** (`ml/retrieval/indexing/neural_index_builder.py`)
  - FAISS index building (HNSW, IVF-PQ)
  - GPU/CPU support
- **Index Manager** (`ml/retrieval/indexing/index_manager.py`)
  - Index lifecycle management

### Retrieval
- **Stage 1 ANN** (`ml/retrieval/retrieval/stage1_ann.py`)
  - Fast approximate nearest neighbor search
- **Stage 2 Reranker** (`ml/retrieval/retrieval/stage2_rerank.py`)
  - Multi-vector reranking
- **Knowledge Augment** (`ml/retrieval/retrieval/knowledge_augment.py`)
  - GNN-based knowledge graph integration
- **Concept Retrieval** (`ml/retrieval/retrieval/concept_retrieval.py`)
  - Concept-based retrieval

### Fusion
- **Attention Fusion** (`ml/retrieval/fusion/attention_fusion.py`)
  - Attention-based multi-vector fusion
- **Meta Fusion** (`ml/retrieval/fusion/meta_fusion.py`)
  - Meta-learning fusion weights
- **Fusion Train** (`ml/retrieval/fusion/fusion_train.py`)
  - Fusion training utilities

### Cross-View Training
- **Cross-View Trainer** (`ml/retrieval/cross_view/cv_training.py`)
  - Multi-view contrastive learning

---

## Optimization & Mobile

### Mobile Optimizations
- **Mobile Optimizer** (`ml/optimization/mobile_optimizations.py`)
  - Model pruning
  - Head disabling
  - Memory estimation
- **Edge-Cloud Hybrid** (`ml/optimization/mobile_optimizations.py`)
  - Smart routing between edge and cloud

---

## Utilities

### Preprocessing
- **Image Preprocessor** (`ml/utils/preprocessing.py`)
  - Condition-specific adaptations
  - Learnable FiLM adapters

### Output Management
- **Output Scheduler** (`ml/utils/output_scheduler.py`)
  - Cross-modal output management
  - Cognitive load budgeting
- **Description Generator** (`ml/utils/description_generator.py`)
  - Natural language descriptions
- **OCR Integration** (`ml/utils/ocr_integration.py`)
  - Text detection and reading

### Memory & Planning
- **Spatial Memory** (`ml/utils/spatial_memory.py`)
  - Object tracking over time
- **Path Planner** (`ml/utils/path_planner.py`)
  - Navigation path planning

### Monitoring & Logging
- **Logging Config** (`ml/utils/logging_config.py`)
  - Thread-safe logging
  - Patient/clinician/dev modes
- **Monitoring** (`ml/utils/monitoring.py`)
  - Performance monitoring
  - Health checks

### Error Handling
- **Error Handling** (`ml/utils/error_handling.py`)
  - Head kill switches
  - Ethical safeguards
- **Schema Validator** (`ml/utils/schema_validator.py`)
  - Output validation

### Other Utilities
- **User Preferences** (`ml/utils/user_preferences.py`)
- **Semantic Grouping** (`ml/utils/semantic_grouping.py`)
- **Sound Processing** (`ml/utils/sound_processing.py`)
- **Adaptive Assistance** (`ml/utils/adaptive_assistance.py`)
- **Stress Testing** (`ml/utils/stress_testing.py`)

---

## Therapy System

### Components (Stub/Placeholder)
- **Task Generator** (`ml/therapy/task_generator.py`)
  - Adaptive therapy task generation
  - **Status**: Placeholder, not integrated
- **Session Manager** (`ml/therapy/session_manager.py`)
  - Therapy session tracking
  - **Status**: Placeholder, not integrated
- **Therapy Integration** (`ml/therapy/therapy_integration.py`)
  - Therapy feedback integration
  - **Status**: Placeholder, not integrated

---

## Application Components

### UI Components
- **Voice Feedback** (`app/ui/voice_feedback.py`)
  - Text-to-speech integration
- **Haptic Feedback** (`app/ui/haptic_feedback.py`)
  - Haptic pattern generation
- **Overlay Engine** (`app/overlays/overlay_engine.py`)
  - Visual overlay rendering

### Personalization
- **Personal Mode Manager** (`app/personal_mode.py`)
  - Active scene exploration
  - Predictive navigation guidance

---

## Scripts

### Training Scripts
- **Train MaxSight** (`scripts/train_maxsight.py`)
  - Main training script
- **Smoke Train** (`scripts/smoke_train.py`)
  - Proof of life training
  - Device auto-selection

### Validation Scripts
- **Validate Forward Passes** (`scripts/validate_forward_passes.py`)
  - Forward pass validation across tiers
- **Analyze Function Flow** (`scripts/analyze_function_flow.py`)
  - Function flow analysis
- **Benchmark Tiers** (`scripts/benchmark_tiers.py`)
  - Performance benchmarking

### Analysis Scripts
- **Analyze Forward Passes** (`scripts/analyze_forward_passes.py`)
- **Analyze Forward Paths** (`scripts/analyze_forward_paths.py`)
- **Throughput Benchmark** (`scripts/throughput_benchmark.py`)

### Data Scripts
- **Setup COCO Splits** (`scripts/setup_coco_splits.py`)
- **Generate MaxSight Dataset** (`scripts/generate_maxsight_dataset.py`)
- **Fix Dataset Splits** (`scripts/fix_dataset_splits.py`)
- **Generate Class Weights** (`scripts/generate_class_weights.py`)

### Testing Scripts
- **Run Stress Tests** (`scripts/run_stress_tests.py`)
- **Profile Integration** (`scripts/profile_integration.py`)

---

## Components Not Currently Used

### Stub Implementations
- **Eye Model** (`ml/models/eye_model/eye_model.py`)
  - Eye tracking and fatigue detection
  - **Status**: Stub implementation, not integrated into forward pass

### Defined But Not Called
- **Predictive Alert Head** (`ml/models/heads/predictive_alert_head.py`)
  - **Status**: Defined in `__init__` but not called in forward pass
  - **Reason**: Future integration planned

---

## Component Statistics

- **Total Model Components**: 50+
- **Head Components**: 20+
- **Backbone Components**: 4
- **Temporal Components**: 3
- **Retrieval Components**: 15+
- **Training Components**: 10+
- **Utility Components**: 15+
- **Application Components**: 5+

---

## Integration Status

### ✅ Fully Integrated
- MaxSightCNN (main model)
- All Tier 1 heads (safety-critical)
- All Tier 2 heads (navigation & context)
- Most Tier 3 heads (enhancement & therapy)
- Temporal processing (T5+)
- Scene graph encoding
- Retrieval system (async, advisory)
- Training infrastructure
- Export functionality

### ⚠️ Partially Integrated
- Predictive Alert Head (defined but not called)
- Therapy components (placeholder implementations)

### ❌ Not Integrated
- Eye Model (stub implementation)
- Some therapy utilities (not in main pipeline)

---

**Last Updated**: $(date)  
**Total Components**: 100+  
**Active Components**: 95+  
**Stub/Placeholder Components**: 5+

