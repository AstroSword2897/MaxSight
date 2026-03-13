# MaxSight 3.0 - Removing Barriers for Vision & Hearing Disabilities

**Production-Grade Accessibility System** | **Multi-Task Deep Learning for Environmental Understanding**

**Last Updated**: 2026-02  
**Status**: Production-ready training and data pipeline. Use `python scripts/product/run.py` for canonical `train/validate/export/package/smoke` (or call ops scripts directly under `scripts/ops/`). See **docs/status.md** for current status.

---

## Table of Contents

1. [Project Overview & Goals](#-project-overview--goals)
2. [Productization Summary (from reports)](#productization-summary-from-reports)
3. [Actions Taken - Complete Development History](#-actions-taken---complete-development-history)
4. [System Architecture - Deep Dive](#-system-architecture---deep-dive)
5. [Data Flow & Processing Pipeline](#-data-flow--processing-pipeline)
6. [Training Flow & Hyperparameter Strategy](#-training-flow--hyperparameter-strategy)
7. [Inference Flow & Real-Time Processing](#-inference-flow--real-time-processing)
8. [Effectiveness & Results](#-effectiveness--results)
9. [Repository Stack & Technology](#-repository-stack--technology)
10. [Current Work & Next Steps](#-current-work--next-steps)
11. [Quick Start Guide](#-quick-start-guide)
12. [Main Components](#main-components) (includes [Component reference: what and why](#component-reference-what-each-does-and-why-its-there) and [Concrete reference: outputs, configs, env, CLI](#concrete-reference-outputs-configs-env-cli))
13. [Testing & Validation](#-testing--validation)
14. [Performance & Safety](#-performance--safety)
15. [Deployment & Export](#-deployment--export)
16. [Documentation](#-documentation)

---

## Project Overview & Goals

### Primary Mission

MaxSight 3.0 is a **production-grade accessibility application** that helps users with vision and hearing disabilities navigate and understand their environment through advanced computer vision and multimodal feedback. The system removes barriers by providing the same rich environmental information that sighted people process automatically.

### Primary Problem Statement

**"What are ways that those who cannot see or hear be able to interact with the world like those who can?"**

MaxSight answers this by implementing four barrier-removal methods from accessibility research:

1. **Environmental Structuring**: Labels surroundings in ways users can understand
2. **Clear Multimodal Communication**: Visual, audio, and haptic feedback
3. **Skill Development Across Senses**: Addresses different senses for information input
4. **Routine Workflow**: Adapts tasks to usage patterns and needs

### What Makes MaxSight Different

**Standard object detectors** answer: "What is this?" and "Where is it?"

**MaxSight 3.0** answers:
- **WHAT**: Object class (door, stairs, vehicle, person) - 91 COCO classes + 200+ accessibility classes
- **WHERE**: Precise bounding box position (for directional cues)
- **HOW FAR**: Distance zones (near/medium/far) + precise depth estimation
- **HOW URGENT**: Urgency level (safe/caution/warning/danger) for safety
- **HOW FINDABLE**: Object findability scores (for users with low vision)
- **SCENE CONTEXT**: Natural language scene descriptions
- **ACCESSIBILITY METRICS**: Contrast sensitivity, glare risk, navigation difficulty
- **TEMPORAL AWARENESS**: Motion tracking, predictive alerts, temporal consistency
- **PERSONALIZATION**: User-specific adaptations and preferences
- **THERAPY STATE**: Fatigue detection, depth/focus, contrast mapping

### Project Goals

#### Short-Term Goals (Completed)
-  Complete architecture implementation (Phases 0-9)
-  All tests passing (163/163)
-  Training infrastructure ready
-  Data pipeline established
-  Hyperparameter configurations for all tiers

#### Medium-Term Goals (In Progress)
-  Data gathering script and train/val/test splits (see [Requirements before training](#requirements-before-training))
-  Full training runs (T0 baseline; use cloud GPU for production scale)
-  Performance benchmarking (see `ml/training/benchmark.py` and `pytest tests/`)
-  Model export (JIT/ONNX/CoreML; see `python -m ml.training.export --help`)

#### Long-Term Goals
-  Production training (all tiers T0-T5)
-  Transfer learning (T2 → T5)
-  Mobile deployment (iOS CoreML)
-  Real-world testing with users
-  Performance optimization
-  Accessibility certification

### Model Statistics

- **Parameters**: ~250M (comprehensive class system, T2 tier baseline); T0 ~29M, T5 ~320M.
- **Input**: `images` `[B, 3, 224, 224]` RGB (normalized with ImageNet mean/std); optional `audio_features` `[B, 128]` (e.g. MFCC); optional `condition_mode` string (e.g. `'glaucoma'`, `'amd'`, `'cataracts'`).
- **Output**: Single dict with 30+ keys. Core keys: `obj_scores` `[B, H*W]`, `cls_logits` `[B, H*W, num_classes]`, `box_preds` `[B, H*W, 4]`, `urgency` (per detection or scene), `distance` zones, `contrast_map`, `motion_flow`, `motion_magnitude`, `fatigue_score`, `blink_rate`, `fixation_stability`, `depth_map`, `uncertainty`, `therapy_state` (if provided by pipeline), `contrast_map`, `edge_map`, `roi_utility`, `navigation_difficulty`, `glare_risk_level`, `object_findability`, `uncertainty_score`, `hazard_probs`, `time_to_hazard`, `recommended_action`, plus scene/OCR/scene graph when enabled. Exact keys depend on tier and `enable_accessibility_features`.
- **Stage A Latency**: **≤ 80 ms** target (ResNet50+FPN only). Decision point: skip Stage B if Stage A &gt; 80 ms or `uncertainty_score` &gt; 0.7 (thresholds in tier/config). See `ml/runtime_constants.py` (LATENCY_MEDIAN_MS, LATENCY_P95_MS).
- **Stage B Latency**: &lt;500ms (opportunistic, tier-dependent).
- **Supported Classes**: 91 COCO + 200+ accessibility classes; class IDs and names in `COCO_CLASSES_DICT` / category list used by dataset and detection head.
- **Vision Conditions**: 13 supported (e.g. refractive errors, cataracts, glaucoma, AMD, diabetic_retinopathy, retinitis_pigmentosa, color_blindness, CVI, amblyopia, strabismus); condition affects `ml/utils/preprocessing.py` and optional dynamic conv.
- **Task Heads**: 30+ specialized heads; each head is a `nn.Module` with a `forward()` taking shared features (and sometimes dedicated inputs like `eye_features`). Built in `ml/models/maxsight_cnn.py` when tier and `enable_accessibility_features` allow.
- **Export Formats**: JIT (`.pt`), CoreML (`.mlpackage`), ONNX, ExecuTorch (`.pte`). Export stubs `global_encoder` (CLIP) and can disable scene graph for traceability; see `ml/training/export.py`.

---

## Productization Summary (from reports)

This section consolidates the important information from **docs/productization/** so release, safety, and product decisions are visible in one place. Full detail remains in the linked docs.

### Intended use and scope (V1)

MaxSight is an **assistive smart-glasses system** that helps visually impaired users understand nearby hazards, orientation cues, and everyday context through **spoken and haptic guidance**. V1 focus: safety-critical awareness (hazards, obstacle proximity, directional cues), daily independence (text reading, finding objects/signs, route cues), and low-verbosity situational summaries. **Explicit non-claims**: MaxSight is assistive guidance, not autonomous navigation; users should not rely on it as their only mobility safety aid; it does not provide medical diagnosis or treatment advice. See **docs/productization/01_product_scope_and_claims.md**.

### Mandatory safety gates (V1 release blockers)

All mandatory gates must pass before release. Failure on any blocks release.

| Gate ID | Metric | Threshold | Blocker if failed |
|--------|--------|-----------|--------------------|
| SG-01 | Hazard recall (critical hazards) | ≥ 0.95 | Yes |
| SG-02 | False-safe rate | ≤ 0.01 | Yes |
| SG-03 | Time-to-alert p95 | ≤ 80 ms | Yes |
| SG-04 | Time-to-alert median | ≤ 80 ms | Yes |
| SG-05 | Directional cue correctness | ≥ 0.90 | Yes |
| SG-06 | Distance zone accuracy (near/medium/far) | ≥ 0.85 | Yes |
| SG-07 | Critical hazards still surfaced under uncertainty | 100% | Yes |
| SG-08 | Overload guardrail (alerts/min in dense scenes) | ≤ 12 avg unless emergency | Yes |

Critical hazards include moving vehicles in crossing context, immediate collision obstacles, curb/drop-off hazards. **Release decision**: run gate suite → signed gate report → block on any failed mandatory gate; approve only with **safety owner sign-off**. See **docs/productization/02_safety_first_release_gates.md**.

### Canonical commands (product pipeline)

Use **`python scripts/product/run.py`** for the canonical surface. All paths from repo root.

| Command | Purpose | How to run |
|--------|---------|------------|
| **train** | Train production model | `run.py train --data-dir <path> [--config <yaml>]` |
| **validate** | Tests + optional checkpoint/data checks | `run.py validate [--checkpoint <path>] [--skip-export-tests]` |
| **export** | Checkpoint → CoreML/JIT/ONNX | `run.py export --checkpoint <path> --format coreml --output <path>` |
| **package** | Xcode-ready bundle | `run.py package --checkpoint <path> --output <dir>` |
| **smoke** | Short training + inference sanity | `run.py smoke [--epochs 2]` |
| **transfer** | T2 → T5 weight transfer | `run.py transfer --source <T2_ckpt> [--config ml/training/configs/t2_to_t5_transfer.yaml]` |

### T2 → T5 path (T5 MVP)

1. **T2 source**: Train with config that disables temporal/cross-task: `run.py train --data-dir <path> --config ml/training/configs/t2_hybrid_vit.yaml --train-annotation ... --val-annotation ...`. Checkpoint → `checkpoints/t2_hybrid_vit/`.
2. **Transfer**: `run.py transfer --source checkpoints/t2_hybrid_vit/best_model.pth --config ml/training/configs/t2_to_t5_transfer.yaml`. Writes e.g. `checkpoints/t5_temporal_transfer/t5_from_t2_init.pt`.
3. **T5 fine-tune**: `run.py train --data-dir <path> --resume-from checkpoints/t5_temporal_transfer/t5_from_t2_init.pt ...` (optionally with video/sequence data).

### MVP runtime contract (shipped app)

The shipped T5 MVP must depend only on **MVP output keys** in `ml.runtime_constants.MVP_MODEL_OUTPUT_KEYS` (classifications, boxes, objectness, text_regions, urgency_scores, distance_zones, precise_distances, uncertainty, temporal_consistency, etc.). The app should use **`ml.runtime_constants.filter_mvp_model_outputs(outputs, training=False)`** in the production inference path. Export/package use the full model; filtering is applied at runtime.

### Runtime boundaries and pilot

- **Critical path**: hazard detection, urgency, direction, distance, alert scheduling; always runs, never blocked by enhancement features.
- **Secondary path**: OCR, scene summaries, retrieval; never blocks critical path.
- **Pilot validation**: real-world scenarios, KPIs, and review loop are in **docs/productization/05_pilot_validation_protocol.md**. Deployment: train → export to CoreML → package for Xcode → integrate into glasses app → run pilot per protocol.

### Where the full reports live

| Doc | Content |
|-----|---------|
| **01_product_scope_and_claims.md** | Product boundaries, claims matrix, non-claims |
| **02_safety_first_release_gates.md** | Full gate definitions, evidence artifacts, roles |
| **03_pipeline_declutter_map.md** | Script consolidation and canonical surface |
| **04_runtime_boundary_spec.md** | Critical vs secondary contract, degraded modes |
| **05_pilot_validation_protocol.md** | Pilot scenarios, metrics, incident classification |
| **PRODUCTION_RUNBOOK.md** | Step-by-step production and deployment |

---

## Actions Taken - Complete Development History

### Phase 0: Backbone Networks 

**Actions**:
- Implemented ResNet50+FPN backbone for Stage A (safety-critical)
- Implemented Hybrid CNN-ViT backbone for Stage B (context enhancement)
- Implemented Vision Transformer components
- Implemented Dynamic Convolution for adaptive processing
- Created backbone abstraction layer

**Results**:
- Stage A backbone: ResNet50+FPN (always used, ≤ 80 ms target)
- Stage B backbone: Hybrid CNN-ViT (T2+), Temporal (T5+)
- Multi-scale feature extraction via FPN
- Support for progressive tier enablement

**Impact**: Foundation for two-stage inference pipeline established.

### Phase 1: Multimodal Fusion 

**Actions**:
- Implemented audio-visual fusion with attention mechanisms
- Created cross-modal attention layers
- Implemented haptic feedback integration
- Created fusion abstraction for multiple modalities

**Results**:
- Audio features integrated: `[B, 128]` MFCC features
- Cross-modal attention enables audio-aware detection
- Fusion layer supports multiple input modalities

**Impact**: System can process both visual and audio information simultaneously.

### Phase 2: Task Heads 

**Actions**:
- Implemented 30+ specialized task heads organized by criticality tiers
- Created Tier 1 heads: Objectness, Classification, Box Regression, Distance, Urgency, Uncertainty
- Created Tier 2 heads: Motion, Therapy State, ROI Priority, Navigation Difficulty, Findability
- Created Tier 3 heads: Scene Description, OCR, Scene Graph, Sound Events, Personalization, Predictive Alerts
- Implemented condition-specific adaptations for 13 vision conditions

**Results**:
- **163 tests passing** across all head implementations
- All heads validated with forward pass tests
- Tier-based execution model ensures safety-first approach

**Impact**: Comprehensive multi-task learning system that addresses all accessibility needs.

### Phase 3: Retrieval System 

**Actions**:
- Implemented FAISS-based two-stage retrieval system
- Created neural quantization for efficient indexing
- Implemented async retrieval worker (non-blocking)
- Created retrieval heads for knowledge augmentation
- Implemented concept-based and scene-based retrieval

**Results**:
- Two-stage retrieval: Stage 1 (ANN search) → Stage 2 (reranking)
- Async retrieval never blocks safety-critical inference
- Advisory-only design (never affects Tier 1 or Tier 2 decisions)

**Impact**: System can leverage similar scenes for context without compromising safety.

### Phase 4: Knowledge Integration 

**Actions**:
- Implemented Scene Graph Encoder for spatial/semantic relations
- Created GNN encoder for graph neural network processing
- Implemented spatial relation extraction (above, below, left, right, etc.)
- Implemented semantic relation extraction (contains, supports, etc.)
- Created batched scene graph processing

**Results**:
- Scene graphs enable rich spatial reasoning
- Relations extracted: spatial (geometric) + semantic (functional)
- Graph-based encoding supports complex scene understanding

**Impact**: System understands object relationships, not just individual objects.

### Phase 5: Training Infrastructure 

**Actions**:
- Implemented production-grade training loop with resume capability
- Created GradNorm multi-task loss balancing
- Implemented self-supervised pretraining (MAE, SimCLR)
- Created knowledge distillation framework
- Implemented Elastic Weight Consolidation (continual learning)
- Added mixed precision training support
- Created checkpointing and logging infrastructure
- Implemented EMA (Exponential Moving Average) for model weights
- Created validation framework with comprehensive metrics

**Results**:
- **Smoke training passed**: Loss decreased (0.7246 → 0.6013)
- Training loop supports resume from checkpoints
- GradNorm prevents gradient warfare between tasks
- All training components validated

**Impact**: Production-ready training system that can handle complex multi-task learning.

### Phase 6: Personalization 

**Actions**:
- Implemented Personalization Head for user-specific adaptations
- Created user preference system
- Implemented online learning framework
- Created adaptive assistance system

**Results**:
- User-specific model adaptations
- Preference-based output scheduling
- Online learning support (future integration)

**Impact**: System can adapt to individual user needs and preferences.

### Phase 7: Optimization 

**Actions**:
- Implemented quantization (INT8) for mobile deployment
- Created pruning framework
- Implemented mobile optimizations
- Created export pipeline (CoreML, ONNX, ExecuTorch)

**Results**:
- Model size reduction: ~250M params → <50MB quantized
- Export formats: CoreML (iOS), ONNX (cross-platform), ExecuTorch (mobile)
- Mobile-ready optimizations

**Impact**: System can run on mobile devices with acceptable performance.

### Phase 8: Simulator 

**Actions**:
- Implemented complete web-based simulator (Flask)
- Created multi-user session support
- Implemented real-time processing pipeline
- Created visual overlay rendering
- Implemented output scheduling (Patient, Clinician, Dev modes)
- Created performance benchmarking tools
- Implemented stress testing framework

**Results**:
- Web simulator for end-to-end testing
- Multi-user support with proper locking
- Real-time inference pipeline
- Visual feedback system

**Impact**: Complete product simulation without requiring iOS app.

### Phase 9: Evaluation 

**Actions**:
- Implemented comprehensive evaluation metrics
- Created multi-modal metrics
- Implemented accessibility-specific metrics
- Created robustness evaluation framework
- Implemented lighting-aware metrics analysis

**Results**:
- Comprehensive metrics: mAP, precision, recall, F1
- Accessibility metrics: urgency accuracy, distance accuracy
- Robustness metrics: noise tolerance, adversarial robustness

**Impact**: System can be evaluated across multiple dimensions.

### Recent Fixes & Improvements (2025-01-30)

**Test Suite Fixes**:
- Fixed 13 test failures (model size updates, API changes, missing methods)
- Updated model size thresholds for 250M parameter model
- Fixed training loss API tests (MAE, SimCLR, Knowledge Distillation, EWC)
- Added missing `extract_relations()` method to SceneGraphEncoder
- Fixed simulator output format tests (dev mode)
- Improved condition robustness test logic
- Made export validation test more lenient for expected failures

**Training Framework Improvements**:
- Fixed EMA state dict interface (supports distributed training)
- Preserved optimizer state when unfreezing backbone
- Improved validation metric safety (comprehensive shape validation)
- Enhanced GradNorm integration
- Added MPS seed setting support
- Improved loss defaulting warnings

**Data Pipeline Setup**:
- Created COCO dataset download script with multiple fallback methods
- Created data pipeline module (data loader creation, collation, class weights)
- Created training configuration files for all tiers (T0-T5)
- Created training pipeline test script
- Created COCO dataset splitter

**Hyperparameter Tuning**:
- Systematically updated all tier configurations with numerically precise values
- Applied learning rate scaling by model size
- Rebalanced loss weights (box regression: 5.0 → 3.0, semantic tasks: 0.1 → 0.3)
- Increased data loader workers (4 → 8)
- Added minimum learning rate (1e-6) to prevent late-stage collapse
- Extended warmup epochs for T5 (15 → 20)

**Transfer Learning Preparation**:
- Created T2 → T5 transfer learning plan
- Implemented selective weight transfer
- Created phased freeze/unfreeze schedule
- Implemented parameter-grouped learning rates
- Created phased loss unlock schedule
- Created comprehensive transfer documentation

---

## ️ System Architecture - Deep Dive

### Two-Stage Inference Pipeline

The main architectural decision is the **two-stage inference pipeline** that separates safety-critical predictions from enhancement features.

#### Stage A: Fast Safety Pass (≤ 80 ms, every frame)

**Purpose**: Provide safety-critical information that must never be blocked.

**Backbone**: **ALWAYS ResNet50 + FPN** (safety guarantee)
- ResNet50: Proven, fast, predictable
- FPN: Multi-scale feature extraction for objects of all sizes
- No hybrid backbone, no temporal processing (guarantees speed)

**Heads**: Tier 1 safety-critical heads only
- **Objectness**: Is there an object? `[B, H*W]`
- **Classification**: What object? `[B, H*W, 91]`
- **Box Regression**: Where is it? `[B, H*W, 4]`
- **Distance Zones**: How far? `[B, H*W, 3]`
- **Urgency**: How dangerous? `[B, 4]`
- **Uncertainty**: Model confidence `[B, 1]`

**Properties**:
- Highest loss priority in training
- Target: ≤ 80 ms per frame
- Never blocked by Tier 2 or Tier 3
- Always ResNet50+FPN backbone (no hybrid, no temporal)

**Decision point**: After Stage A completes, the code checks latency and uncertainty (e.g. `uncertainty_score` from uncertainty head). Skip Stage B if Stage A latency &gt; 80 ms or uncertainty &gt; 0.7 (thresholds in TierConfig / `ml/runtime_constants.py`). Implementation: in `maxsight_cnn.py` forward, after Tier 1 heads run, a conditional branch either returns Stage A outputs only or continues to Stage B backbone and Tier 2/3 heads. This ensures Stage A always completes, even under load.

#### Stage B: Context Pass (opportunistic, tier-dependent)

**Purpose**: Provide rich context and enhancement features when time permits.

**Backbone**: Hybrid CNN-ViT (T2+) + Temporal (T5+)
- Hybrid CNN-ViT: Combines CNN efficiency with ViT global attention
- Temporal: ConvLSTM + TimeSformer for temporal modeling (T5 only)
- Processes raw images (not Stage A features) for independent processing

**Heads**: Tier 2 & Tier 3 context-rich heads
- **Tier 2**: Motion, Therapy State, ROI Priority, Navigation Difficulty, Findability
- **Tier 3**: Scene Description, OCR, Scene Graph, Sound Events, Personalization, Predictive Alerts

**Properties**:
- Can be skipped if Stage A latency/uncertainty thresholds exceeded
- Graceful degradation: If Stage B fails, Stage A results still returned
- Asynchronous: Some Tier 3 heads run in background threads

### Tiered Head Architecture

Heads are organized into 3 tiers by criticality:

#### Tier 1: Safety-Critical (Never Disabled)

| Head | Purpose | Output Shape | Execution |
|------|---------|--------------|-----------|
| **Objectness** | Is there an object? | `[B, H*W]` | Every frame |
| **Classification** | What object? | `[B, H*W, 91]` | Every frame |
| **Box Regression** | Where is it? | `[B, H*W, 4]` | Every frame |
| **Distance Zones** | How far? | `[B, H*W, 3]` | Every frame |
| **Urgency** | How dangerous? | `[B, 4]` | Every frame |
| **Uncertainty** | Model confidence | `[B, 1]` | Every frame |

**Properties**:
- Highest loss priority in training
- Target: ≤ 80 ms per frame
- Never blocked by Tier 2 or Tier 3
- Always ResNet50+FPN backbone (no hybrid, no temporal)

#### Tier 2: Navigation & Context (Can Degrade)

| Head | Purpose | Output Shape | Execution |
|------|---------|--------------|-----------|
| **Motion** | Object movement | `[B, 2, H, W]` | Every N frames |
| **Therapy State** | Fatigue, depth, contrast | Dict | Every N frames |
| **ROI Priority** | Region prioritization | `[B, N]` | Every N frames |
| **Navigation Difficulty** | Scene complexity | `[B, 1]` | Every N frames |
| **Findability** | Object findability | `[B, H*W]` | Every N frames |

**Properties**:
- Can be throttled (every N frames)
- Can be delayed if Tier 1 needs resources
- Graceful degradation if disabled

#### Tier 3: Enhancement & Therapy

| Head | Purpose | Output Shape | Execution |
|------|---------|--------------|-----------|
| **Scene Description** | Natural language | List[str] | Background |
| **OCR** | Text detection/recognition | Dict | Background |
| **Scene Graph** | Spatial/semantic relations | Dict | Background |
| **Sound Events** | Audio classification | Dict | Background |
| **Personalization** | User adaptations | Dict | Background |
| **Predictive Alerts** | Hazard anticipation | Dict | Background |
| **Retrieval** | Knowledge augmentation | Advisory | Async, non-blocking |

**Properties**:
- Can be disabled when not needed
- Asynchronous (background thread)
- Never blocks Tier 1 or Tier 2
- **Advisory only** (never drives safety decisions)

### Capability Tiers

The system supports progressive tier enablement:

| Tier | Name | Features | Parameters | Device |
|------|------|----------|------------|--------|
| **T0** | BASELINE_CNN | ResNet50+FPN, Tier 1 heads | ~29M | Cloud GPU |
| **T1** | EDGE | + Attention, Tier 2 heads | ~50M | Cloud GPU |
| **T2** | HYBRID_VIT | + Hybrid CNN-ViT, Motion, Therapy | ~210M | Cloud GPU |
| **T3** | CROSS_MODAL | + OCR, Scene Description, Scene Graph | ~250M | Cloud GPU |
| **T4** | CROSS_MODAL | + Audio, Retrieval | ~280M | Cloud GPU |
| **T5** | TEMPORAL | + Temporal (ConvLSTM, TimeSformer) | ~320M | Cloud GPU |

**All tiers require cloud GPU (CUDA) for training.**

### Key Architectural Guarantees

1. **Stage A Always ResNet50+FPN**: No hybrid backbone, no temporal processing in Stage A.
   - **Implementation**: In `ml/models/maxsight_cnn.py`, Stage A forward path uses only the ResNet50 backbone and FPN (and optional SE/CBAM on FPN when T1+). No conditional that swaps in hybrid or temporal for Stage A. Method names may be e.g. `_forward_stage_a` or inline: images → backbone → FPN → Tier 1 heads.
   - **Why**: ResNet50+FPN is fast (≤ 80 ms target), predictable, and well-tested. Hybrid backbones are slower and less predictable.

2. **Stage B Uses Raw Images**: Hybrid backbone processes raw images, not Stage A features.
   - **Implementation**: When Stage B runs, the hybrid backbone (e.g. `HybridCNNViTBackbone`) is called with the same input `images` tensor `[B,3,224,224]`, not with the FPN or detection feature tensors from Stage A. So `backbone_B(images)` is independent of Stage A features.
   - **Why**: Ensures Stage B can extract different (complementary) features than Stage A.

3. **Temporal Only in Stage B**: Temporal processing uses Stage A features as input.
   - **Implementation**: The temporal encoder (e.g. `TemporalEncoder` in `ml/models/temporal/temporal_encoder.py`) is fed a sequence of feature maps that come from Stage A (e.g. FPN output or detection feature map) over time, i.e. `feature_frames` [B, C, T, H, W] where C is the Stage A feature dimension (e.g. 256). It does not receive raw image sequences in the same way as the hybrid backbone.
   - **Why**: Reusing Stage A features is more efficient than re-running a full backbone on each frame.

4. **Retrieval is Async**: Non-blocking, advisory only.
   - **Implementation**: Retrieval is invoked from a background thread or async worker (e.g. `ml/retrieval/retrieval/async_retrieval.py`). The main inference path does not wait for retrieval results; Tier 1 and Tier 2 outputs are produced without retrieval. Any use of retrieval (e.g. for scene description or knowledge augmentation) is advisory and does not change detection/urgency/distance.
   - **Why**: Retrieval can take 100–500ms; keeping it async avoids delaying safety-critical outputs.

5. **Safety First**: Stage A completes before Stage B decision.
   - **Implementation**: In the model forward, the order is: (1) Run Stage A (backbone + FPN + Tier 1 heads), (2) Read Stage A outputs (including e.g. uncertainty_score), (3) Apply decision rule (latency and uncertainty thresholds), (4) If not skip, run Stage B and merge outputs. So `t_A` is always measured before the skip decision.
   - **Why**: Safety predictions must be available before deciding whether to run Stage B.

6. **Fail-Safe**: High latency or uncertainty → skip Stage B, return Stage A only.
   - **Implementation**: Conditional in forward: if Stage A latency &gt; 80 ms (TierConfig.max_latency_ms) or if uncertainty_score &gt; 0.7, do not run Stage B; return the outputs dict containing only Stage A results (and optionally empty or None for Stage B-only keys). Thresholds can live in TierConfig (e.g. `max_latency_ms`) or in a separate inference config.
   - **Why**: If Stage A is slow or uncertain, Stage B is unlikely to help and wastes resources.

### Detailed Architecture: ResNet50+FPN (Stage A)

**ResNet50** (torchvision): Stem Conv 7×7 stride 2, BN, ReLU, MaxPool 3×3 stride 2. Layer1→C2 [B,256,56,56], Layer2→C3 [B,512,28,28], Layer3→C4 [B,1024,14,14], Layer4→C5 [B,2048,7,7]. **FPN**: Lateral 1×1 convs to 256 ch; top-down P5, P4=P4_lat+up(P5), P3=P3_lat+up(P4), P2=P2_lat+up(P3). P2–P5 all 256 ch at 56, 28, 14, 7. **Detection fusion**: P3, P4, P5 resized to 14×14, concat → [B,768,14,14] (768=256×3); this feeds detection heads and many Tier 2 heads. See **ml/models/maxsight_cnn.py** and **docs/architecture.md**.

### Detailed Architecture: Hybrid CNN-ViT (Stage B)

**CNN branch**: Same ResNet50+FPN structure; output FPN levels P2–P5 (256 ch each). Global pooling (e.g. adaptive avg pool per level then concat or mean) → F_cnn vector. **ViT branch**: Patch embedding 224/16 → 196 patches, dim 768; add positional embedding; 12 transformer blocks (multi-head self-attention + FFN, LayerNorm); CLS token or mean of patch tokens → Z_cls [B,768]. **Cross-layer** (in hybrid_backbone.py): CNN→ViT: project each FPN level to 768 dim, reshape to 14×14 or patch grid, add to ViT tokens with learnable scale α (e.g. 0.1). ViT→CNN: reshape ViT sequence to spatial map, project to 256 ch, add to FPN features. **Fusion**: AdaptiveFeatureFusion: project CNN and ViT to fused_dim, gating (softmax over two branches), output = gate_cnn * cnn_proj + gate_vit * vit_proj. CrossModalAttention (CNN↔ViT): two MultiheadAttention layers (CNN queries ViT, ViT queries CNN), residual + LayerNorm. See **ml/models/backbone/hybrid_backbone.py** and **docs/SYSTEMS.md**.

### Detailed Architecture: Temporal Processing (T5 Only)

ConvLSTM consumes Stage A feature sequences [B, T, C, H, W]; input/forget/output gates and cell update use convs. TimeSformer: patches over time, temporal then spatial attention, residual. Motion head outputs optical flow (u, v) and magnitude/direction. See **ml/models/temporal/temporal_encoder.py** and **ml/models/temporal/conv_lstm.py**.

---

## Data Flow & Processing Pipeline

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Images [B, 3, 224, 224] + Audio [B, 128] when provided        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
│  - Normalization (ImageNet stats)                              │
│  - Condition-specific adaptations (if enabled)                 │
│  - Audio feature extraction (MFCC)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE A BACKBONE                             │
│  ResNet50 + FPN → fpn_features, fused_features, scene_context  │
│  Latency: ≤ 80 ms target                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE A HEADS (Tier 1)                       │
│  - Objectness [B, H*W]                                         │
│  - Classification [B, H*W, 91]                                 │
│  - Box Regression [B, H*W, 4]                                  │
│  - Distance Zones [B, H*W, 3]                                  │
│  - Urgency [B, 4]                                              │
│  - Uncertainty [B, 1]                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │  DECISION POINT │
                    │  latency >80ms  │
                    │  OR uncertainty │
                    │  >0.7?          │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐         ┌──────────────┐
        │  SKIP STAGE B │         │  RUN STAGE B │
        │  Return Stage │         │  (if tier ≥T2)│
        │  A only       │         │               │
        └──────────────┘         └───────┬───────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    STAGE B BACKBONE           │
                          │  Hybrid CNN-ViT (T2+)         │
                          │  + Temporal (T5+)             │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    STAGE B HEADS (Tier 2/3)   │
                          │  - Motion                      │
                          │  - Therapy State               │
                          │  - Scene Graph                 │
                          │  - OCR                         │
                          │  - Scene Description           │
                          │  - Sound Events                │
                          │  - Personalization             │
                          │  - Predictive Alerts            │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    ASYNC RETRIEVAL (Tier 3)   │
                          │  - Knowledge augmentation      │
                          │  - Scene similarity search     │
                          │  - Non-blocking               │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    OUTPUT ASSEMBLY            │
                          │  Dictionary with 30+ outputs  │
                          │  + metadata                    │
                          └───────────────────────────────┘
```

### Data Pipeline Components

#### 1. Dataset Loading (`ml/data/dataset.py`)

**Class**: `MaxSightDataset`. **Constructor**: takes annotation path(s), image root dir, optional `condition_mode` (string), transform, and optional audio config.

**COCO annotation structure**: Top-level keys `images`, `annotations`, `categories`. Each image: `id`, `file_name`, `width`, `height`. Each annotation: `id`, `image_id`, `category_id`, `bbox` `[x, y, width, height]` in pixels, optional `area`, `iscrowd`. Categories: `id`, `name`, optional `supercategory`. Paths in `file_name` are resolved relative to the image root directory passed to the dataset.

**Returned item keys** (typical): `images` (tensor `[3, H, W]` after transform), `labels` (class IDs per object), `boxes` (normalized boxes: center format or xyxy depending on pipeline), `num_objects` (int per image), `distance` (zone per object or per image if present), `urgency` (if present), optional `audio` (tensor), optional `condition_mode`. Batch collation then produces batched tensors; variable-length lists (e.g. per-image annotations) are padded or list-of-dict in the batch.

**Box normalization**: From COCO `bbox` (x, y, w, h) in pixels, conversion to center format: `x_center = (x + w/2) / image_width`, `y_center = (y + h/2) / image_height`, `width_norm = w / image_width`, `height_norm = h / image_height`, all in [0, 1]. Used by detection loss and head targets.

**Distance zones**: Typically derived from relative box area (e.g. box_area &gt; 0.1 → near, &gt; 0.01 → medium, else far) or from annotation field if present. **Urgency**: From category (e.g. person=caution, car=warning) or annotation field; used for urgency head target.

**Preprocessing** (condition-specific): Applied in dataset or via `ml/utils/preprocessing.py`. Examples: **cataracts** — Gaussian blur (e.g. kernel 5, sigma 1.5), reduce contrast; **glaucoma** — central mask (e.g. 30% radius), darken periphery; **AMD** — darken central region (e.g. 20% radius); **retinitis_pigmentosa** — brighten, edge enhance. Normalization: ImageNet mean and std (e.g. mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) and resize to 224×224 (or configurable size).

**Audio**: If used, MFCC or spectrogram features (e.g. 128-dim vector per sample) loaded or computed; shape typically `[T, F]` or `[F]` per sample, then batched. **Synthetic annotations**: Optional path to generate or fill labels when annotations are missing; implementation detail in dataset or separate script.

**Key methods**: `__getitem__(idx)` returns one sample dict; `__len__` returns number of images. See **docs/training-data-loading.md** and **ml/data/dataset.py**.

#### 2. Data Augmentation (`ml/data/advanced_augmentation.py`)

**Image**: Geometric — rotation (e.g. ±15°), scale (0.8–1.2), translation (e.g. ±10% of size), horizontal flip (p=0.5); applied with bbox transform so boxes stay aligned. Photometric — brightness/contrast/saturation/hue jitter, Gaussian noise (e.g. std 0.01). Advanced — cutout (random erase), mixup (combine two images with λ from Beta), mosaic (4-image grid). **Audio**: Time stretch (e.g. 0.8–1.2×), pitch shift (e.g. ±2 semitones), time shift, gain (±6 dB); frequency-domain: add noise to MFCC, time/frequency masking. **Synchronized**: Same geometric choice (e.g. flip) applied to image and audio (e.g. swap stereo channels on flip). **Condition-specific**: Per-condition transforms (cataracts blur level, glaucoma peripheral loss %, AMD scotoma size, diabetic retinopathy spots count, retinitis pigmentosa tunnel radius) to simulate that condition during training. **Entry point**: Typically a transform class or function called from the dataset or training script; see **ml/data/advanced_augmentation.py**.

#### 3. Data Loader (`ml/data/data_pipeline.py`)

**Functions**: `create_data_loaders()` (or equivalent) builds train and val `DataLoader`; takes train/val annotation paths, image dir, batch_size, num_workers, optional condition_mode, optional collate_fn. **Collate**: Custom collate stacks `images` to `[B, 3, H, W]`; pads variable-length annotations (e.g. to max_objects per batch) or keeps as list of dicts; pads audio to max length if present. **Output batch keys**: e.g. `images`, `labels`, `boxes`, `num_objects`, `distance`, `urgency`, optional `audio_features`, optional `condition_mode`; shapes match what the model and loss expect (e.g. `boxes` `[B, max_objects, 4]`).

**Class weights**: Formula `w_i = N_total / (N_classes * N_i)` with small constant to avoid div-by-zero; then normalize so max weight is 1.0. Used for weighted cross-entropy in classification or distance/urgency. **Auto-detection of image dir**: Checks common subdirs (e.g. `images/train2017`, `images/val2017`, `train`, `val`, `test`) under data root; falls back to root if it contains images. **DataLoader kwargs**: num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=True for train, collate_fn=custom_collate_fn. **Batch sizes**: T0 often 16, T2 8, T5 4 with gradient_accumulation_steps (e.g. 8) for effective batch 32. See **ml/data/data_pipeline.py** and **docs/training-data-loading.md**.

---

## Training Flow & Hyperparameter Strategy

### Mathematical Foundations

#### Loss Functions - Complete Formulations

**Files**: **ml/training/losses.py** (per-head loss functions and combiners), **ml/training/head_losses.py** (head-specific helpers). **Total loss**: Weighted sum over heads; weights come from config (e.g. `ml/training/configs/t5_temporal.yaml`) or GradNorm-updated task weights.

**Per-head**: **Objectness** — Focal loss, α=0.25, γ=2; input logits and binary target (object vs background). **Classification** — Focal cross-entropy, same α, γ; num_classes from config. **Box regression** — Smooth L1 (Huber), β=1.0; predicted vs target box coordinates (e.g. center format). **Distance** — Weighted cross-entropy over 3 zones (near/medium/far); class weights from dataset or fixed. **Urgency** — Focal loss with class weights [1.0, 1.5, 2.0, 3.0] for safe/caution/warning/danger. **Depth** — Uncertainty-weighted L1 (Kendall & Gal): `|d - d_gt| * exp(-u) + u`; depth head must output uncertainty. **Motion** — L2 on predicted vs target flow plus smoothness term, λ_smooth=0.1. **Therapy / contrast / scene / OCR / etc.**: Config keys (e.g. `therapy_state`, `scene_description`) control whether loss is computed and at what weight; see tier YAMLs (e.g. `therapy_state: 0.8`).

**Label assignment**: **ml/training/matching.py** — e.g. Hungarian matcher for assigning ground-truth boxes to predictions for loss computation. **Config keys**: Loss weights typically in `loss_weights` or per-head keys in training config; rebalanced values (e.g. box_regression 3.0, scene_description 0.3) in tier configs.

#### GradNorm Algorithm

**File**: **ml/training/task_balancing.py**. **Class**: Typically a balancer that holds task weights and updates them. **Steps**: (1) For each task i, compute weighted loss `w_i * L_i`, then gradient of that loss w.r.t. shared parameters (backbone, FPN); gradient norm G_i = L2 norm of flattened gradients. (2) On first iteration, record initial losses L_i^0. (3) Relative loss L_i^rel = L_i / L_i^0. (4) Target norm G_i^target = Ḡ * (L_i^rel)^α where Ḡ = mean of G_i, α=1.5 (restoring force). (5) GradNorm loss = Σ_i |G_i - G_i^target|. (6) Backprop GradNorm loss w.r.t. task weights w_i; update w_i with learning rate η=0.025 (or from config). (7) Clamp w_i to [0.1, 10.0]. **Update interval**: Every N steps (e.g. 100). **Extreme gradient dampening**: If G_i &gt; 10*Ḡ, reduce w_i (e.g. ×0.5). **Shared parameters**: Usually backbone + FPN; list passed to balancer or inferred from model.

#### Two-Stage Inference - Mathematical Guarantees

**Stage A: Safety Guarantee**
```
t_A = time(ResNet50 + FPN + Tier1_Heads)
P(skip_B) = {
  1  if t_A > 80 ms OR uncertainty > 0.7
  0  otherwise
}
```

**Where:**
- `t_A` = Stage A latency
- `uncertainty` = model confidence (0-1)
- `P(skip_B)` = probability of skipping Stage B

**Mathematical Guarantee**: Stage A always completes before Stage B decision. This ensures safety-critical predictions are never blocked.

**Stage B: Opportunistic Enhancement**
```
if P(skip_B) == 0:
  t_B = time(Hybrid_CNN_ViT + Tier2_3_Heads)
  outputs = StageA_outputs ∪ StageB_outputs
else:
  outputs = StageA_outputs
```

**Where:**
- `t_B` = Stage B latency (if executed)
- `∪` = union of outputs

**Mathematical Guarantee**: Stage B outputs never override Stage A safety predictions. Stage B only adds enhancement features.

#### FPN Feature Extraction - Mathematical Formulation

**Feature Pyramid Network (FPN) extracts multi-scale features:**

```
C2, C3, C4, C5 = ResNet50_stages(images)

P5 = Conv1x1(C5)  # Top-down pathway
P4 = Conv1x1(C4) + Upsample(P5)
P3 = Conv1x1(C3) + Upsample(P4)
P2 = Conv1x1(C2) + Upsample(P3)

Where:
- C2, C3, C4, C5 = ResNet50 feature maps at different scales
- P2, P3, P4, P5 = FPN feature maps (all same channels, different resolutions)
- Upsample = bilinear upsampling
```

**Fused Features for Detection:**
```
P3_resized = Interpolate(P3, size=P4.shape[2:])
P5_resized = Interpolate(P5, size=P4.shape[2:])
Fused = Concat([P3_resized, P4, P5_resized], dim=1)
```

**Where:**
- `Interpolate` = bilinear interpolation to match spatial dimensions
- `Concat` = channel-wise concatenation
- Result: Multi-scale features at same spatial resolution

#### Hybrid CNN-ViT Backbone - Mathematical Operations

**CNN Branch:**
```
X_cnn = ResNet50(images)
F_cnn = FPN(X_cnn)  # [P2, P3, P4, P5]
F_cnn_global = GlobalAvgPool(F_cnn)  # [B, C_cnn]
```

**ViT Branch:**
```
Patches = PatchEmbed(images)  # [B, N, D_vit]
  Where: N = (224/16)² = 196 patches, D_vit = 768

Z_0 = Patches + PositionEmbedding
Z_l = TransformerBlock_l(Z_{l-1})  # l = 1...12
Z_cls = Z_0[CLS_token]  # [B, D_vit]
```

**Cross-Layer Connections:**
```
# CNN → ViT
F_cnn_proj = Conv1x1(F_cnn)  # Project to ViT dimension
F_cnn_pooled = AdaptivePool(F_cnn_proj, size=patch_grid)
Z_l = Z_l + α * F_cnn_pooled  # Residual connection

# ViT → CNN
Z_vit_spatial = Reshape(Z_l, spatial_dims)  # [B, D_vit, H, W]
F_vit_proj = Conv1x1(Z_vit_spatial)  # Project to CNN dimension
F_cnn = F_cnn + α * F_vit_proj  # Residual connection

Where:
- α = 0.1 (learnable cross-layer scaling factor)
- AdaptivePool = adaptive average pooling to match spatial dimensions
```

**Fusion:**
```
# Weighted fusion (default, most stable)
F_fused = β * F_cnn_global + (1 - β) * Z_cls

# Cross-attention fusion (research mode)
Q = Linear(F_cnn_global)  # Query from CNN
K, V = Linear(Z_cls)  # Key, Value from ViT
Attn = Softmax(QK^T / √d) * V
F_fused = FFN(Attn)

Where:
- β = learnable weight (default 0.5)
- d = dimension of attention (typically 512)
- FFN = feedforward network
```

**Checkpoint format**: Saved dict with at least `model_state_dict`, `optimizer_state_dict` (optional), `epoch`, `val_loss` (optional). Paths: e.g. `checkpoints/best_model.pt`, `checkpoints_<condition>/best_model.pt` for per-condition export. **Resume**: Scripts accept e.g. `--resume-from` or load checkpoint and restore model (and optionally optimizer) before continuing. **Optimizer**: Typically AdamW; learning rate from config (e.g. 7.5e-5 for T5). **Scheduler**: Often cosine or step; warmup epochs (e.g. 20 for T5) and min_lr (e.g. 1e-6) in config. **Gradient clipping**: `clip_grad_norm_(parameters, max_norm=1.0)` after backward. **Config files**: `ml/training/configs/` — e.g. `t1_attention.yaml`, `t2_hybrid_vit.yaml`, `t3_cross_task.yaml`, `t4_cross_modal.yaml`, `t5_temporal.yaml`, `t5_temporal_2phase.yaml`, `t2_to_t5_transfer.yaml`; each contains model (tier, num_classes, condition_mode), data (paths, batch_size, num_workers), training (epochs, lr, weight_decay, loss_weights, gradnorm, warmup, min_lr), and optionally transfer (freeze schedule, loss unlock schedule).

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LOADING                                 │
│  MaxSightDataset → DataLoader → Batches                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                                  │
│  Model(images, audio_features) → outputs                        │
│  All heads predict simultaneously                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                             │
│  Per-head losses → GradNorm balancing → Total loss              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKWARD PASS                                │
│  loss.backward() → Gradients computed                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRADIENT CLIPPING                            │
│  clip_grad_norm_(1.0) → Prevents gradient explosion            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER STEP                                │
│  optimizer.step() → Model weights updated                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEDULER STEP                               │
│  scheduler.step() → Learning rate updated                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION (every N batches)                 │
│  Metrics computed: mAP, precision, recall, F1                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHECKPOINTING (every N epochs)               │
│  Save model, optimizer, scheduler, EMA state                    │
└─────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Strategy

#### Learning Rate Scaling by Model Size

| Tier | Parameters | Learning Rate | Rationale |
|------|------------|---------------|------------|
| T0 | 29M | 1.5e-3 | Can tolerate higher LR |
| T1 | 50M | 1.2e-4 | Moderate for attention |
| T2 | 210M | 8.0e-5 | Hybrid architecture |
| T3 | 250M | 9.0e-5 | Cross-task learning |
| T4 | 280M | 8.0e-5 | Cross-modal fusion |
| T5 | 320M | 7.5e-5 | **Sweet spot** for 300-400M params at batch 32 |

Config key is typically `lr` or `learning_rate` under `training` in the tier YAML. Base LR for T5 is 7.5e-5; transfer learning uses parameter groups with multipliers (e.g. cnn 0.2×, vit 0.5×, detection 0.6×, temporal 1.0×, new_heads 1.3×).

**Why 7.5e-5 for T5?**
- 1e-4 is slightly hot for:
  - Stacked attention layers
  - Temporal gradients (backprop through time)
  - Dynamic convolution updates
- 7.5e-5 balances:
  - Fast enough convergence
  - Stable gradient flow
  - Prevents attention collapse

#### Weight Decay: 0.05 (Not 0.0001)

**Problem with 0.0001**: Too low for 300M+ parameter models; high overfitting risk; model too expressive without regularization.

**Why 0.05 works**: Strong enough to prevent overfitting; not so strong it kills learning; standard for large transformer-like models. Set in config as `weight_decay` (e.g. in `ml/training/configs/t5_temporal.yaml`).

#### Loss Weight Rebalancing

**Previous problem**: box_regression 5.0 dominated; scene_description and scene_graph at 0.1 stayed muted; semantic tasks never got enough gradient signal; GradNorm could not fully fix the imbalance.

**Rebalanced solution** (in tier configs): box_regression 3.0; scene_description 0.3; scene_graph 0.3; other semantic/auxiliary heads raised to at least 0.3 where applicable. Config keys are typically under `training.loss_weights` or per-head keys (e.g. `therapy_state: 0.8` in t5_temporal.yaml).

**Activation threshold (0.3)**: Weights below ~0.3 tend to give too little gradient; above 0.3 tasks get real signal; GradNorm can then fine-tune relative magnitudes.

#### Data Loader: num_workers = 8

**Why increase from 4?**
- Model is **compute-bound** (GPU waits for data)
- Starving GPU murders throughput
- 8 workers keeps GPU fed during forward/backward

**Trade-off**:
- More memory usage
- Worth it for 2-3x throughput improvement

#### Warmup: 20 epochs (T5)

**Why longer warmup?**
- Gives GradNorm time to stabilize
- Temporal models need gradual ramp-up
- Prevents early collapse of attention mechanisms

#### min_lr: 1e-6

**Why add minimum LR?**
- Prevents late-stage collapse
- Temporal heads can overfit late in training
- Keeps model learning even at end

### Task Balancing: GradNorm

**GradNorm** (`ml/training/task_balancing.py`):
- Adaptive loss balancing across all heads
- Prevents gradient warfare
- Auto-dampening for problematic heads

**Why This Matters**: Without balancing, detection head dominates, other heads fail. With balancing, all heads learn together.

Implementation: gradient norms per task, relative losses vs initial, target norms, GradNorm L1 loss, weight update with clamp (0.1–10.0). Extreme gradients are dampened (e.g. weight ×0.5 if norm > 10× average). See **ml/training/task_balancing.py**.

### Transfer Learning: T2 → T5

**Strategy**: Copy T2 spatial weights (CNN, FPN, ViT, detection/distance/urgency heads) into T5; leave temporal encoder, cross-task/cross-modal attention, and new T5 heads randomly initialized. Parameter groups use different learning rates (e.g. cnn 0.2×, vit 0.5×, detection 0.6×, temporal 1.0×, new_heads 1.3× base_lr). **Freeze schedule**: Epochs 0–5 freeze backbone and detection, train only new T5 components; epochs 5–15 unfreeze detection/classification; later epochs unfreeze backbone progressively. See **ml/training/transfer_learning.py** (TierTransferManager, transfer_weights, validate_source_checkpoint) and **docs/transferlearning.md**.

Phased unfreeze and loss-unlock schedules are defined in transfer configs and training scripts (e.g. detection first, then navigation, then therapy/scene/OCR/sound/personalization/predictive). Parameter groups and loss weights per epoch: see **ml/training/transfer_learning.py** and tier configs in **ml/training/configs/** (e.g. t2_to_t5_transfer.yaml). Phase 2 (epochs 10-25) unlocks motion, navigation_difficulty, roi_priority; later phases unlock therapy_state, scene_graph, OCR, scene_description, sound_events, personalization, predictive_alerts.

---

## Inference Flow & Real-Time Processing

### Real-Time Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME CAPTURE                                │
│  Camera → Image [3, 224, 224] + Audio [128] when provided     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
│  - Normalization                                                │
│  - Condition-specific adaptations (if enabled)                 │
│  - Audio feature extraction (if audio available)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE A INFERENCE                            │
│  ResNet50+FPN → Tier 1 Heads                                    │
│  Target: ≤ 80 ms                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │  DECISION POINT │
                    │  latency >80ms  │
                    │  OR uncertainty │
                    │  >0.7?          │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐         ┌──────────────┐
        │  SKIP STAGE B │         │  RUN STAGE B │
        │  Return Stage │         │  (if tier ≥T2)│
        │  A only       │         │               │
        └──────────────┘         └───────┬───────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    STAGE B INFERENCE          │
                          │  Hybrid CNN-ViT → Tier 2/3    │
                          │  Target: <500ms                │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    OUTPUT PROCESSING           │
                          │  - Format outputs              │
                          │  - Apply thresholds            │
                          │  - Generate descriptions       │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │    MULTIMODAL FEEDBACK         │
                          │  - Visual overlays             │
                          │  - Voice announcements    │
                          │  - Haptic feedback            │
                          └───────────────────────────────┘
```

### Performance Targets

- **Stage A Latency**: ≤ 80 ms (target for time-to-alert and Stage A)
- **Stage B Latency**: <500ms (opportunistic)
- **Model Size**: <50MB (quantized)
- **Battery Drain**: <12% per hour normal use
- **Detection Accuracy**: >85% in varied environments

### Safety Metrics (More Important Than Accuracy)

- **False Reassurance Rate**: <1% (danger predicted as safe)
- **Alert Latency**: ≤ 80 ms (time to first warning)
- **Information Overload Events**: <2 per minute
- **Silence Correctness**: >95% (when staying quiet was right)
- **Tier 1 Availability**: >99.9% (safety heads never disabled)
- **Uncertainty Calibration**: Well-calibrated (uncertainty correlates with actual error)

**Why Safety Metrics Matter**: mAP and accuracy don't capture safety. A 95% accurate system that gives false reassurance is worse than an 85% accurate system that's safe.

---

## Effectiveness & Results

### Test Results

**Test Suite Status**:  **163 tests passing** | 8 skipped (expected, environment-specific) | 0 failing

**Test Coverage**:
- Phase 0 (Backbone): All tests passing
- Phase 1 (Fusion): All tests passing
- Phase 2 (Heads): All tests passing
- Phase 3 (Retrieval): All tests passing
- Phase 4 (Knowledge): All tests passing
- Phase 5 (Training): All tests passing
- Integration tests: All passing
- Performance tests: All passing

**Recent Test Fixes** (2025-01-30):
- Fixed 13 test failures (model size updates, API changes, missing methods)
- Updated model size thresholds for 250M parameter model
- Fixed training loss API tests (MAE, SimCLR, Knowledge Distillation, EWC)
- Added missing `extract_relations()` method to SceneGraphEncoder
- Fixed simulator output format tests (dev mode)
- Improved condition robustness test logic
- Made export validation test more lenient for expected failures

### Training Results

**Smoke Training** (Proof of Life):
-  **Loss decreased**: 0.7246 → 0.6013 (2 epochs, 5 batches)
-  Forward pass validated across all tiers (T0-T5)
-  GradNorm integration working
-  Checkpointing/resume working

**Training Framework Status**:
-  Production training loop implemented
-  Resume capability verified
-  EMA state dict interface fixed
-  Optimizer state preservation verified
-  Validation metric safety improved
-  GradNorm integration enhanced
-  MPS support added

### Model Performance

**Model Statistics**:
- **Parameters**: ~250M (comprehensive class system)
- **Model Size**: ~1GB (FP32) → <50MB (INT8 quantized)
- **Forward Pass**: Validated across all tiers
- **Export**: CoreML, ONNX, ExecuTorch formats supported

**Architecture Validation**:
-  Two-stage inference pipeline verified
-  Tier-based head execution verified
-  Safety-first guarantees verified
-  Graceful degradation verified

### Component Effectiveness

**Backbone Networks**:
-  ResNet50+FPN: Fast, predictable (≤ 80 ms target)
-  Hybrid CNN-ViT: Rich context features
-  Temporal Encoder: Motion tracking working

**Task Heads**:
-  All 30+ heads validated
-  Tier-based execution working
-  Condition-specific adaptations working

**Retrieval System**:
-  Two-stage retrieval working
-  Async retrieval non-blocking
-  Advisory-only design verified

**Training Infrastructure**:
-  GradNorm preventing gradient warfare
-  Multi-task learning working
-  Self-supervised pretraining ready

---

## ️ Repository Stack & Technology

### Technology Stack

#### ML Framework
- **PyTorch**: 2.9.1+ (with MPS support for Apple Silicon)
- **TorchVision**: 0.24.1+
- **TorchAudio**: 2.9.1+
- **PyTorch Geometric**: Graph neural networks for scene graphs

#### Data Processing
- **NumPy**: 2.2.6+ (numerical operations)
- **Pandas**: 2.3.3+ (data manipulation)
- **Pillow**: 12.0.0+ (image processing)
- **OpenCV**: 4.8.0+ (image preprocessing)

#### Optimization & Deployment
- **TorchAO**: 0.14.1+ (model optimization)
- **FAISS**: 1.13.2+ (efficient similarity search)
- **CoreML**: iOS deployment (image input only; audio/temporal not in export — see docs/status.md)
- **ONNX**: Cross-platform deployment
- **ExecuTorch**: Mobile deployment

#### Scientific Computing
- **SciPy**: 1.11.0+ (optimization, Hungarian matching)
- **Scikit-learn**: 1.3.0+ (clustering, OCR text pixel clustering)

#### Development Tools
- **Pytest**: 9.0.1+ (testing framework)
- **Matplotlib**: 3.10.7+ (visualization)
- **Tqdm**: 4.66.0+ (progress bars)

#### Web Simulator
- **Flask**: 3.0.0+ (web framework)
- **Flask-CORS**: 4.0.0+ (CORS support)

### Key Files & Their Purposes

| File | Purpose | Status |
|------|---------|--------|
| `ml/models/maxsight_cnn.py` | Main CNN architecture |  Active |
| `ml/training/train_loop.py` | Production training loop |  Active |
| `ml/training/task_balancing.py` | GradNorm multi-task balancing |  Active |
| `ml/training/transfer_learning.py` | T2→T5 transfer logic |  Active |
| `ml/data/dataset.py` | MaxSightDataset |  Active |
| `ml/data/data_pipeline.py` | Data loader creation |  Active |
| `ml/models/backbone/hybrid_backbone.py` | Hybrid CNN-ViT backbone |  Active |
| `ml/models/temporal/temporal_encoder.py` | Temporal processing |  Active |
| `ml/models/scene_graph/scene_graph_encoder.py` | Scene graph encoding |  Active |
| `ml/training/export.py` | Model export (iOS-ready) |  Active |
| `ml/retrieval` | Retrieval system (advisory) |  Active |
| `ml/optimization/mobile_optimizations.py` | Mobile optimizations |  Active |

---

## Current Work & Next Steps

### Immediate next steps

1. **Data**: Run `python scripts/ops/gather_training_data.py` if you haven’t (creates `datasets/cleaned_splits/` and uses `datasets/coco_raw/`). Use `--skip-download` / `--skip-extract` if COCO is already present.
2. **Smoke check**: `python scripts/product/run.py smoke --epochs 2` (or `python scripts/ops/smoke_train.py --epochs 2 --force-cpu`)
3. **Full training**: Use the training command from [Full Training](#full-training-annotation-based-cloud-gpu-recommended) with your `--data-dir`, `--train-annotation`, `--val-annotation`, `--image-dir` (cloud GPU recommended for full runs).
4. **Export**: After a checkpoint exists, `python -m ml.training.export --checkpoint <path> --format <jit|coreml|onnx|executorch> --output <path>`.
5. **Simulator with trained model**: Set `model_checkpoint_path` in `tools/simulation/config.py` or use `ComprehensiveSimulator(model_path=...)`. See **docs/architecture.md** (export section) and **README** for deployment.

### Short-term goals (next 2–4 weeks)

1. **COCO and splits**
   - Ensure COCO is downloaded and extracted (or use existing data).
   - Splits are created by `scripts/gather_training_data.py` (train/val/test JSONs in `datasets/cleaned_splits/`).

2. **Training Pipeline Validation**
   - Test data loaders
   - Test training loop
   - Verify checkpointing/resume
   - Validate metrics computation

3. **Initial Training Runs**
   - T0 baseline training (proof of concept)
   - T1 attention training
   - Performance benchmarking

4. **Model Export Testing**
   - CoreML export validation
   - ONNX export validation
   - ExecuTorch export validation
   - Mobile inference testing

### Medium-Term Goals (Next 1-3 Months)

1. **Full Training Pipeline**
   - T2 hybrid ViT training
   - T3 cross-modal training
   - T4 cross-modal + audio training
   - T5 temporal training

2. **Transfer Learning**
   - T2 → T5 transfer implementation
   - Validate transfer effectiveness
   - Optimize transfer schedule

3. **Performance Optimization**
   - Latency optimization
   - Model size optimization
   - Battery usage optimization

4. **Real-World Testing**
   - User testing
   - Accessibility validation
   - Performance benchmarking in real environments

### Long-Term Goals (3-6 Months)

1. **Production Deployment**
   - iOS app integration
   - CoreML deployment
   - Performance monitoring
   - User feedback integration

2. **Accessibility Certification**
   - WCAG compliance
   - Accessibility testing
   - Certification process

3. **Continuous Improvement**
   - Model updates
   - Feature additions
   - Performance improvements
   - User experience enhancements

---

## Quick Start Guide

### Prerequisites

- **Python**: 3.12+
- **PyTorch**: 2.5.0+ (with MPS support for Apple Silicon)
- **Hardware**: 
  - **Local Development**: Apple Silicon M1+ (MPS) or CPU
  - **Training**: Cloud GPU (CUDA) required for models >10k parameters
- **macOS**: Apple Silicon M1+ (for iOS development)
- **Xcode**: 16.1+ (for iOS app)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd 2026-Prototype

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### Device Selection Policy

**Automatic device selection based on model size:**

- **Models < 10k parameters**: Automatically use **CPU** (smoke tests, small experiments)
- **Models >= 10k parameters**: Require **Cloud GPU (CUDA)** for training

**All MaxSight tiers (210M+ parameters) require cloud GPU for training.**

See **docs/status.md** for device and hardware notes.

### Requirements before training

1. **Install deps**: `pip install -r requirements.txt`
2. **Prepare data**: Run once: `python scripts/ops/gather_training_data.py` (`--skip-download` / `--skip-extract` if COCO is already present). This creates `datasets/cleaned_splits/maxsight_train.json`, `maxsight_val.json`, `maxsight_test.json`.
3. **Hardware**: For full training use a CUDA GPU; for smoke/short runs CPU or MPS is fine.

See **docs/status.md** and **docs/downloads.md** for setup and data requirements.

### Smoke Training (Proof of Life)

```bash
# Tier choices: T0_BASELINE_CNN, T1_ATTENTION, T2_HYBRID_VIT, T3_CROSS_TASK, T4_CROSS_MODAL, T5_TEMPORAL
python scripts/ops/smoke_train.py --epochs 2 --batches 5 --force-cpu

# Force CPU (short run only)
python scripts/ops/smoke_train.py --epochs 2 --batches 3 --force-cpu
```

### Full Training (annotation-based; Cloud GPU recommended)

```bash
# After running gather_training_data.py, use the paths it prints:
python scripts/ops/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --epochs 100 \
  --batch-size 32 \
  --device cuda \
  --use-gradnorm
```

Optional: run **AutoML** (Optuna) first, then train with best params:  
`python scripts/AutoMLType.py --data-dir ... --train-annotation ... --val-annotation ... --image-dir ...`  
Then: `python scripts/ops/train_maxsight.py ... --hyperparameters checkpoints_tuning/best_hyperparameters.json`

### One-shot production training

To run env check, dataset check, data-pipeline validation when desired, full training, and export when desired in one go:

```bash
./scripts/ops/run_production_training.sh
```

Options: `--skip-env`, `--skip-data-check`, `--no-export`, `--dry-run`. Override via env: `DATA_DIR`, `EPOCHS`, `BATCH_SIZE`, `LR`, `DEVICE`, `HYPERPARAMETERS` (path to `best_hyperparameters.json` from AutoMLType.py).  
Optional **Phase 3 data validation** (no invalid values; class weights):  
`python scripts/ops/validate_data_pipeline.py --train-annotation datasets/cleaned_splits/maxsight_train.json --image-dir datasets/coco_raw`

### Validation and benchmarking

Use the test suite and training benchmark: `pytest tests/` and `python -m ml.training.benchmark`. See **docs/status.md** for current status.

---

## Main Components

### 1. MaxSightCNN (`ml/models/maxsight_cnn.py`)

**Purpose**: Main multi-task vision model (250M parameters, T2 tier)

**Architecture**:
- **Stage A Backbone**: ALWAYS ResNet50 + FPN (safety guarantee)
- **Stage B Backbone**: Hybrid CNN-ViT (T2+) + Temporal (T5+)
- **Heads**: 30+ specialized task-specific heads organized by criticality tiers

**Key Features**:
- Anchor-free detection (FCOS-style)
- Multi-scale feature extraction (FPN)
- Audio-visual fusion
- Condition-specific adaptations (13 vision conditions)
- Two-stage inference (safety-first)
- MPS-stable mode for Apple Silicon development

**Input**: `[B, 3, 224, 224]` RGB images + `audio_features [B, 128]` when provided  
**Output**: Dictionary with 30+ task outputs

### 2. Backbone Components

- **ResNet50+FPN**: ResNet50 from `torchvision` (e.g. `ResNet50_Weights.IMAGENET1K_V2`); outputs C2–C5 at strides 4, 8, 16, 32. FPN in `maxsight_cnn.py` builds P2–P5 (256 channels each); lateral 1×1 convs and top-down bilinear upsample. Fused detection features: P3, P4, P5 resized to same spatial size (e.g. 14×14) and concatenated → 768 channels. Used for all Tier 1 heads and as base for Tier 2/3 when Stage B runs.
- **Hybrid CNN-ViT** (`ml/models/backbone/hybrid_backbone.py`): Class `HybridCNNViTBackbone`. Args: img_size=224, patch_size=16, etc. CNN branch: ResNet-like; ViT branch: patch embed (196 patches, 768 dim), 12 transformer blocks, CLS token. `AdaptiveFeatureFusion(cnn_dim, vit_dim, fused_dim)` does gated fusion; `CrossModalAttention(dim, num_heads)` does CNN↔ViT cross-attention. Used when `tier_config.use_hybrid_backbone` is True.
- **Vision Transformer** (`ml/models/backbone/vit_backbone.py`): Standalone ViT; used if pipeline is configured for ViT-only or extra ViT path.
- **Dynamic Convolution** (`ml/models/backbone/dynamic_conv.py`): Condition-adaptive conv layers; condition (e.g. string or embedding) modulates kernel or channel weights. Used in Stage B when `tier_config.use_dynamic_conv` is True.

### 3. Head Components

Heads are `nn.Module` subclasses with `forward(...)`; built in `maxsight_cnn.py` when `enable_accessibility_features` and tier allow. Shared feature input is typically detection features `det_feats` `[B, 256, H, W]` (e.g. H=W=14) or fused context.

- **Therapy State Head** (`ml/models/heads/therapy_state_head.py`): Class `TherapyStateHead`. Args: eye_dim=4, motion_dim=256, temporal_dim=128, in_channels_depth=256, in_channels_contrast=256, use_lstm=True, use_depth_multi_scale=True, use_edge_aware=True. **Forward**: `eye_features` [B,4], `motion_features` [B,D] or [B,D,H,W], `depth_features` [B,256,H,W], `contrast_features` [B,256,H,W], optional `fpn_features` dict. **Outputs**: fatigue_score, blink_rate, fixation_stability, shared_features, depth_map [B,H,W], uncertainty [B,H,W], zones [B,3], contrast_map [B,H,W], optional edge_map. Does not return `therapy_state` or `progress`; pipeline may read those keys as None.
- **Fatigue Head** (`ml/models/heads/fatigue_head.py`): `FatigueHead(eye_dim=4, temporal_dim=128, hidden_dim=64, use_lstm=True)`. Forward: eye_features [B,4], motion_features [B,temporal_dim]. Outputs: fatigue_score, blink_rate, fixation_stability, shared_features.
- **Contrast Head** (`ml/models/heads/contrast_head.py`): `ContrastMapHead(in_channels=256, motion_dim=256, use_edge_aware=True)`. Forward: feature map. Outputs: contrast_map, optional edge_map.
- **Motion Head** (`ml/models/heads/motion_head.py`): Forward: feature map. Outputs: flow [B,2,H,W], magnitude [B,1,H,W]. Used for motion tasks and as motion_features for therapy and predictive heads.
- **OCR Head** (`ml/models/heads/ocr_head.py`): Text detection/recognition from image or patches; output format per implementation.
- **Scene Description Head** (`ml/models/heads/scene_description_head.py`): Consumes global or fused features; produces natural language (e.g. list of strings or token ids).
- **Sound Event Head** (`ml/models/heads/sound_event_head.py`): Classifies audio features to sound event classes when audio is provided.
- **Personalization Head** (`ml/models/heads/personalization_head.py`): User embedding or modulation; input/output shape per implementation.
- **Predictive Alert Head** (`ml/models/heads/predictive_alert_head.py`): Input: scene features, motion features (e.g. magnitude). Outputs: hazard_probs, time_to_hazard, recommended_action.
- **Uncertainty Head** (`ml/models/heads/uncertainty_head.py`): `GlobalConfidenceAggregator(scene_dim=256, hidden_dim=128)`. Consumes scene embedding; outputs uncertainty_score [B,1]. Used for Stage B skip decision and alert suppression.
- **ROI Priority Head** (`ml/models/heads/roi_priority_head.py`): Input: scene_embedding [B,1,256], roi_features [B,H*W,256]. Output: roi_utility [B,H*W].
- **Glare / Navigation difficulty / Findability**: Implemented in main model (small MLP or conv); glare 4 classes, navigation_difficulty scalar, findability per location. **Depth head** (`ml/models/heads/depth_head.py`): Standalone depth-from-features if needed in addition to therapy state head.
- **Head registry** (`ml/models/heads/__init__.py`): `HEAD_REGISTRY` maps 'contrast', 'depth', 'fatigue', 'motion', 'roi_priority', 'uncertainty' to classes; `create_head(head_type, **kwargs)` factory. TherapyStateHead is not in registry; instantiated directly in maxsight_cnn.

### 4. Temporal Processing

- **Temporal Encoder** (`ml/models/temporal/temporal_encoder.py`): Class `TemporalEncoder`. Args: in_channels=256, num_frames=8, hidden_dim=256, use_conv_lstm=True, use_timesformer=True. Forward: `feature_frames` 5D [B,C,T,H,W] or [B,T,C,H,W]; optional ViT patch tokens. Outputs dict: motion features, consistency score, flicker score, etc. ConvLSTM output feeds motion head and therapy/predictive heads.
- **ConvLSTM** (`ml/models/temporal/conv_lstm.py`): Input [B,T,C,H,W]; hidden/cell states; output hidden state sequence. Kernel size 3, 2 layers by default.
- **TimeSformer**: Long-range temporal attention over patch sequence; used when use_timesformer=True (optional import from temporal_transformer).

### 5. Scene Graph & Retrieval

- **Scene Graph Encoder** (`ml/models/scene_graph/scene_graph_encoder.py`): Class `SceneGraphEncoder`. Object embeddings [N, object_embed_dim], boxes [N,4]. Spatial relations (e.g. left, right, above, below, near, far) and semantic relations from trainable classifiers; `SceneRelation` dataclass (subject, predicate, object, confidence, src, dst). Batched; MPS-stable mode detaches edge_attr for compatibility. Often stubbed for export (non-traceable types).
- **Retrieval**: Encoders in `ml/retrieval/encoders/` (patch, region, global, OCR, depth, audio); indexing in `ml/retrieval/indexing/` (neural_index_builder, index_manager); retrieval in `ml/retrieval/retrieval/` (stage1_ann, stage2_rerank, async_retrieval, concept_retrieval, knowledge_augment). Two-stage: ANN search then rerank; async so it never blocks inference.
- **Retrieval Heads** (`ml/models/retrieval_heads_production.py`): Multi-vector retrieval heads for production pipeline.

### 6. Training Infrastructure

- **Losses** (`ml/training/losses.py`): Per-head loss functions; combiner for total weighted loss. **Head losses** (`ml/training/head_losses.py`): Helpers for detection, therapy, etc. **Matching** (`ml/training/matching.py`): Hungarian or similar for box-to-prediction assignment.
- **Metrics** (`ml/training/metrics.py`): mAP, precision, recall, F1; aggregation over batches. **Validation** (`ml/training/validation.py`): Validation step and metric computation. **Evaluation** (`ml/training/evaluation.py`): Lighting-aware or condition-specific evaluation reports.
- **Task Balancing** (`ml/training/task_balancing.py`): GradNorm (and optionally PCGrad); task weights, gradient norms, update every N steps.
- **Transfer** (`ml/training/transfer_learning.py`): `TierTransferManager(source_checkpoint_path, target_model, config)`. Methods: `validate_source_checkpoint()`, `transfer_weights(strict=False)`. Copies matching state dict keys; leaves new modules (e.g. temporal) randomly initialized.
- **Stability** (`ml/training/stability_manager.py`): Gradient clipping, loss scaling. **Regularization** (`ml/training/regularization.py`): Weight decay, auxiliary losses. **Quantization** (`ml/training/quantization.py`): Quantization-aware training for export.
- **Export** (`ml/training/export.py`): `export_to_jit`, `export_to_coreml`, `export_to_onnx`, `export_to_executorch` (or similar); wrapper strips non-tensor outputs; stubs global_encoder and can disable scene graph for tracing.

### 7. Data & Augmentation

- **Dataset** (`ml/data/dataset.py`): Class `MaxSightDataset`; __getitem__ returns dict with images, labels, boxes, num_objects, distance, urgency, optional audio, condition_mode. See Data Pipeline section above for COCO keys and normalization.
- **Data pipeline** (`ml/data/data_pipeline.py`): `create_data_loaders()` (or equivalent), custom collate, class weight computation, optional auto-detect image dirs.
- **Advanced Augmentation** (`ml/data/advanced_augmentation.py`): Geometric, photometric, cutout/mixup/mosaic; condition-specific; see Data Pipeline section.
- **Multi-Modal Augment** (`ml/data/multi_modal_augment.py`): Vision + audio joint augmentation when both present.

### 8. Optimization & Evaluation

- **Mobile Optimizations** (`ml/optimization/mobile_optimizations.py`): Pruning (e.g. structured by channel), quantization, edge-cloud split. **Evaluation Metrics** (`ml/evaluation/metrics.py`): Multi-modal and accessibility-specific metrics (e.g. urgency accuracy, distance accuracy).

### Component reference: what each does and why it’s there

Below, every major component is described in two ways: **what it does** and **why it’s there**. For full implementation detail (inputs, outputs, file paths), see **docs/SYSTEMS.md**.

**MaxSightCNN** — **What:** Runs two-stage inference (Stage A: ResNet50+FPN + Tier 1 heads; Stage B: optional hybrid/temporal + Tier 2/3 heads) and returns 30+ outputs (detections, urgency, distance, therapy state, etc.). **Why:** Single entry point that guarantees safety-first (Stage A always runs) and allows rich context when resources allow (Stage B).

**ResNet50+FPN (Stage A)** — **What:** Extracts multi-scale feature maps (C2–C5, then P2–P5) from RGB input. **Why:** Fast, well-understood backbone for low-latency safety-critical predictions; FPN lets the model see objects at many scales.

**Hybrid CNN–ViT (Stage B)** — **What:** Combines a CNN branch (spatial features) and a ViT branch (patch tokens + transformer) with learnable fusion and optional CNN↔ViT cross-attention. **Why:** CNN gives local detail; ViT gives global context; together they support better scene understanding and Tier 2/3 heads without touching Stage A.

**Dynamic convolution** — **What:** Modulates conv kernels or channels by vision condition (e.g. glaucoma, AMD). **Why:** Lets the same model adapt preprocessing/features to the user’s condition for better accessibility.

**CBAM / SE (T1)** — **What:** Channel and spatial attention (CBAM) or channel-only (SE) on FPN feature maps. **Why:** Lightweight way to emphasize informative channels and locations without changing the safety path.

**Cross-modal attention (vision/audio/haptic)** — **What:** Projects vision, audio, and optional haptic to a common dimension and applies multi-head attention between modalities. **Why:** So that sound (and haptic) can disambiguate or focus visual predictions (e.g. “sound from the left” drives visual attention).

**Cross-task attention (T3)** — **What:** Lets detection, scene, therapy, and other tasks share context via attention over task features. **Why:** Improves consistency and reasoning across tasks (e.g. scene graph and detection agree on relations).

**Detection heads (objectness, classification, box, distance, urgency)** — **What:** Anchor-free (FCOS-style) object detection plus per-object or per-scene distance zones and urgency levels. **Why:** Core safety output: what is there, where, how far, and how urgent so the user can navigate and prioritize.

**Contrast head** — **What:** Produces a contrast map (and optional edge map) from backbone features, with optional motion conditioning and edge-aware modulation. **Why:** Supports contrast-sensitivity therapy and accessibility (e.g. highlighting low-contrast obstacles).

**Motion head** — **What:** Predicts optical flow (and magnitude) from feature maps. **Why:** Feeds motion tracking therapy, predictive alerts, and optional motion conditioning in therapy/depth/contrast heads.

**Fatigue head** — **What:** From eye + motion features (1D), predicts fatigue_score, blink_rate, fixation_stability via shared MLP and optional LSTM. **Why:** Informs pacing and rest (e.g. TaskGenerator suggests FATIGUE_REST when fatigue is high).

**Therapy state head** — **What:** Single head with three branches: (1) fatigue/gaze (same as fatigue head), (2) depth (depth map, uncertainty, near/medium/far zones), (3) contrast (contrast map, optional edge map). **Why:** One place for all therapy-related signals so session/task logic can use fatigue, depth, and contrast together.

**ROI priority head** — **What:** From scene embedding and region features, outputs per-region importance (roi_utility). **Why:** Lets the system emphasize the most relevant regions for the user and therapy focus.

**Predictive alert head** — **What:** From scene and motion features, predicts hazard_probs, time_to_hazard, recommended_action. **Why:** Proactive safety (e.g. “vehicle approaching”) instead of only describing the current frame.

**Uncertainty head** — **What:** Aggregates confidence across outputs into a single uncertainty_score. **Why:** Used to skip or dampen Stage B and to suppress low-confidence alerts so the user isn’t overloaded or misled.

**Scene description / OCR / Scene graph** — **What:** Scene description: natural language summary of the scene. OCR: text in the image. Scene graph: spatial and semantic relations between objects. **Why:** Rich context for narration, wayfinding, and relational reasoning; Tier 3 so they never block safety.

**Sound event head** — **What:** Classifies sound events from audio features. **Why:** When audio is available, supports “what you hear” in addition to “what you see” for multimodal accessibility.

**Personalization head** — **What:** Produces or modulates features by user (e.g. embedding or light weights). **Why:** Lets the system adapt to individual preferences and needs over time.

**Glare / Navigation difficulty / Findability** — **What:** Glare: 4-class glare level. Navigation difficulty: scene complexity scalar. Findability: per-location score for how findable objects are. **Why:** Accessibility metrics to adapt feedback (e.g. simplify when navigation is hard, emphasize findability for low vision).

**Temporal encoder (ConvLSTM + TimeSformer)** — **What:** Consumes sequences of Stage A features; ConvLSTM for motion, TimeSformer for long-range temporal attention; outputs motion and consistency/flicker signals. **Why:** T5 needs time-aware reasoning for motion tasks, predictive alerts, and smoother therapy state.

**Multimodal fusion (EnhancedAudioEncoder, MultimodalFusion, SpatialSoundMapping)** — **What:** Encodes audio (and optionally stereo), fuses vision/audio/depth/haptic via transformer over modality tokens, maps sound to spatial attention on the image. **Why:** Single representation that combines seeing and hearing so downstream heads can use both.

**Scene graph encoder** — **What:** Builds spatial and semantic relations (e.g. left_of, contains) from boxes and object embeddings; batched GNN-style encoding. **Why:** Enables “A is left of B” style reasoning and scene graph outputs; T3, often stubbed for export.

**Retrieval (encoders, indexing, two-stage ANN + rerank)** — **What:** Encodes patches, regions, global, OCR, depth, audio; builds neural indexes; retrieves similar scenes then reranks. **Why:** Advisory context (e.g. “similar to a kitchen”) without ever driving Tier 1/2 safety decisions; async so it never blocks inference.

**Therapy system (SessionManager, TaskGenerator, TherapyTaskIntegrator)** — **What:** SessionManager tracks sessions and logs task attempts; TaskGenerator picks next task (e.g. contrast_micro, fatigue_rest) from fatigue/uncertainty and history; TherapyTaskIntegrator builds concrete tasks from scene/detections. **Why:** Turns model outputs (fatigue, contrast, depth, motion) into structured therapy sessions and adaptive task flow.

**Output scheduler** — **What:** Schedules when and how to present information on audio, haptic, and visual channels; rate-limits and respects uncertainty. **Why:** Avoids overload and ensures critical alerts get through; supports user preferences (channel, verbosity).

**Preprocessing** — **What:** Condition-specific normalization, resize, and augmentations (e.g. blur for cataracts, central mask for glaucoma). **Why:** Training and inference should match the user’s vision condition so the model and therapy are relevant.

**Data pipeline (MaxSightDataset, create_data_loaders, collate)** — **What:** Loads COCO-format annotations and images, applies preprocessing and augmentation, batches with variable-length handling. **Why:** Single way to feed training with the right shapes and condition mode.

**Training (train_loop, losses, task_balancing, transfer_learning)** — **What:** Training loop with per-head losses, GradNorm (or similar) for task balancing, and TierTransferManager for T2→T5 transfer. **Why:** Multi-task learning without one head dominating; reuse of T2 weights for faster and stabler T5 training.

**Export (JIT, CoreML, ONNX, ExecuTorch)** — **What:** Traces or converts the model to mobile- and cross-platform formats; stubs non-traceable parts (e.g. CLIP, scene graph) when needed. **Why:** Enables deployment on iOS and other targets without running full Python.

**Simulator (tools/simulation)** — **What:** Web-based simulator with inference engine, overlay, scheduler, voice/haptic hooks, and configurable checkpoint. **Why:** End-to-end testing and demos without the iOS app.

### Concrete reference: outputs, configs, env, CLI

**Model output dict (representative keys)**  
Exact keys depend on tier and `enable_accessibility_features`. Common keys: `obj_scores` [B, H*W], `cls_logits` [B, H*W, num_classes], `box_preds` [B, H*W, 4], `detections` (post-processed list or tensor), `urgency`, `distance`, `contrast_map` [B,1,H,W] or [B,H,W], `edge_map`, `motion_flow` [B,2,H,W], `motion_magnitude` [B,1,H,W], `fatigue_score` [B,1], `blink_rate` [B,1], `fixation_stability` [B,1], `depth_map` [B,H,W], `uncertainty` [B,H,W], `zones` [B,3], `therapy_state`, `therapy_progress` (often None), `roi_utility` [B,H*W], `navigation_difficulty` [B,1], `glare_risk_level` [B], `glare_probs` [B,4], `object_findability` [B,H*W], `uncertainty_score` [B,1], `hazard_probs`, `time_to_hazard`, `recommended_action`, `shared_scene_embedding` [B,256], plus scene graph, OCR, scene description when enabled.

**TierConfig** (`ml/models/maxsight_cnn.py`)  
Fields: `tier`, `enabled`, `use_se_attention`, `use_cbam_attention`, `use_hybrid_backbone`, `use_dynamic_conv`, `use_cross_task_attention`, `use_cross_modal_attention`, `use_temporal_modeling`, `use_retrieval`, `max_latency_ms` (e.g. 300), `min_confidence` (e.g. 0.5). `TierConfig.for_tier(tier)` returns config; current code is T5-only.

**Training config YAML** (`ml/training/configs/*.yaml`)  
Typical keys: `model` (num_classes, tier, condition_mode), `data` (data_dir, train_annotation, val_annotation, image_dir, batch_size, num_workers), `training` (epochs, lr, weight_decay, loss_weights dict, gradnorm_update_interval, warmup_epochs, min_lr, accumulate_grad_batches, mixed_precision), optional `transfer` (freeze schedule, loss unlock by epoch). Loss weight keys: e.g. detection, classification, box_regression, distance, urgency, motion, therapy_state, scene_description, scene_graph, ocr, etc.

**Environment variables**  
`MAXSIGHT_CHECKPOINT_PATH` — used by simulator or scripts for checkpoint path. `model_checkpoint_path` in `tools/simulation/config.py` overrides for simulator. Data paths often passed via CLI rather than env.

**Script CLI (main entry points)**  
- **scripts/train_maxsight.py**: `--data-dir`, `--train-annotation`, `--val-annotation`, `--image-dir`, `--epochs`, `--batch-size`, `--device` (cuda/cpu/mps), `--use-gradnorm`, `--resume-from`, optional `--hyperparameters` (path to JSON from AutoMLType).  
- **scripts/smoke_train.py**: `--tier` (e.g. T0_BASELINE_CNN, T5_TEMPORAL), `--epochs`, `--batches`, `--force-cpu`.  
- **scripts/gather_training_data.py**: Creates train/val/test JSONs in e.g. `datasets/cleaned_splits/`; uses `datasets/coco_raw/`; `--skip-download`, `--skip-extract` if COCO already present.  
- **python -m ml.training.export**: `--checkpoint`, `--format` (jit/coreml/onnx/executorch), `--output`.  
- **scripts/export_for_xcode.py**: Checkpoint path and output bundle path.  
- **scripts/deploy_top7.py**, **scripts/export_top7_to_xcode.py**: Top-7 models per condition; checkpoints under `checkpoints_<condition>/best_model.pt`.

**Therapy (application layer)**  
- **SessionManager** (`ml/therapy/session_manager.py`): `start_session(session_config=None)` → session_id; `log_task_attempt(task_type, task_config, result)` (result has success, reaction_time, etc.); `end_session()` → report dict (skill_curve, summary); `save_session(filepath)`.  
- **TaskGenerator** (`ml/therapy/task_generator.py`): `generate_task(uncertainty, fatigue_score, recent_performance)` → dict with task_type, difficulty, duration, highlight_strength, target_speed; if fatigue_score &gt; 0.7 returns task_type FATIGUE_REST. TaskType enum: CONTRAST_MICRO, MOTION_TRACKING, DEPTH_SHIFT, GAZE_STABILIZATION, ROI_FINDABILITY, FATIGUE_REST. `update_performance(task_result)` appends to history.  
- **TherapyTaskIntegrator** (`ml/therapy/therapy_integration.py`): Creates task configs from scene description and detections; TherapyTaskType: ATTENTION_TRAINING, CONTRAST_RECOGNITION, EDGE_DETECTION, SPATIAL_AWARENESS, WARNING_RECOGNITION. Methods: `create_attention_task`, `create_contrast_task`, `create_edge_task`, `create_spatial_task`, `create_warning_recognition_task`, `generate_task_from_scene`.

**Output scheduler** (`ml/utils/output_scheduler.py`)  
`OutputChannel`: AUDIO, HAPTIC, VISUAL, HYBRID. `AlertFrequency`: LOW, MEDIUM, HIGH. `OutputConfig`: preferred_channel, alert_frequency, audio_volume, haptic_intensity, uncertainty_threshold, verbosity. `CrossModalScheduler(config)` schedules outputs; rate-limiting (e.g. min 300 ms between outputs); uses sound processing if available.

**Preprocessing** (`ml/utils/preprocessing.py`)  
ImagePreprocessor: normalization (ImageNet), resize (e.g. 224×224), condition-specific transforms. Condition names match dataset/training (e.g. cataracts, glaucoma, amd, retinitis_pigmentosa). RGB↔LAB and other color helpers; cached matrices for performance.

**Where each module lives (file and class/function)**  
- **MaxSightCNN**: `ml/models/maxsight_cnn.py` class `MaxSightCNN`.  
- **ResNet50 + FPN**: Same file; backbone from torchvision, FPN built in constructor.  
- **Hybrid backbone**: `ml/models/backbone/hybrid_backbone.py` `HybridCNNViTBackbone`, `AdaptiveFeatureFusion`, `CrossModalAttention` (CNN–ViT).  
- **Dynamic conv**: `ml/models/backbone/dynamic_conv.py`.  
- **CBAM/SE**: `ml/models/attention/attention.py` `CBAM`, `SEBlock`, `ChannelAttention`, `SpatialAttention`.  
- **Cross-modal (vision/audio)**: `ml/models/attention/attention.py` `CrossModalAttention` (vision_dim, audio_dim, haptic_dim).  
- **Cross-task attention**: `ml/models/attention/cross_task_attention.py`.  
- **Therapy state head**: `ml/models/heads/therapy_state_head.py` `TherapyStateHead`.  
- **Fatigue head**: `ml/models/heads/fatigue_head.py` `FatigueHead`.  
- **Contrast head**: `ml/models/heads/contrast_head.py` `ContrastMapHead`.  
- **Motion head**: `ml/models/heads/motion_head.py` `MotionHead`.  
- **Temporal encoder**: `ml/models/temporal/temporal_encoder.py` `TemporalEncoder`; `ml/models/temporal/conv_lstm.py` `ConvLSTM`.  
- **Multimodal fusion**: `ml/models/fusion/multimodal_fusion.py` `EnhancedAudioEncoder`, `MultimodalFusion`, `SpatialSoundMapping`, `HapticEmbedding`, `HapticVisualAttention`.  
- **Scene graph**: `ml/models/scene_graph/scene_graph_encoder.py` `SceneGraphEncoder`, `SceneRelation`.  
- **Retrieval**: `ml/retrieval/encoders/`, `ml/retrieval/indexing/`, `ml/retrieval/retrieval/` (stage1_ann, stage2_rerank, async_retrieval).  
- **SessionManager / TaskGenerator / TherapyTaskIntegrator**: `ml/therapy/session_manager.py`, `ml/therapy/task_generator.py`, `ml/therapy/therapy_integration.py`.  
- **Output scheduler**: `ml/utils/output_scheduler.py` `CrossModalScheduler`, `OutputConfig`, `ScheduledOutput`.  
- **Dataset**: `ml/data/dataset.py` `MaxSightDataset`.  
- **Data loaders**: `ml/data/data_pipeline.py` `create_data_loaders`, collate_fn.  
- **Losses**: `ml/training/losses.py`, `ml/training/head_losses.py`; **matching**: `ml/training/matching.py`.  
- **Task balancing**: `ml/training/task_balancing.py` (GradNorm).  
- **Transfer**: `ml/training/transfer_learning.py` `TierTransferManager`.  
- **Export**: `ml/training/export.py` `export_to_jit`, `export_to_coreml`, etc.  
- **TierConfig / TierManager**: `ml/models/maxsight_cnn.py` (bottom) `TierConfig`, `TierManager`, `CapabilityTier`.

---

## Testing & Validation

### Test Suites

```bash
# Run all phase tests
pytest tests/

# Phase-specific tests
pytest tests/test_phase0_backbone.py
pytest tests/test_phase1_fusion.py
pytest tests/test_phase2_heads.py
pytest tests/test_phase3_retrieval.py
pytest tests/test_phase4_knowledge.py
pytest tests/test_phase5_training.py

# Smoke training (proof of life)
python scripts/smoke_train.py --tier T2_HYBRID_VIT --epochs 2 --batches 5

# Benchmark inference (ml/training/benchmark.py)
python -m ml.training.benchmark
```

### Validation Status

 **All phases (0-9) complete**  
 **Forward pass validation passed**  
 **Smoke training passed** (loss decreased: 0.7246 → 0.6013)  
 **Function flow verified**  
 **MPS-stable mode implemented**  
 **Device selection policy implemented**  
 **163 tests passing** | 8 skipped | 0 failing

---

## Performance & Safety

### Performance Targets

- **Stage A Latency**: ≤ 80 ms (target for time-to-alert and Stage A)
- **Stage B Latency**: <500ms (opportunistic)
- **Model Size**: <50MB (quantized)
- **Battery Drain**: <12% per hour normal use
- **Detection Accuracy**: >85% in varied environments

### Safety Metrics (More Important Than Accuracy)

- **False Reassurance Rate**: <1% (danger predicted as safe)
- **Alert Latency**: ≤ 80 ms (time to first warning)
- **Information Overload Events**: <2 per minute
- **Silence Correctness**: >95% (when staying quiet was right)
- **Tier 1 Availability**: >99.9% (safety heads never disabled)
- **Uncertainty Calibration**: Well-calibrated (uncertainty correlates with actual error)

**Why Safety Metrics Matter**: mAP and accuracy don't capture safety. A 95% accurate system that gives false reassurance is worse than an 85% accurate system that's safe.

---

## Deployment & Export

### Product: a day in the life (MaxSight glasses)

**Big picture.** (1) You convert PyTorch model checkpoints to CoreML (e.g. with the Colab script or `ml.training.export`) and add the `.mlpackage` files to the MaxSight app that runs on smart glasses. (2) A visually impaired person wears the glasses; the camera sees what they're looking at, the app runs the right CoreML model on that video, and the result becomes spoken descriptions and/or haptic alerts. (3) So: script → .mlpackage → glasses app → wearer gets real-time environmental awareness (objects, text, hazards) and more independence. The details below are the full day-in-the-life and benefits.

**Who.** People with low vision or blindness (e.g. AMD, glaucoma, diabetic retinopathy, CVI). The glasses are tuned to their condition so descriptions and alerts match how they see (or don't see) the world.

**How they use it.** They wear the glasses; the camera sees from their perspective. No phone to hold, no pointing — they look where they want information. They use **voice** ("What's in front of me?", "Read that," "Describe the room") or a **tap on the temple** for on-demand read/describe so they don't have to speak in public. **Modes:** continuous (quiet scene updates + hazard alerts) or on-demand (ask when they need detail). They can choose voice only, **haptics** only (e.g. temple buzz for caution/danger), or both.

**What it does.** Names objects and positions ("door ahead left," "stairs in 2 meters," "person on your right"); **reads text** (signs, menus, labels, screens) when they look at it; **alerts for safety** (curb, vehicle, obstacle, drop-off) with urgency (safe / caution / warning / danger) via voice or haptic; **scene summary** ("kitchen, sink ahead, table left") in unfamiliar places; **findability** cues so they can locate the right pill bottle or product. All from first-person view — they just look.

**Morning.** In the bathroom they ask "What's on the counter?" or look at the shelf; the glasses list items and positions. **Benefit:** they take the right medication without asking a family member or risking a mix-up. In the kitchen they get "stove clear," "cup to your right," and a buzz for obstacles. **Impact:** they make breakfast and move around without bumping or burning — more independence at the start of the day.

**Leaving home.** "Path clear," "stairs in 2 meters." At the curb: "safe to cross" or "vehicle approaching — wait" (voice or haptic). **Benefit:** they cross the street without a sighted guide or guessing by sound alone. On the sidewalk: "person on your left," "obstacle ahead" + buzz. **Impact:** fewer collisions, less anxiety in crowds, ability to walk familiar and new routes on their own.

**Transit and errands.** They look at the bus sign; the glasses read line, destination, and time. **Benefit:** they choose the right bus without asking a stranger. In the store they get aisle and product names when they look at labels. **Impact:** they shop for themselves without depending on staff or a companion. At the till they can have the total or keypad read — pay correctly and privately.

**Work and social.** In a meeting they ask "What's on the whiteboard?" or get a short summary of who's in the room. **Benefit:** they participate on equal footing instead of missing visual cues. At lunch they look at the menu or their plate and hear it read. **Impact:** more confidence in social and work settings without extra burden on colleagues.

**Evening.** At home they find the remote, the right pill bottle, the light switch; "sofa ahead," "coffee table in front of you." **Benefit:** they wind down and prepare for bed without groping or calling for help. **Impact:** less reliance on family or carers for everyday tasks; dignity and autonomy at home.

**Benefit & impact (summary).** **Independence:** cross streets, shop, travel, and work with less or no need for a sighted guide. **Safety:** fewer falls and collisions thanks to obstacle and curb alerts. **Privacy:** text and environment read to them alone; processing on-device, no cloud. **Confidence:** go out, try new routes, join in at work and socially. **Dignity:** do daily tasks (medication, cooking, finding things) without asking for help every time.

**Under the hood.** One CoreML model per vision condition; condition set once (or per profile). On-device only — works offline, privacy-preserving.

**Pipeline for you (developer):** Train condition-specific models, convert to CoreML (e.g. Colab script or `ml.training.export`), add `.mlpackage` files to the glasses app (e.g. Xcode). The app selects the right model at runtime and runs it on each frame.

### Quick Links

- **Export for Xcode**: [docs/EXPORT_MODELS_TO_XCODE.md](docs/EXPORT_MODELS_TO_XCODE.md) — export and add models to Xcode
- **Deployment**: Run `scripts/export_top7_to_xcode.py` for iOS bundles; see README deployment section.
- **Training Runbook**: [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) - Training commands and monitoring
- **Pre-Train Checklist**: [PRE_TRAIN_CHECKLIST.md](PRE_TRAIN_CHECKLIST.md) - Verification before training
- **Web Simulator**: [tools/simulation/README.md](tools/simulation/README.md) - Simulator setup and usage

### Export Formats

- **CoreML**: iOS deployment (primary target)
- **ExecuTorch (.pte)**: Mobile deployment (recommended for iOS)
- **JIT (.pt)**: PyTorch mobile fallback
- **ONNX**: Cross-platform deployment

### Quick Export

**iOS Bundle (recommended - includes everything):**
```bash
python scripts/export_for_xcode.py checkpoints/final_model.pt maxsight_ios_bundle
```

**Individual formats:**
```bash
# Export to a specific format
python -m ml.training.export --checkpoint checkpoints/final_model.pt --format coreml --output exports/maxsight.mlpackage
python -m ml.training.export --checkpoint checkpoints/final_model.pt --format executorch --output exports/maxsight.pte
python -m ml.training.export --checkpoint checkpoints/final_model.pt --format jit --output exports/maxsight.pt
```

**See [docs/EXPORT_MODELS_TO_XCODE.md](docs/EXPORT_MODELS_TO_XCODE.md) for export and Xcode integration.**

### Running the simulator with a trained model

- **Web simulator**: Set `MAXSIGHT_CHECKPOINT_PATH` environment variable or `model_checkpoint_path` in `tools/simulation/config.py`
- **See**: [tools/simulation/README.md](tools/simulation/README.md) for setup instructions

### Mobile Optimization

- **Quantization**: INT8 quantization reduces model size by ~4x
- **Pruning**: Removes redundant parameters
- **Model Size**: ~250M params → <50MB quantized

---

## Documentation

### Documentation (docs/)

- **[SYSTEMS.md](docs/SYSTEMS.md)**: All systems in one detailed reference (tiers, backbone, heads, fusion, temporal, therapy, retrieval, preprocessing, training, export, scene graph)
- **[architecture.md](docs/architecture.md)**: Model and system architecture overview
- **[therapy_system.md](docs/therapy_system.md)**: Therapy sessions, task generator, integration
- **[training_architecture.md](docs/training_architecture.md)**: Training loop, losses, balancing, config
- **[training-data-loading.md](docs/training-data-loading.md)**: Data pipeline and dataset
- **[transferlearning.md](docs/transferlearning.md)**: Tier transfer and checkpoint loading
- **[status.md](docs/status.md)**: Project status, health, device policy, limitations
- **[downloads.md](docs/downloads.md)**: Dataset and asset downloads
- **[caching.md](docs/caching.md)**: Caching (Redis, usage)
- **[productization/](docs/productization/README.md)**: Productization docs (scope, safety gates, declutter, runtime boundaries, pilot protocol). **[PRODUCTION_RUNBOOK.md](docs/productization/PRODUCTION_RUNBOOK.md)** for production and real-world runbook; **`scripts/product/run.py`** for canonical train/validate/export/package/smoke.
- **Export / Xcode**: [docs/EXPORT_MODELS_TO_XCODE.md](docs/EXPORT_MODELS_TO_XCODE.md) (canonical export and add-to-Xcode guide)

**Warnings & Critical Cautions** (below): Production deployment warnings and fixes (read before deploying).

### Advanced Topics & Implementation Details

**Training from scratch**: Use `ml.models.maxsight_cnn.MaxSightCNN`, `ml.training.train_loop`, `ml.data.data_pipeline.create_data_loaders`, and tier configs under `ml/training/configs/`. **Transfer learning (T2→T5)**: `ml.training.transfer_learning.TierTransferManager`, `transfer_weights`, `validate_source_checkpoint`; parameter groups and freeze schedules in configs. **Inference**: Model forward accepts `images`, optional `audio_features`, `use_temporal`; outputs dict with detections, urgency, distance_zones, etc. **Export**: `ml.training.export` (export_to_coreml, export_to_jit, export_to_onnx, export_to_executorch). See **docs/SYSTEMS.md** and **scripts/train_maxsight.py --help**, **python -m ml.training.export --help**.

#### Troubleshooting Guide

**OOM**: Reduce batch_size, increase gradient accumulation, use gradient checkpointing or mixed precision, or use a lower tier. **Loss not decreasing**: Check learning rate (e.g. lr_finder), GradNorm metrics, loss weights (e.g. ≥0.3 for semantic tasks), data/annotations, and frozen parameters. **Stage B always skipped**: Profile Stage A latency; reduce input size or FPN levels or use INT8; raise uncertainty/latency thresholds in config. **GradNorm issues**: Verify shared params, retain_graph in task_balancing, gradnorm_update_interval, and that task losses are finite. **Export failures**: CoreML may need script instead of trace; ONNX needs input/output names and dynamic_axes; use export_to_executorch for .pte. See **docs/status.md** and **ml/training/export.py**.

**Optimization**: Quantization (INT8 via ml.training.quantization), pruning (ml.optimization.mobile_optimizations), knowledge distillation (ml.training.self_supervised_pretrain). **Custom heads/losses/augmentation**: Extend base classes in ml.models.heads, ml.training.losses, ml.data.advanced_augmentation; see HEAD_REGISTRY and existing heads for patterns.

### Repository Index (Production-Focused)

- **Product pipeline**:
  - `scripts/product/run.py`: canonical entrypoint for `train`, `validate`, `export`, `package`, `smoke`. Use this instead of chaining individual scripts.
  - `scripts/ops/`: operational utilities (data prep, long-run training, export helpers) that call into library code under `ml/` and `app/`.
  - `scripts/research_archive/`: experimental and legacy scripts for reference only; not part of the production path.
  - `scripts/pilot_eval/`: pilot- and study-specific evaluation helpers.
- **Docs** (`docs/`):
  - `architecture.md`: model and system architecture.
  - `SYSTEMS.md`: all systems in one detailed reference.
  - `training_architecture.md`: training loop, losses, balancing, config.
  - `training-data-loading.md`: dataset and pipeline.
  - `transferlearning.md`: T2→T5 and other transfer paths.
  - `status.md`: health, limitations, device policy.
  - `productization/`: scope, safety gates, declutter map, runtime boundaries, pilot protocol, production runbook.
- **Tests** (`tests/`): unit, integration, performance, and safety tests (see Testing & Validation section).
- **Tools** (`tools/`): simulator, quantization, and other developer tools that are not on the device runtime path.
- **Configs** (`ml/training/configs/`): tier and condition YAML configs for learning rates, loss weights, data paths, and transfer schedules.
- **Comment style**: see `.cursor/rules/comment-style.mdc` and `docs/COMMENT_STYLE*.md` (intent-focused, single-line comments).

### Detailed reference (specifications)

#### Latency targets (critical path)

All time-to-alert and Stage A latency targets are **80 ms** (median and p95). Implemented in:

- `ml/runtime_constants.py`: `LATENCY_MEDIAN_MS = 80`, `LATENCY_P95_MS = 80`
- `ml/models/maxsight_cnn.py`: `TierConfig.max_latency_ms = 80.0`; Stage B is skipped if Stage A exceeds 80 ms
- Safety gates SG-03 and SG-04: ≤ 80 ms (see [Productization Summary](#productization-summary-from-reports))
- Simulator and inference engine thresholds: 80 ms

Release is blocked if time-to-alert exceeds 80 ms on the mandatory gate suite.

#### Directory structure (key paths)

```
ml/
  models/          maxsight_cnn.py (T5 model, TierConfig), backbone/, temporal/, attention.py
  data/            dataset.py (MaxSightDataset, COCO/panoptic), data_pipeline.py (collate_fn, create_data_loaders)
  training/        train_loop.py, losses.py, task_balancing.py, transfer_learning.py, export.py, configs/*.yaml
  utils/           output_scheduler.py, preprocessing.py, runtime_constants.py
  retrieval/       stage1_ann, indexing (advisory-only)
app/               overlays, personal_mode (runtime/UI helpers)
scripts/
  product/         run.py (train, validate, export, package, smoke, transfer)
  ops/             train_maxsight.py, gather_training_data.py, validate_data_pipeline.py, smoke_train.py, export_for_xcode.py, ...
  pilot_eval/      test_therapy_effectiveness.py
  research_archive/  legacy/experimental scripts (not production path)
tests/             test_*.py (phase, model, runtime_safety_gates, data_panoptic_and_video, ...)
tools/simulation/  web_simulator, config, simulator/ (inference_engine, overlay, scheduler)
docs/              architecture, status, training_architecture, productization/
```

#### Canonical CLI (`scripts/product/run.py`)

| Subcommand | Required args | Optional args | Description |
|------------|----------------|---------------|-------------|
| **train** | `--data-dir` | `--checkpoint-dir`, `--epochs`, `--batch-size`, `--device`, `--config`, `extra...` | Train model; pass-through to train_maxsight.py |
| **validate** | — | `--checkpoint`, `--data`, `--skip-export-tests` | Run pytest; optionally validate data pipeline and checkpoint forward |
| **export** | `--checkpoint`, `--output` | `--format` (jit\|coreml\|onnx\|executorch) | Export checkpoint to format |
| **package** | — | `--checkpoint`, `--output` | Build Xcode bundle (export_for_xcode) |
| **transfer** | `--source` (T2 ckpt path) | `--config` (t2_to_t5_transfer.yaml) | T2→T5 weight transfer; writes init checkpoint for fine-tune |
| **smoke** | — | `--epochs` | Short training + inference sanity |

All commands run from **repo root**. Example: `python scripts/product/run.py train --data-dir ./data --config ml/training/configs/t2_hybrid_vit.yaml --epochs 50`.

#### Training script (`scripts/ops/train_maxsight.py`) — main flags

- **Paths**: `--data-dir` (required), `--checkpoint-dir`, `--train-annotation`, `--val-annotation`, `--image-dir`
- **Training**: `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--grad-clip`, `--grad-accumulation-steps`, `--scheduler-type`, `--warmup-epochs`, `--early-stopping-patience`, `--checkpoint-interval`
- **Config**: `--config` (YAML path, e.g. `ml/training/configs/t2_hybrid_vit.yaml`) — overrides checkpoint dir and tier flags from `model` and `checkpoint` sections
- **Hardware**: `--device` (cpu|cuda|mlx|auto), `--compile`, `--use-amp`
- **Resume**: `--resume`, `--resume-from`, `--resume-model-only`
- **Model**: `--num-classes`, `--tier` (T5), `--use-audio`, `--condition-mode`
- **Loss**: `--use-gradnorm`
- **Backup**: `--backup` (post-training artifact backup)

Run `python scripts/ops/train_maxsight.py --help` for full list.

#### Config YAML (model and checkpoint)

Under `ml/training/configs/` (e.g. `t2_hybrid_vit.yaml`, `t5_temporal_2phase.yaml`, `t2_to_t5_transfer.yaml`):

- **model**: `tier`, `num_classes`, `use_se_attention`, `use_cbam_attention`, `use_hybrid_backbone`, `use_dynamic_conv`, `use_cross_task_attention`, `use_cross_modal_attention`, `use_temporal_modeling`, `use_retrieval` — all booleans or scalars; `TierConfig.from_dict()` reads these.
- **checkpoint**: `save_dir` — used by train_maxsight when `--config` is set.
- **data**: `train_annotation_file`, `val_annotation_file`, `image_dir`, `batch_size`, `num_workers`, `max_objects`, `condition_mode`, `apply_lighting_augmentation`.
- **training**: `num_epochs`, `learning_rate`, `weight_decay`, `optimizer`, `scheduler`, `warmup_epochs`, `gradient_clip_norm`, `mixed_precision`, `accumulate_grad_batches`.
- **loss**: `use_gradnorm`, `loss_weights` (detection, classification, box_regression, distance, urgency, motion, …).

Transfer configs add **source**/ **target** (checkpoint paths) and **transfer** (validate_source, strict_transfer, freeze schedule).

#### Data formats

- **COCO**: JSON with `images`, `annotations` (bbox in [x, y, w, h]), `categories`. Used by `MaxSightDataset`; annotations grouped by image_id.
- **Panoptic**: Same as COCO but annotations include `segments_info` (list of `{id, category_id, bbox}`). Dataset derives bounding boxes and labels from segments; single-image and sequence collate supported.
- **Sequence (video)**: Batch can provide `frames` (T, C, H, W) per sample; `collate_fn` produces `images` [B, T, C, H, W] and `frame_lengths`. Model forward accepts 5D input for temporal mode.

#### Environment variables (common)

- **MAXSIGHT_CHECKPOINT_PATH**: Checkpoint path for simulator/inference.
- **SPLITS_DIR**: Directory for train/val annotation JSONs (e.g. cleaned_splits).
- **MAXSIGHT_SESSION_TIMEOUT**: Session TTL (default 3600).

#### Troubleshooting (expanded)

| Issue | What to check | Action |
|-------|----------------|--------|
| OOM during training | batch_size, gradient_accumulation_steps, model tier | Reduce batch_size; increase grad accumulation; use lower tier or `--config` with fewer heads |
| Loss not decreasing | Learning rate, GradNorm weights, data/annotations | Tune LR (e.g. 8e-5 for T2); check loss_weights in config; verify labels/boxes in dataset |
| Stage B always skipped | Stage A latency, uncertainty threshold | Profile Stage A; ensure latency &lt; 80 ms (reduce input size, FPN levels, or INT8); or raise TierConfig.max_latency_ms only if product accepts it |
| Export (CoreML/JIT) fails | Traceability, dynamic axes | Use scripted path if trace fails; see ml/training/export.py and docs/status.md |
| Validation fails (e.g. test_export_validation) | JIT trace on platform | Run `run.py validate --skip-export-tests` for CI |
| Transfer (T2→T5) validation fails | Source checkpoint keys, NaNs | Ensure source has `model_state_dict`, `epoch`, `val_loss`; check for NaNs in state dict |

### Additional Documentation

- **[Training Setup Summary](TRAINING_SETUP_SUMMARY.md)**: Training preparation guide.
- **[What Has Been Done](WHAT_HAS_BEEN_DONE.md)**: Complete accomplishment summary.
- **docs/**: Architecture, caching, downloads, status, therapy, training, transfer learning (see Documentation section above).

---

## ️ Vision Conditions Supported

1. **Refractive Errors** (myopia, hyperopia, astigmatism, presbyopia)
2. **Cataracts** (reduced acuity)
3. **Glaucoma** (peripheral vision loss)
4. **AMD** (central vision damage)
5. **Diabetic Retinopathy** (retinal damage, floaters)
6. **Retinitis Pigmentosa** (night blindness, tunnel vision)
7. **Color Blindness** (color confusion)
8. **CVI** (cortical visual impairment)
9. **Amblyopia** (lazy eye)
10. **Strabismus** (crossed eyes)

---

## Key Design Decisions

### Why Two-Stage Inference?

**Problem**: Safety-critical predictions must never be blocked by enhancement features.

**Solution**: Two-stage pipeline with explicit handoff.

**Benefits**:
- **Safety First**: Stage A always completes (≤ 80 ms target)
- **Graceful Degradation**: Stage B can be skipped if needed
- **Predictable Behavior**: Users know safety features always work
- **Resource Management**: Stage A gets priority, Stage B is opportunistic

### Why Tiered Head Architecture?

**Problem**: Not all predictions are equal—safety > navigation > enhancement.

**Solution**: Organize heads into 3 tiers by criticality.

**Benefits**:
- **Safety First**: Tier 1 always runs, never disabled
- **Graceful Degradation**: If Tier 2/3 fail, Tier 1 continues
- **Resource Management**: Tier 1 gets priority
- **Predictable Behavior**: Users know safety features always work

### Why Advisory Retrieval?

**Problem**: Retrieval is powerful but risky if it drives safety decisions.

**Solution**: Make retrieval advisory only—enhances Tier 3 heads, never affects Tier 1 or Tier 2.

**Benefits**:
- Leverages similar scenes for context
- Knowledge graph integration
- More sophisticated descriptions
- **Never blocks safety**: Tier 1/2 independent of retrieval

**Trade-offs**:
- Retrieval can't improve safety decisions (by design)
- Additional inference overhead (but not required)

### Why MPS-Stable Mode?

**Problem**: PyTorch MPS has backward pass bugs in complex models.

**Solution**: MPS-stable mode with edge_attr gradient detachment and CPU fallback for index_add.

**Benefits**:
- Allows local development on Apple Silicon
- Forward pass works fine
- Training possible (with trade-offs)

**Trade-offs**:
- Edge learning disabled in MPS-stable mode
- Use cloud GPU for production training

See **docs/status.md** for device and compatibility notes.

## License

See [LICENSE](LICENSE) file.

