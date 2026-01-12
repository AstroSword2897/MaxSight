# MaxSight System Architecture

**Version:** 1.0.0  
**Last Updated:** December 2025  
**Status:** Production-Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Component Architecture](#component-architecture)
4. [Model Architecture](#model-architecture)
5. [Data Flow](#data-flow)
6. [Integration Points](#integration-points)
7. [Dependencies & Versioning](#dependencies--versioning)
8. [Performance & Latency](#performance--latency)
9. [Recent Improvements](#recent-improvements)

---

## Executive Summary

MaxSight is a **multi-task computer vision system** designed for accessibility, specifically supporting users with vision and hearing disabilities. The system provides:

- **Environmental Reading**: Object detection, OCR, scene descriptions
- **Safety Awareness**: Urgency scoring, hazard detection
- **Navigation Support**: Distance estimation, path planning, spatial memory
- **Multimodal Output**: Visual overlays, voice feedback (TTS), haptic feedback
- **Condition-Specific Adaptations**: 10+ vision condition modes
- **Therapy Integration**: Adaptive therapy exercises and skill development

**Core Technology Stack:**
- **ML Framework**: PyTorch 2.9.1+
- **Model Architecture**: ResNet50 + FPN + 20 specialized heads
- **Deployment**: ExecuTorch (iOS), CoreML, JIT
- **Web Interface**: Flask-based multi-user simulator
- **Data**: COCO (training), Open Images V6, BDD100K, ADE20K (inference)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 📷 Camera    │  │ 🎤 Microphone│  │ 📱 Sensors   │     │
│  │ (Images)     │  │ (Audio)      │  │ (Motion/Hap) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Processing Layer                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🧠 MaxSightCNN (Multi-task Detection)                │  │
│  │    ├── ResNet50 Backbone                             │  │
│  │    ├── FPN (Feature Pyramid Network)                 │  │
│  │    └── 20 Specialized Heads                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 📝 OCR Integration (Vision Framework)                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 💬 Description Generator                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🗺️ Spatial Memory                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🧭 Path Planner                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🏥 Therapy Integration                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Output Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ ⚙️ Output    │  │ 👁️ Visual    │  │ 🔊 Voice     │     │
│  │ Scheduler    │  │ Overlays     │  │ Feedback     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                         │
│  │ 📳 Haptic    │                                         │
│  │ Feedback     │                                         │
│  └──────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Multi-Task Learning**: Single model performs multiple related tasks efficiently
2. **Condition-Specific**: Adapts to 10+ vision conditions (glaucoma, AMD, cataracts, etc.)
3. **Multimodal**: Integrates vision, audio, and haptic feedback
4. **Real-Time**: Optimized for <500ms inference latency
5. **Production-Grade**: Error handling, fallbacks, monitoring, thread safety

---

## Component Architecture

### 1. MaxSightCNN (`ml/models/maxsight_cnn.py`)

**Purpose**: Core multi-task vision model

**Architecture**:
- **Backbone**: ResNet50 (ImageNet pretrained)
- **Neck**: Simplified FPN (Feature Pyramid Network)
- **Heads**: 20 specialized task-specific heads

**Key Features**:
- Anchor-free detection (FCOS-style)
- Multi-scale feature extraction
- Audio-visual fusion
- Condition-specific preprocessing
- Uncertainty estimation

**Input**: `[B, 3, 224, 224]` RGB images + optional audio features  
**Output**: Dictionary with 20+ task outputs (detections, urgency, distance, etc.)

---

### 2. Specialized Heads

#### Core Detection Heads
- **Classification Head**: 48 environmental classes `[B, 196, 48]`
- **Box Regression Head**: Bounding box coordinates `[B, 196, 4]`
- **Objectness Head**: Object confidence scores `[B, 196]`
- **Text Region Head**: Text detection regions `[B, 196]`

#### Accessibility Heads
- **Urgency Head**: 4-level urgency scoring (safe/caution/warning/danger) `[B, 4]`
- **Distance Head**: Depth map + uncertainty + zones `[B, H, W]` + `[B, 3]`
- **Contrast Head**: Contrast sensitivity map `[B, H, W]` (with learned edge attention)
- **Glare Head**: Glare risk assessment `[B, 4]`
- **Findability Head**: Object findability scores `[B, 196]`
- **Navigation Difficulty Head**: Scene navigation difficulty `[B, 1]`
- **Uncertainty Head**: Model uncertainty estimation `[B, 1]`

#### Advanced Heads
- **Scene Description Head**: Scene embedding for TTS `[B, 512]`
- **Sound Event Head**: Environmental sound classification `[B, 15]`
- **Motion Head**: Optical flow estimation `[B, 2, H, W]`
- **ROI Priority Head**: Region-of-interest prioritization `[B, N]`
- **Predictive Alert Head**: Predictive hazard alerts `[B, 1]`
- **Personalization Head**: User-specific adaptations `[B, 256]`
- **Fatigue Head**: User fatigue detection `[B, 1]`

---

### 3. Preprocessing (`ml/utils/preprocessing.py`)

**ImagePreprocessor**: Condition-specific image adaptations

**Supported Conditions**:
- **Glaucoma**: Peripheral vision emphasis
- **AMD**: Central vision emphasis
- **Cataracts**: Contrast enhancement
- **Color Blindness**: Color detection and announcement
- **Retinitis Pigmentosa**: Brightness enhancement
- **Diabetic Retinopathy**: Edge enhancement
- **CVI**: Simplified processing
- **Refractive Errors**: Distance-aware processing
- **Strabismus**: Depth cue emphasis
- **Amblyopia**: Binocular processing

**AudioPreprocessor**: MFCC feature extraction for audio-visual fusion

**DistanceEstimator**: Monocular depth estimation from object sizes

---

### 4. OCR Integration (`ml/utils/ocr_integration.py`)

**Purpose**: Text detection and reading

**Pipeline**:
1. Model detects text regions (`text_regions` head)
2. Vision Framework OCR extracts text
3. Text-to-speech conversion
4. Integration with scene descriptions

**Features**:
- Multi-line text detection
- Document reading mode
- Real-time text reading

---

### 5. Description Generator (`ml/utils/description_generator.py`)

**Purpose**: Natural language scene descriptions

**Input**: Detections, urgency scores, OCR results, user preferences  
**Output**: Verbose scene descriptions for TTS

**Features**:
- Condition-aware verbosity (brief/normal/detailed)
- Urgency-based prioritization
- Spatial relationships ("door 2 meters ahead, handle on left")
- Personal labels integration

---

### 6. Spatial Memory (`ml/utils/spatial_memory.py`)

**Purpose**: Maintain spatial context across frames

**Features**:
- Object tracking across frames
- Spatial relationship mapping
- Temporal consistency
- Memory decay for moving objects

---

### 7. Path Planner (`ml/utils/path_planning.py`)

**Purpose**: Navigation assistance

**Features**:
- Obstacle detection and avoidance
- Walkable path detection
- Turn-by-turn guidance
- Distance-aware routing

---

### 8. Output Scheduler (`ml/utils/output_scheduler.py`)

**Purpose**: Cross-modal output management

**Features**:
- Priority-based queuing
- Urgency-based interruption
- Cooldown management
- Mode-specific filtering (patient/clinician/dev)

**Output Modes**:
- **Patient**: Clean, accessible output (no debug info)
- **Clinician**: Technical details + patient output
- **Dev**: Full debug information + traces

---

### 9. Therapy Integration (`ml/therapy/`)

**Components**:
- **SessionManager**: User session tracking
- **TaskGenerator**: Adaptive therapy task generation
- **TherapyIntegration**: Therapy feedback generation

**Features**:
- Adaptive difficulty
- Progress tracking
- Fatigue detection
- Skill development exercises

---

### 10. Web Simulator (`tools/simulation/web_simulator.py`)

**Purpose**: Multi-user web interface for testing

**Features**:
- Session-based isolation
- Thread-safe processing
- Rate limiting (per-session + global)
- Health monitoring
- Three output modes
- Graceful degradation

---

## Model Architecture

### Detailed Head Architecture

```
Input Image [B, 3, 224, 224]
    ↓
ResNet50 Backbone
    ├── Layer1 → C2 [B, 256, 56, 56]
    ├── Layer2 → C3 [B, 512, 28, 28]
    ├── Layer3 → C4 [B, 1024, 14, 14]
    └── Layer4 → C5 [B, 2048, 7, 7]
    ↓
FPN (Feature Pyramid Network)
    ├── P2 [B, 256, 56, 56]  (fine detail)
    ├── P3 [B, 256, 28, 28]
    ├── P4 [B, 256, 14, 14]
    └── P5 [B, 256, 7, 7]    (coarse detail)
    ↓
Scene Context Extraction
    └── Scene Embedding [B, 256]
    ↓
Detection Features
    └── Det Features [B, 256, H, W]
    ↓
┌─────────────────────────────────────────┐
│  Core Heads (Parallel Execution)        │
│  ├── Classification [B, 196, 48]       │
│  ├── Box Regression [B, 196, 4]        │
│  ├── Objectness [B, 196]                │
│  └── Text Regions [B, 196]              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Accessibility Heads (Parallel)         │
│  ├── Urgency [B, 4]                     │
│  ├── Contrast [B, H, W] (edge-aware)   │
│  ├── Glare [B, 4]                       │
│  ├── Navigation Difficulty [B, 1]       │
│  └── Uncertainty [B, 1]                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Dependent Heads                        │
│  ├── Distance [B, H, W] + [B, 3]       │
│  │   └── Uses: boxes + scene_embedding │
│  └── Findability [B, 196]               │
│      └── Uses: det_features             │
└─────────────────────────────────────────┘
```

### Head Execution Order

1. **Backbone** → Multi-scale features (C2, C3, C4, C5)
2. **FPN** → FPN features (P2, P3, P4, P5)
3. **Scene Context** → Scene embedding
4. **Detection Features** → Detection features
5. **Shared Scene Embedding** → Shared embedding
6. **Core Heads** (parallel): Classification, Box, Objectness, Text
7. **Scene-Level Heads** (parallel): Urgency, Contrast, Glare, Navigation, Uncertainty
8. **Dependent Heads**: Distance (uses boxes), Findability (uses det_feats)

---

## Data Flow

### Training Pipeline

```
COCO Dataset (350K train, 70K val)
    ↓
Annotation Generation (generate_annotations.py)
    ├── Map COCO → Environmental categories
    ├── Assign urgency scores
    ├── Estimate distance zones
    └── Generate scene descriptions
    ↓
MaxSightDataset (dataset.py)
    ├── Load images + annotations
    ├── Apply condition-specific preprocessing
    ├── Apply augmentations
    └── Format targets
    ↓
DataLoader (with augmentations)
    ↓
MaxSightCNN Forward Pass
    ├── Backbone extraction
    ├── FPN features
    └── All 20 heads
    ↓
Multi-Task Loss Calculation (losses.py)
    ├── DetectionLoss (Focal + IoU)
    ├── DepthLoss (uncertainty-weighted)
    ├── ContrastLoss (edge-aware)
    └── Other head losses
    ↓
Backpropagation
    ├── Gradient accumulation
    ├── Mixed precision (AMP)
    └── EMA (Exponential Moving Average)
    ↓
Model Checkpoint
```

### Inference Pipeline

```
Input Image/Audio
    ↓
Preprocessing (ImagePreprocessor)
    ├── Condition-specific adaptations
    ├── Resize to 224x224
    └── ImageNet normalization
    ↓
MaxSightCNN Inference
    ├── Forward pass (all heads)
    └── Output: 20+ task outputs
    ↓
Post-Processing
    ├── NMS (Non-Maximum Suppression)
    ├── Confidence filtering (>0.5)
    └── Detection formatting
    ↓
OCR Integration (if text detected)
    ├── Extract text regions
    ├── Vision Framework OCR
    └── Text extraction
    ↓
Description Generation
    ├── Combine detections + OCR + urgency
    ├── Generate natural language
    └── Condition-aware verbosity
    ↓
Spatial Memory Update
    ├── Track objects across frames
    └── Update spatial relationships
    ↓
Path Planning
    ├── Detect obstacles
    ├── Find walkable paths
    └── Generate navigation cues
    ↓
Output Scheduler
    ├── Priority filtering
    ├── Urgency-based queuing
    └── Cooldown management
    ↓
Multi-Modal Output
    ├── Visual Overlays (bounding boxes, labels)
    ├── Voice Feedback (TTS)
    └── Haptic Feedback (patterns)
```

### Web Simulator Flow

```
HTTP Request (POST /api/process)
    ├── Image upload (base64)
    ├── Condition mode
    ├── Output mode (patient/clinician/dev)
    └── Session ID
    ↓
Session Validation
    ├── Check session exists
    ├── Validate permissions
    └── Rate limiting check
    ↓
Image Decoding
    ├── Base64 → PIL Image
    ├── Format validation
    └── Size validation
    ↓
MaxSightCore.process_frame()
    ├── Preprocessing
    ├── Model inference
    ├── Post-processing
    ├── OCR integration
    ├── Description generation
    └── Output scheduling
    ↓
Response Shaping (by output_mode)
    ├── Patient: Clean output (no debug)
    ├── Clinician: Technical + patient
    └── Dev: Full debug + traces
    ↓
JSON Response
    ├── Detections
    ├── Scene description
    ├── Urgency scores
    ├── Navigation cues
    └── (Debug info if dev mode)
```

---

## Integration Points

### 1. Model → OCR

**Connection**: `text_regions` head output → Vision Framework OCR

**Flow**:
```python
text_regions = model_outputs['text_regions']  # [B, 196]
boxes = model_outputs['boxes']  # [B, 196, 4]
ocr_results = ocr_manager.process_image_for_ocr(image, text_regions, boxes)
```

---

### 2. Model → Description Generator

**Connection**: Detections + urgency + OCR → Natural language

**Flow**:
```python
description = description_generator.generate_scene_description(
    detections=detections,
    urgency_scores=model_outputs['urgency_scores'],
    text_results=ocr_results,
    verbosity=user_preferences['verbosity']
)
```

---

### 3. Model → Spatial Memory

**Connection**: Detections → Spatial tracking

**Flow**:
```python
spatial_memory.update(detections)
spatial_context = spatial_memory.get_context()
```

---

### 4. Spatial Memory → Path Planner

**Connection**: Spatial context → Navigation planning

**Flow**:
```python
path_info = path_planner.plan_path(spatial_context)
```

---

### 5. All → Output Scheduler

**Connection**: All outputs → Prioritized multimodal output

**Flow**:
```python
scheduled_outputs = output_scheduler.schedule(
    detections=detections,
    urgency_scores=urgency,
    scene_description=description,
    path_info=path_info,
    uncertainty=uncertainty
)
```

---

## Dependencies & Versioning

### Component Versions

| Component | Version | Status |
|-----------|---------|--------|
| MaxSightCNN | 1.0.0 | ✅ Production |
| Backbone (ResNet50) | 1.0.0 | ✅ Stable |
| FPN | 1.0.0 | ✅ Stable |
| All Heads | 1.0.0 | ✅ Stable |
| Preprocessing | 1.0.0 | ✅ Stable |
| OCR Integration | 1.0.0 | ✅ Stable |
| Description Generator | 1.0.0 | ✅ Stable |
| Spatial Memory | 1.0.0 | ✅ Stable |
| Path Planner | 1.0.0 | ✅ Stable |
| Output Scheduler | 1.0.0 | ✅ Stable |
| Therapy Integration | 1.0.0 | ✅ Stable |

### Dependency Graph

```
model (v1.0.0)
    └── No dependencies
        ↓ outputs: classifications, boxes, urgency_scores, etc.

preprocessing (v1.0.0)
    └── No dependencies
        ↓ outputs: preprocessed_image

ocr (v1.0.0)
    └── Depends on: model.text_regions
        ↓ outputs: text_results

description_generator (v1.0.0)
    └── Depends on: model.detections, model.urgency_scores, ocr.text_results
        ↓ outputs: scene_description

spatial_memory (v1.0.0)
    └── Depends on: model.detections
        ↓ outputs: spatial_context

path_planner (v1.0.0)
    └── Depends on: spatial_memory.spatial_context
        ↓ outputs: path_info

output_scheduler (v1.0.0)
    └── Depends on: model.detections, model.urgency_scores, model.uncertainty
        ↓ outputs: scheduled_outputs

therapy_integration (v1.0.0)
    └── Depends on: model.detections, session_manager
        ↓ outputs: therapy_feedback
```

### Python Dependencies

**Core**:
- PyTorch >= 2.9.1
- torchvision >= 0.24.1
- torchaudio >= 2.9.1

**ML Utilities**:
- scipy >= 1.11.0 (Hungarian matching)
- scikit-learn >= 1.3.0 (clustering)
- numpy >= 1.24.0

**Optional**:
- sentence-transformers >= 2.2.2 (OCR text embeddings)
- torch-geometric (scene graph, optional)

---

## Performance & Latency

### Latency Targets

| Component | Target | Acceptable | Critical |
|-----------|--------|------------|----------|
| Core Heads Only | <200ms | <250ms | ✅ Yes |
| Core + Text | <250ms | <300ms | ✅ Yes |
| Core + Urgency | <300ms | <350ms | ⚠️ Optional |
| Core + Distance | <350ms | <400ms | ⚠️ Optional |
| All Heads | <500ms | <600ms | ⚠️ Optional |
| Full Pipeline | <1000ms | <1500ms | ⚠️ Optional |

### Head Latency Breakdown (Estimated)

| Head | Latency | Critical | Notes |
|------|---------|----------|-------|
| Classification | 50ms | ✅ | Core detection |
| Box Regression | 40ms | ✅ | Core detection |
| Objectness | 30ms | ✅ | Core detection |
| Text Region | 20ms | ⚠️ | Optional |
| Urgency | 15ms | ⚠️ | Safety feature |
| Distance | 25ms | ⚠️ | Navigation |
| Contrast | 10ms | ⚠️ | Accessibility |
| Glare | 10ms | ⚠️ | Accessibility |
| Findability | 20ms | ⚠️ | Accessibility |
| Navigation Difficulty | 10ms | ⚠️ | Navigation |
| Uncertainty | 10ms | ⚠️ | Quality metric |
| Scene Description | 30ms | ⚠️ | TTS input |
| Sound Event | 25ms | ⚠️ | Audio fusion |
| Motion | 50ms | ⚠️ | Temporal |
| ROI Priority | 15ms | ⚠️ | Prioritization |
| Predictive Alert | 10ms | ⚠️ | Safety |
| Personalization | 20ms | ⚠️ | User adaptation |
| Fatigue | 15ms | ⚠️ | Therapy |

**Total (all heads)**: ~400ms estimated (hardware-dependent)

### Optimization Strategies

1. **Selective Head Execution**: Enable only required heads
2. **Quantization**: INT8 quantization (4x compression, <1% accuracy loss)
3. **Model Pruning**: Remove unused heads for deployment
4. **Caching**: Cache shared embeddings to avoid recomputation
5. **Batch Processing**: Process multiple frames together when possible

---

## Recent Improvements

### Architecture Fixes (December 2025)

#### 1. Contrast Head - Edge Injection ✅
- **Before**: Edges computed only in loss (inefficient)
- **After**: Learned edge attention modulates features during forward pass
- **Impact**: More efficient training, better edge-aware predictions

#### 2. Depth Head - Uncertainty Weighting ✅
- **Before**: Uncertainty predicted but not used in loss
- **After**: Unified Kendall & Gal formulation: `L = |d - d_gt| * exp(-u) + u`
- **Impact**: Proper uncertainty calibration, better depth quality

#### 3. Depth Head - Zone Grounding ✅
- **Before**: Zones used only mean depth (scalar)
- **After**: Zones use depth percentiles (p25, p50, p75) for distributional awareness
- **Impact**: Zones consistent with depth map, better navigation

#### 4. Depth Head - Robust Normalization ✅
- **Before**: BatchNorm (fragile to small batches)
- **After**: GroupNorm (robust to variable batch sizes)
- **Impact**: Better deployment stability

#### 5. Loss Functions - Vectorization ✅
- **Before**: O(N²) Python loops in ROI loss
- **After**: Fully vectorized tensor operations
- **Impact**: Scales to large ROI counts

#### 6. Loss Functions - Dtype Safety ✅
- **Before**: `torch.tensor(0.0)` (dtype mismatches)
- **After**: `torch.zeros((), dtype=dtype)` (consistent)
- **Impact**: AMP compatibility, no dtype errors

#### 7. Motion Loss - Robustness ✅
- **Before**: Naive L1 smoothness loss
- **After**: Charbonnier loss + edge-weighted smoothness
- **Impact**: More robust to outliers, better optical flow

---

## System Capabilities

### Supported Vision Conditions

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

### Core Features

- ✅ **Environmental Reading**: Object detection (48 classes)
- ✅ **Text Reading**: OCR integration with TTS
- ✅ **Safety Awareness**: Urgency scoring (4 levels)
- ✅ **Navigation**: Distance estimation + path planning
- ✅ **Multimodal Output**: Visual + audio + haptic
- ✅ **Personalization**: Custom labels, verbosity adjustment
- ✅ **Therapy Integration**: Adaptive exercises
- ✅ **Real-Time Processing**: <500ms inference target

---

## Deployment Architecture

### iOS Deployment (Target)

```
MaxSight iOS App
    ├── ExecuTorch Runtime
    │   └── maxsight_traced.pt (model file)
    ├── Camera Manager (AVFoundation)
    ├── Audio Manager (AVAudioEngine)
    ├── Model Inference (ExecuTorch)
    ├── OCR (Vision Framework)
    ├── TTS (AVSpeechSynthesizer)
    └── Haptics (Core Haptics)
```

### Web Simulator (Current)

```
Flask Web Server
    ├── Multi-user session management
    ├── Thread-safe processing
    ├── Rate limiting
    ├── Health monitoring
    └── REST API endpoints
```

---

## Future Enhancements

### Planned (Sprint 2-4)

1. **iOS App**: Native iOS implementation
2. **Real-Time Video**: Continuous frame processing
3. **Advanced Sound**: Directional audio, sound prioritization
4. **Enhanced Navigation**: Turn-by-turn guidance
5. **User Customization**: Preference persistence
6. **Performance Optimization**: Battery management, thermal throttling

---

## References

- **Model Architecture**: `ml/models/maxsight_cnn.py`
- **Dependency Graph**: `docs/DEPENDENCY_GRAPH.md`
- **Training Pipeline**: `ml/training/train_loop.py`
- **Loss Functions**: `ml/training/losses.py`, `ml/training/head_losses.py`
- **Web Simulator**: `tools/simulation/web_simulator.py`
- **Recent Fixes**: `docs/analysis/FIXES_APPLIED.md`

---

**Last Updated**: December 2025  
**Maintainer**: Engineering Team  
**Status**: Production-Ready v1.0.0

