# MaxSight - Removing Barriers for Vision & Hearing Disabilities

**Visual Focus First** | **60-Day Implementation Plan**

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement & Solution](#-problem-statement--solution)
3. [System Architecture](#-system-architecture)
4. [Core Components Deep Dive](#-core-components-deep-dive)
5. [Data Flow & Processing Pipeline](#-data-flow--processing-pipeline)
6. [Model Architecture Explained](#-model-architecture-explained)
7. [Training Infrastructure](#-training-infrastructure)
8. [Production Features & Safety](#-production-features--safety)
9. [Repository Structure](#-repository-structure)
10. [Quick Start Guide](#-quick-start-guide)
11. [Key Design Decisions & Rationale](#-key-design-decisions--rationale)

---

## 🎯 Project Overview

MaxSight is a **production-grade accessibility application** that helps users with vision and hearing disabilities navigate and understand their environment through advanced computer vision and multimodal feedback. Unlike simple object detectors, MaxSight provides rich, structured environmental information tailored to each user's specific vision condition.

### What Makes MaxSight Different

**Standard object detectors** answer: "What is this?" and "Where is it?"

**MaxSight** answers:
- **WHAT**: Object class (door, stairs, vehicle, person)
- **WHERE**: Precise bounding box position (for directional cues)
- **HOW FAR**: Distance zones (near/medium/far for navigation)
- **HOW URGENT**: Urgency level (safe/caution/warning/danger for safety)
- **HOW FINDABLE**: Object findability scores (for users with low vision)
- **SCENE CONTEXT**: Natural language scene descriptions
- **ACCESSIBILITY METRICS**: Contrast sensitivity, glare risk, navigation difficulty
- **TEMPORAL AWARENESS**: Motion tracking, predictive alerts
- **PERSONALIZATION**: User-specific adaptations and preferences

### Core Capabilities

MaxSight implements four key barrier-removal methods from accessibility research:

1. **Environmental Structuring**: Labels surroundings in ways users can understand
2. **Clear Multimodal Communication**: Visual, audio, and haptic feedback
3. **Skill Development Across Senses**: Addresses different senses for information input
4. **Routine Workflow**: Adapts tasks to usage patterns and needs

---

## 🎯 Problem Statement & Solution

### The Problem

People with vision and hearing disabilities face barriers when interacting with their environment:
- **Visual barriers**: Cannot see objects, obstacles, text, or spatial relationships
- **Safety barriers**: Cannot assess urgency or danger in their environment
- **Navigation barriers**: Cannot estimate distances or plan paths
- **Information barriers**: Cannot access text, signs, or environmental cues

### The Solution

MaxSight uses **multi-task deep learning** to provide the same rich environmental information that sighted people process automatically:

1. **Multi-Task Learning**: Single model performs 20+ related tasks efficiently
2. **Condition-Specific Adaptations**: Adapts to 10+ vision conditions (glaucoma, AMD, cataracts, etc.)
3. **Multimodal Integration**: Combines vision, audio, and haptic feedback
4. **Real-Time Processing**: Optimized for <500ms inference latency on mobile devices
5. **Production-Grade Safety**: Error handling, fallbacks, uncertainty suppression, kill switches

### Why Multi-Task Architecture?

Traditional single-task models (e.g., object detection only) don't provide enough information for accessibility. Users need:
- **Spatial awareness**: Where objects are and how far
- **Safety assessment**: Whether objects are dangerous
- **Context understanding**: What the overall scene means
- **Accessibility metrics**: Contrast, glare, navigation difficulty

A multi-task architecture shares feature extraction across all tasks, making it efficient while providing comprehensive information.

---

## 🏗️ System Architecture

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
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🔍 Retrieval System (Multi-Vector)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Output Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ ⚙️ Output    │  │ 👁️ Visual    │  │ 🔊 Voice     │  │
│  │ Scheduler    │  │ Overlays     │  │ Feedback     │  │
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
6. **Safety-First**: Uncertainty suppression, ethical safeguards, graceful degradation

---

## 🔍 Core Components Deep Dive

### 1. MaxSightCNN (`ml/models/maxsight_cnn.py`)

**Purpose**: Core multi-task vision model that powers all environmental understanding.

**Architecture**:
- **Backbone**: ResNet50 (ImageNet pretrained) - extracts visual features
- **Neck**: Simplified FPN (Feature Pyramid Network) - multi-scale feature extraction
- **Heads**: 20 specialized task-specific heads - each predicts different information

**Why ResNet50 + FPN?**
- ResNet50 provides strong feature extraction with proven performance
- FPN enables detection of objects at multiple scales (small and large objects)
- This combination balances accuracy and efficiency for mobile deployment

**Input**: `[B, 3, 224, 224]` RGB images + optional audio features  
**Output**: Dictionary with 20+ task outputs (detections, urgency, distance, etc.)

**Key Features**:
- Anchor-free detection (FCOS-style) - simpler and more efficient than anchor-based
- Multi-scale feature extraction - detects objects of all sizes
- Audio-visual fusion - combines vision and sound for better understanding
- Condition-specific preprocessing - adapts to user's vision condition
- Uncertainty estimation - knows when it's uncertain

---

### 2. Specialized Heads (20+ Heads)

Each head is a specialized neural network that predicts specific information. All heads share the same backbone features but predict different outputs.

#### Core Detection Heads

| Head | Purpose | Output Shape | Why It Exists |
|------|---------|--------------|---------------|
| **Classification** | What object is this? | `[B, 196, 48]` | Identifies 48 environmental classes (doors, stairs, vehicles, etc.) |
| **Box Regression** | Where is the object? | `[B, 196, 4]` | Provides bounding box coordinates for spatial awareness |
| **Objectness** | Is there an object here? | `[B, 196]` | Confidence score to filter out background noise |
| **Text Region** | Where is text? | `[B, 196]` | Identifies text regions for OCR processing |

#### Accessibility Heads

| Head | Purpose | Output Shape | Why It Exists |
|------|---------|--------------|---------------|
| **Urgency** | How urgent/dangerous? | `[B, 4]` | 4-level urgency (safe/caution/warning/danger) for safety |
| **Distance** | How far away? | `[B, H, W]` + `[B, 3]` | Depth map + distance zones (near/medium/far) for navigation |
| **Contrast** | Contrast sensitivity map | `[B, H, W]` | Identifies low-contrast regions that are hard to see |
| **Glare** | Glare risk assessment | `[B, 4]` | Predicts glare risk to warn users |
| **Findability** | How easy to find? | `[B, 196]` | Scores how easy objects are to locate (for low vision) |
| **Navigation Difficulty** | Scene navigation difficulty | `[B, 1]` | Overall scene complexity score |
| **Uncertainty** | Model confidence | `[B, 1]` | Estimates when model is uncertain (critical for safety) |

#### Advanced Heads

| Head | Purpose | Output Shape | Why It Exists |
|------|---------|--------------|---------------|
| **Scene Description** | Scene embedding for TTS | `[B, 512]` | Encodes scene context for natural language descriptions |
| **Sound Event** | Environmental sound classification | `[B, 15]` | Classifies sounds (alarms, sirens, vehicles, speech) |
| **Motion** | Optical flow estimation | `[B, 2, H, W]` | Tracks object motion for predictive alerts |
| **ROI Priority** | Region-of-interest prioritization | `[B, N]` | Identifies most important regions to focus on |
| **Predictive Alert** | Predictive hazard alerts | `[B, 1]` | Predicts potential hazards before they occur |
| **Personalization** | User-specific adaptations | `[B, 256]` | Learns user preferences and adapts outputs |
| **Fatigue** | User fatigue detection | `[B, 1]` | Detects when user is fatigued (for therapy) |

**Why So Many Heads?**

Each head addresses a specific accessibility need:
- **Detection heads** provide basic object recognition
- **Accessibility heads** address specific vision condition needs
- **Advanced heads** enable sophisticated features (navigation, personalization, therapy)

All heads share the same backbone features, making this efficient. The alternative (separate models) would be much slower and use more memory.

---

### 3. Preprocessing (`ml/utils/preprocessing.py`)

**Purpose**: Condition-specific image adaptations that modify input images based on user's vision condition.

**Why Condition-Specific Preprocessing?**

Different vision conditions require different processing:
- **Glaucoma** (peripheral vision loss): Emphasizes peripheral regions
- **AMD** (central vision damage): Emphasizes central regions
- **Cataracts** (blur): Contrast enhancement
- **Color Blindness**: Color detection and explicit color announcements
- **Retinitis Pigmentosa** (night blindness): Brightness enhancement
- **Diabetic Retinopathy**: Edge enhancement
- **CVI** (cortical visual impairment): Simplified processing

**How It Works**:
1. User selects their vision condition
2. Preprocessor applies condition-specific transformations
3. Model processes adapted image
4. Outputs are tailored to condition

This ensures the model provides useful information regardless of the user's specific vision condition.

---

### 4. OCR Integration (`ml/utils/ocr_integration.py`)

**Purpose**: Text detection and reading for accessibility.

**Pipeline**:
1. Model detects text regions (`text_regions` head)
2. iOS Vision Framework OCR extracts text
3. Text-to-speech conversion
4. Integration with scene descriptions

**Why Separate OCR Integration?**

The model detects *where* text is, but iOS Vision Framework is better at *reading* text. This hybrid approach:
- Uses model for fast text region detection
- Uses Vision Framework for accurate text recognition
- Combines both for comprehensive text accessibility

---

### 5. Description Generator (`ml/utils/description_generator.py`)

**Purpose**: Generates natural language scene descriptions for text-to-speech.

**Input**: Detections, urgency scores, OCR results, user preferences  
**Output**: Verbose scene descriptions (e.g., "Door 2 meters ahead, handle on left")

**Why Natural Language Descriptions?**

Users need descriptions that are:
- **Spatial**: Where objects are relative to the user
- **Actionable**: What the user should do
- **Contextual**: How objects relate to each other
- **Condition-Aware**: Tailored to user's vision condition

This directly supports "Clear Multimodal Communication" by providing structured, understandable descriptions.

---

### 6. Spatial Memory (`ml/utils/spatial_memory.py`)

**Purpose**: Maintains a memory of objects and their positions over time.

**Why Spatial Memory?**

Users need to:
- Remember where objects were
- Track objects as they move
- Build a mental map of their environment
- Navigate based on remembered locations

Spatial memory enables these capabilities by maintaining a persistent object map.

---

### 7. Path Planner (`ml/utils/path_planner.py`)

**Purpose**: Plans navigation paths based on detected objects and spatial memory.

**Why Path Planning?**

Users need:
- Safe paths around obstacles
- Efficient routes to destinations
- Real-time navigation assistance
- Obstacle avoidance

Path planner combines detection, spatial memory, and navigation logic to provide these capabilities.

---

### 8. Therapy Integration (`ml/therapy/`)

**Purpose**: Adaptive therapy exercises and skill development.

**Components**:
- **SessionManager**: Tracks user sessions and progress
- **TaskGenerator**: Generates adaptive therapy tasks
- **TherapyIntegration**: Integrates therapy feedback into outputs

**Why Therapy Integration?**

MaxSight supports both:
1. **Immediate assistance**: Real-time environmental awareness
2. **Long-term skill development**: Vision therapy exercises

Therapy integration enables:
- Adaptive difficulty based on performance
- Progress tracking over time
- Skill-specific exercises (attention, contrast, edge recognition)
- Gradual independence (reducing reliance on app)

This directly supports "Skill Development Across Senses" by providing exercises that develop skills directly applicable to real-world use.

---

### 9. Retrieval System (`ml/retrieval/`)

**Purpose**: Multi-vector retrieval for knowledge-augmented scene understanding.

**Why Retrieval System?**

The retrieval system enables:
- **Similar scene matching**: Find similar scenes from training data
- **Knowledge augmentation**: Enhance understanding with retrieved context
- **Concept retrieval**: Retrieve relevant concepts and relationships
- **Cross-view learning**: Learn from multiple viewpoints

**Components**:
- **Encoders**: Extract embeddings from images, audio, text, depth, scene graphs
- **Indexing**: FAISS-based indexing for fast similarity search
- **Two-Stage Retrieval**: Fast ANN search + multi-vector reranking
- **Knowledge Augmentation**: GNN-based knowledge graph integration

**How It Works**:
1. Extract multiple embeddings (global, region, patch, depth, OCR, audio, scene graph)
2. Build FAISS index for fast similarity search
3. Stage 1: Fast ANN search for candidate retrieval (<20ms)
4. Stage 2: Multi-vector reranking for final results
5. Knowledge augmentation: Enhance with scene graph knowledge

This enables more sophisticated scene understanding by leveraging similar scenes and knowledge graphs.

---

### 10. Output Scheduler (`ml/utils/output_scheduler.py`)

**Purpose**: Cross-modal output management (audio/visual/haptic).

**Why Output Scheduler?**

Users receive information through multiple channels:
- **Visual**: Overlays, highlighting, bounding boxes
- **Audio**: Text-to-speech, sound alerts
- **Haptic**: Directional vibration patterns

The scheduler:
- **Prioritizes**: Urgent information first
- **Rate limits**: Prevents information overload
- **Coordinates**: Ensures channels don't conflict
- **Spatializes**: Provides directional cues (left/right/front/back)

This ensures users receive information in a clear, non-overwhelming way.

---

## 🔄 Data Flow & Processing Pipeline

### Complete Processing Flow

```
1. INPUT ACQUISITION
   ├── Camera captures image [B, 3, 224, 224]
   ├── Microphone captures audio [B, T, F]
   └── Sensors capture motion/haptic data

2. PREPROCESSING
   ├── Condition-specific image adaptation
   ├── Audio feature extraction (MFCC)
   └── Sensor data normalization

3. MODEL INFERENCE
   ├── ResNet50 backbone extracts features
   ├── FPN creates multi-scale features
   ├── 20 heads predict outputs in parallel
   └── Outputs combined into dictionary

4. POST-PROCESSING
   ├── OCR on text regions
   ├── Description generation
   ├── Spatial memory update
   ├── Path planning
   └── Therapy integration

5. OUTPUT SCHEDULING
   ├── Priority filtering
   ├── Rate limiting
   ├── Channel selection (visual/audio/haptic)
   └── Spatial positioning

6. MULTIMODAL OUTPUT
   ├── Visual overlays (max 10% screen)
   ├── Voice feedback (TTS)
   └── Haptic feedback (directional vibration)
```

### Example: Detecting a Door

1. **Input**: Camera image of a door
2. **Preprocessing**: Condition-specific adaptation (e.g., contrast enhancement for cataracts)
3. **Model Inference**:
   - Classification head: "door" (confidence: 0.95)
   - Box regression head: [x=0.5, y=0.3, w=0.2, h=0.4]
   - Distance head: "near" zone (2.5 meters)
   - Urgency head: "safe" (0.1)
   - Findability head: 0.8 (easy to find)
4. **Post-Processing**:
   - Description: "Door 2.5 meters ahead, centered, handle on right"
   - Spatial memory: Add door to object map
   - Path planning: Door is accessible
5. **Output Scheduling**:
   - Priority: Medium (not urgent)
   - Channel: Audio (voice description)
   - Rate limit: Allow (not too frequent)
6. **Output**: TTS: "Door 2.5 meters ahead, centered, handle on right"

---

## 🧠 Model Architecture Explained

### Backbone: ResNet50

**Why ResNet50?**
- Proven performance on ImageNet
- Efficient for mobile deployment
- Good balance of accuracy and speed
- Pretrained weights available

**Architecture**:
```
Input [B, 3, 224, 224]
  ↓
Conv1: [B, 64, 112, 112]
  ↓
Layer1: [B, 256, 56, 56]   (C2)
  ↓
Layer2: [B, 512, 28, 28]   (C3)
  ↓
Layer3: [B, 1024, 14, 14]  (C4)
  ↓
Layer4: [B, 2048, 7, 7]    (C5)
```

### Neck: Feature Pyramid Network (FPN)

**Why FPN?**
- Multi-scale feature extraction
- Detects objects of all sizes (small and large)
- Top-down pathway combines high-level and low-level features

**Architecture**:
```
C2 [B, 256, 56, 56] ──┐
C3 [B, 512, 28, 28] ──┤
C4 [B, 1024, 14, 14] ─┤
C5 [B, 2048, 7, 7] ───┘
         ↓
    FPN Processing
         ↓
P2 [B, 256, 56, 56]  (fine detail - small objects)
P3 [B, 256, 28, 28]
P4 [B, 256, 14, 14]
P5 [B, 256, 7, 7]    (coarse detail - large objects)
```

### Heads: Task-Specific Predictions

All heads share the same FPN features but predict different outputs:

```
Shared FPN Features [B, 256, H, W]
         ↓
    ┌────┴────┬────────┬─────────┐
    ↓         ↓         ↓         ↓
Classification  Box      Objectness  Text
   Head      Regression   Head      Head
    ↓         ↓         ↓         ↓
  [B,196,48] [B,196,4] [B,196]  [B,196]
```

**Why Shared Features?**

- **Efficiency**: Extract features once, use for all tasks
- **Consistency**: All tasks see the same visual information
- **Transfer Learning**: Features learned for one task help others

---

## 🎓 Training Infrastructure

### Training Pipeline

```
1. DATA LOADING
   ├── MaxSightDataset loads COCO + accessibility data
   ├── Condition-specific augmentations
   └── Multi-modal data (images + audio)

2. FORWARD PASS
   ├── Model processes batch
   ├── All 20 heads predict outputs
   └── Losses computed for each head

3. LOSS COMPUTATION
   ├── Detection loss (classification + box regression)
   ├── Accessibility losses (urgency, distance, contrast, etc.)
   ├── Advanced losses (motion, personalization, fatigue)
   └── GradNorm balancing (prevents gradient warfare)

4. BACKWARD PASS
   ├── Gradients computed for all heads
   ├── GradNorm adjusts head weights
   └── Optimizer updates parameters

5. EVALUATION
   ├── Metrics computed (mAP, precision, recall)
   ├── Head-specific metrics
   └── Stress tests (gradient warfare detection)
```

### Task Balancing (`ml/training/task_balancing.py`)

**Problem**: With 20+ heads, some heads may dominate training while others starve (gradient warfare).

**Solution**: GradNorm adaptive loss balancing

**How GradNorm Works**:
1. Monitor loss ratios between heads
2. Adjust head weights to balance learning rates
3. Prevent dominant heads from starving others
4. Auto-dampen problematic heads

**Why This Matters**:
- Without balancing: Detection head dominates, other heads fail
- With balancing: All heads learn together, system provides comprehensive information

### Stress Testing (`ml/training/stress_tests.py`)

**Purpose**: Validate system stability and detect issues before deployment.

**Tests**:
- **Head Isolation**: Detect gradient interference between heads
- **Loss Scaling**: Ensure no loss term dominates training
- **Input Corruption**: Validate robustness to real-world conditions
- **Temporal Stability**: Check frame-to-frame consistency
- **Head Dropout**: Verify graceful degradation

**Why Stress Testing?**

Accessibility systems must be reliable. Stress tests ensure:
- System doesn't fail catastrophically
- Heads don't interfere with each other
- Model is robust to real-world conditions
- Graceful degradation when components fail

---

## 🛡️ Production Features & Safety

### 1. Task Balancing & Gradient Warfare Prevention

**Problem**: Multi-task learning with 20+ heads can cause gradient conflicts.

**Solutions**:
- **GradNorm**: Adaptive loss balancing across all heads
- **PCGrad**: Projected conflicting gradients for multi-task learning
- **Stress Monitoring**: Real-time detection of dominant, oscillating, or plateaued heads
- **Auto-Dampening**: Automatic weight reduction for problematic heads

**Why This Matters**: Prevents silent head collapse and ensures all heads learn effectively.

---

### 2. Runtime Safety & Reliability

**Head Kill Switches** (`ml/utils/error_handling.py`):
- Runtime enable/disable of heads for performance, battery, or debugging
- Allows graceful degradation when heads fail
- Enables performance optimization by disabling non-critical heads

**Ethical Safeguards**:
- **Uncertainty Suppression**: High uncertainty (>0.7) suppresses all actions
- **Safety Checks**: Validates outputs before presentation
- **Graceful Degradation**: System continues operating when components fail

**Schema Validation** (`ml/utils/schema_validator.py`):
- Output validation with automatic downgrade on failure
- Ensures outputs match expected format
- Prevents malformed outputs from reaching users

**Why Safety Features?**

Accessibility systems must:
- Never provide false confidence
- Degrade gracefully, not fail catastrophically
- Suppress uncertain outputs
- Continue operating when components fail

---

### 3. Stress Testing Infrastructure

**Head Isolation Tests**: Detect gradient interference between heads  
**Loss Scaling Tests**: Ensure no loss term dominates training  
**Input Corruption Tests**: Validate robustness to real-world conditions  
**Temporal Stability Tests**: Check frame-to-frame consistency  
**Head Dropout Tests**: Verify graceful degradation

**Why Stress Testing?**

Production systems must be validated before deployment. Stress tests ensure reliability and catch issues early.

---

### 4. Logging & Monitoring

**Thread-Safe Logging** (`ml/utils/logging_config.py`):
- Production-grade logging with patient mode enforcement
- Three output modes: patient (simple), clinician (technical), dev (debug)
- Thread-safe for multi-user web simulator

**Performance Monitoring** (`ml/utils/monitoring.py`):
- Real-time metrics tracking
- Bottleneck identification
- Dashboard integration

**Why Monitoring?**

Production systems need:
- Debugging capabilities
- Performance tracking
- Issue detection
- User feedback

---

## 📁 Repository Structure

```
2026-Prototype/
├── ml/                          # Core ML code
│   ├── models/                  # Model architectures
│   │   ├── maxsight_cnn.py      # Main CNN (2103 lines)
│   │   ├── heads/               # 20 specialized output heads
│   │   │   ├── depth_head.py
│   │   │   ├── sound_event_head.py
│   │   │   ├── scene_description_head.py
│   │   │   ├── personalization_head.py
│   │   │   └── ... (16 more heads)
│   │   ├── backbone/            # Backbone architectures
│   │   │   ├── vit_backbone.py
│   │   │   └── hybrid_backbone.py
│   │   ├── fusion/              # Multi-modal fusion
│   │   ├── temporal/            # Temporal processing
│   │   └── scene_graph/        # Scene graph encoding
│   │
│   ├── training/               # Training infrastructure
│   │   ├── train_loop.py        # Production training loop + stress tests
│   │   ├── task_balancing.py    # GradNorm, PCGrad, loss monitoring
│   │   ├── losses.py            # Multi-task losses
│   │   ├── metrics.py           # Evaluation metrics (mAP, precision, recall)
│   │   ├── matching.py          # Hungarian matching for detection
│   │   ├── scene_metrics.py     # Scene-level metrics
│   │   ├── evaluation.py        # Evaluation reports
│   │   ├── benchmark.py         # Inference latency benchmarking
│   │   ├── quantization.py      # INT8 quantization
│   │   └── export.py            # Model export (CoreML, ExecuTorch, JIT, ONNX)
│   │
│   ├── data/                    # Dataset utilities
│   │   ├── dataset.py           # MaxSightDataset (COCO, audio, environmental)
│   │   ├── create_accessibility_dataset.py  # Therapy-focused dataset
│   │   ├── download_datasets.py # Dataset downloaders
│   │   ├── generate_annotations.py  # Annotation generation
│   │   ├── inference_datasets.py  # Open Images, BDD100K, ADE20K
│   │   └── advanced_augmentation.py  # Advanced data augmentation
│   │
│   ├── retrieval/               # Retrieval system
│   │   ├── encoders/            # Feature encoders
│   │   │   ├── global_encoder.py
│   │   │   ├── region_extractor.py
│   │   │   ├── patch_extractor.py
│   │   │   ├── depth_extractor.py
│   │   │   ├── ocr_encoder.py
│   │   │   ├── audio_encoder.py
│   │   │   └── scene_graph_encoder.py
│   │   ├── indexing/            # FAISS indexing
│   │   │   ├── index_manager.py
│   │   │   └── neural_index_builder.py
│   │   ├── retrieval/           # Retrieval logic
│   │   │   ├── stage1_ann.py    # Fast ANN search
│   │   │   ├── stage2_rerank.py # Multi-vector reranking
│   │   │   ├── concept_retrieval.py
│   │   │   └── knowledge_augment.py
│   │   └── fusion/              # Fusion training
│   │
│   ├── therapy/                 # Therapy system
│   │   ├── task_generator.py    # Task generation logic
│   │   ├── session_manager.py   # Session tracking
│   │   └── therapy_integration.py  # Therapy feedback
│   │
│   └── utils/                   # Utilities
│       ├── preprocessing.py     # Meta AI-style preprocessing
│       ├── output_scheduler.py  # Output scheduling
│       ├── error_handling.py    # Error handling, kill switches, ethical safeguards
│       ├── schema_validator.py  # Schema validation + stress tests
│       ├── logging_config.py    # Thread-safe logging with patient mode
│       ├── monitoring.py        # Performance monitoring
│       ├── ocr_integration.py   # OCR integration
│       ├── description_generator.py  # Natural language descriptions
│       └── spatial_memory.py    # Spatial memory system
│
├── app/                         # Application code
│   ├── overlays/                # Overlay engine
│   │   └── overlay_engine.py
│   ├── ui/                      # UI components
│   │   ├── voice_feedback.py    # Voice prompts
│   │   └── haptic_feedback.py   # Haptic feedback
│   └── session_manager/         # Session management
│
├── tools/                       # Development tools
│   ├── quantization/           # Quantization tools
│   └── simulation/             # Simulation harness
│       ├── web_simulator.py     # Multi-user web simulator
│       └── simulator/           # Simulator components
│
├── scripts/                     # Training scripts
│   ├── train_maxsight.py       # Main training script
│   ├── run_stress_tests.py     # Stress test runner
│   └── setup_coco_splits.py    # Dataset setup
│
├── tests/                       # Test suite
│   ├── test_model.py
│   ├── test_integration.py
│   └── test_performance.py
│
├── checkpoints/                 # Model checkpoints
├── datasets/                    # Training data
├── docs/                        # Documentation
└── exports/                     # Exported models (CoreML, ExecuTorch)
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.12+
- PyTorch 2.5.0+ (with MPS support for Apple Silicon)
- macOS with Apple Silicon M1+ (for iOS development)
- Xcode 16.1+ (for iOS app)

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

### Training

```bash
# Train model with GradNorm task balancing
python scripts/train_maxsight.py \
    --data-dir datasets/coco \
    --epochs 100 \
    --batch-size 32 \
    --device mps \
    --use-gradnorm
```

### Stress Testing

```bash
# Run stress tests to validate system stability
python scripts/run_stress_tests.py \
    --checkpoint checkpoints/model.pt \
    --config configs/stress_test_config.json
```

### Export for iOS

```python
from ml.training import export_to_executorch
from ml.models.maxsight_cnn import create_model

model = create_model()
export_to_executorch(model, "model.pte", input_size=(1, 3, 224, 224))
```

---

## 🎯 Key Design Decisions & Rationale

### Why Multi-Task Architecture?

**Problem**: Users need multiple types of information (detection, distance, urgency, etc.)

**Solution**: Single model with multiple heads sharing features

**Benefits**:
- **Efficiency**: Extract features once, use for all tasks
- **Consistency**: All tasks see the same visual information
- **Transfer Learning**: Features learned for one task help others
- **Mobile-Friendly**: Single model is faster than multiple models

**Trade-offs**:
- More complex training (need task balancing)
- Potential gradient conflicts (solved with GradNorm)

---

### Why Condition-Specific Preprocessing?

**Problem**: Different vision conditions require different processing

**Solution**: Preprocessing adapts images based on user's condition

**Benefits**:
- Tailored to each user's needs
- Improves model performance for specific conditions
- Enables condition-specific features

**Trade-offs**:
- More preprocessing overhead
- Requires user to specify condition

---

### Why GradNorm Task Balancing?

**Problem**: With 20+ heads, some heads dominate training while others starve

**Solution**: GradNorm adaptively balances loss weights

**Benefits**:
- All heads learn effectively
- Prevents gradient warfare
- Automatic weight adjustment

**Trade-offs**:
- More complex training loop
- Requires monitoring

---

### Why Retrieval System?

**Problem**: Single forward pass may miss context from similar scenes

**Solution**: Multi-vector retrieval with knowledge augmentation

**Benefits**:
- Leverages similar scenes from training data
- Knowledge graph integration
- More sophisticated scene understanding

**Trade-offs**:
- Additional inference overhead
- Requires indexing infrastructure

---

### Why Kill Switches?

**Problem**: Some heads may fail or need to be disabled for performance

**Solution**: Runtime head enable/disable with graceful degradation

**Benefits**:
- Performance optimization
- Debugging capabilities
- Graceful degradation

**Trade-offs**:
- Reduced functionality when heads disabled
- Requires careful head categorization

---

## 👁️ Vision Conditions Supported (10 Types)

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

## 📊 Performance Targets

- **Inference Latency**: <500ms (target: <400ms)
- **Model Size**: <50MB (quantized)
- **Battery Drain**: <12% per hour normal use
- **Detection Accuracy**: >85% in varied environments
- **OCR Accuracy**: >90% text recognition
- **Sound Classification**: >80% accuracy
- **Gradient Stability**: No gradient warfare (monitored via GradNorm)
- **System Reliability**: Graceful degradation when heads fail

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test model
python tests/test_model.py

# Test preprocessing
python -m ml.utils.preprocessing

# Run stress tests
python scripts/run_stress_tests.py
```

---

## 📝 Usage Examples

### Training with GradNorm Task Balancing

```python
from ml.training import ProductionTrainLoop, MaxSightLoss
from ml.training.task_balancing import GradNormMultiHeadLoss, GradNormStressIntegrator
from ml.data import MaxSightDataset
from ml.models.maxsight_cnn import create_model

# Create model
model = create_model(num_classes=48)

# Initialize GradNorm loss with stress monitoring
head_losses = {
    'detection': MaxSightLoss(num_classes=48),
    'depth': DepthLoss(),
    'urgency': UrgencyLoss(),
    # ... other heads
}

gradnorm_loss = GradNormMultiHeadLoss(
    head_losses=head_losses,
    shared_params=list(model.backbone.parameters()),
    alpha=1.5,
    update_interval=50
)

# Integrate with stress monitoring
stress_integrator = GradNormStressIntegrator(
    loss_module=gradnorm_loss,
    monitor_window=100,
    auto_dampen=True  # Automatically reduce weights for problematic heads
)

# Train with integrated monitoring
trainer = ProductionTrainLoop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=stress_integrator,  # Use integrator instead of raw loss
    num_epochs=100,
    device="mps"
)
results = trainer.train()

# Check for gradient warfare issues
metrics_summary = stress_integrator.get_metrics_summary()
if metrics_summary['detected_issues']:
    print("⚠️ Gradient warfare detected:", metrics_summary['detected_issues'])
```

### Runtime Head Kill Switches

```python
from ml.utils.error_handling import HeadKillSwitchManager, wrap_heads_with_killswitch

# Create kill switch manager
kill_switch = HeadKillSwitchManager()

# Disable non-critical heads for performance
kill_switch.disable_heads_by_category('optional')

# Wrap model with kill switches
model = wrap_heads_with_killswitch(model, kill_switch)

# Runtime control
kill_switch.disable_head('motion')  # Disable motion head
kill_switch.enable_head('motion')   # Re-enable
```

### Ethical Safeguards

```python
from ml.utils.error_handling import EthicalGuard, apply_ethical_guards

# Apply ethical safeguards to model outputs
guard = EthicalGuard(
    uncertainty_threshold=0.7,
    suppression_mode='soft',
    enable_safety_checks=True
)

outputs = model(images)
guarded = guard.guard_outputs(outputs)

if not guarded['safety_info']['safe']:
    print("⚠️ Unsafe outputs detected:", guarded['safety_info']['reasons'])
```

### Export

```python
from ml.training import export_to_executorch, export_to_coreml

# Export for iOS
export_to_executorch(model, "model.pte")
export_to_coreml(model, "model.mlpackage")
```

---

## 🔗 Key Modules Reference

| Module | Purpose | Why It Exists |
|--------|---------|---------------|
| `ml.models.maxsight_cnn` | Main CNN architecture | Core multi-task vision model |
| `ml.models.backbone.hybrid_backbone` | Hybrid CNN-ViT backbone | Combines CNN efficiency with ViT attention |
| `ml.models.attention.attention` | Unified attention suite | CBAM, Cross-Modal, Cross-Task attention |
| `ml.training.train_loop` | Production training loop | Robust training with stress tests |
| `ml.training.task_balancing` | GradNorm, PCGrad, loss monitoring | Prevents gradient warfare in multi-task learning |
| `ml.utils.error_handling` | Error handling, kill switches, ethical safeguards | Production safety and reliability |
| `ml.utils.schema_validator` | Schema validation + stress tests | Ensures output correctness |
| `ml.utils.logging_config` | Thread-safe logging with patient mode | Production logging for debugging |
| `ml.training.export` | Model export (iOS-ready) | Deploys models to mobile devices |
| `ml.data.dataset` | Dataset loading | Loads COCO + accessibility data |
| `ml.utils.preprocessing` | Image preprocessing | Condition-specific adaptations |
| `ml.utils.output_scheduler` | Output scheduling | Cross-modal output management |
| `ml.retrieval` | Retrieval system | Multi-vector retrieval with knowledge augmentation |
| `ml.therapy` | Therapy system | Adaptive therapy exercises and skill development |

---

## 📄 License

See [LICENSE](LICENSE) file.

---

## 📚 Additional Documentation

- [System Architecture](docs/SYSTEM_ARCHITECTURE.md) - Detailed architecture documentation
- [System Limitations](docs/SYSTEM_LIMITATIONS.md) - Known limitations and failure modes
- [Application Analysis](docs/APPLICATION_ANALYSIS.md) - Comprehensive codebase analysis
- [Stress Testing Guide](docs/STRESS_TESTING_GUIDE.md) - How to run stress tests

---

**Status**: 🟢 Active Development  
**Timeline**: 60 days (Nov 15, 2025 - Jan 14, 2026)  
**Platform**: iOS (iOS 17+)  
**Tech Stack**: PyTorch, ExecuTorch, CoreML

---

## 🤝 Contributing

This is a research prototype. For questions or contributions, please refer to the documentation in `docs/`.

---

## 🙏 Acknowledgments

MaxSight is designed based on accessibility research and barrier-removal methods. The system implements condition-specific adaptations and multimodal communication strategies to support users with vision and hearing disabilities.
