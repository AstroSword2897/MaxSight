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

### Safety-First, Tiered Architecture

MaxSight uses a **tiered head architecture** where heads are organized by criticality, not treated as equals. This ensures safety-critical predictions are never blocked by enhancement features.

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
│         Condition Adapter (Learnable Embedding + FiLM)      │
│         Adapts to user's vision condition                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Shared Backbone (Mobile-Optimized CNN/ViT)          │
│         └── Feature Pyramid Network (FPN)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────────┐         ┌───────────────────────┐
│  Stage A: Safety Pass  │         │ Stage B: Context Pass │
│  (<150ms, Every Frame) │         │ (Opportunistic)       │
│                        │         │                       │
│  Tier 1 Heads:        │         │ Tier 2 Heads:        │
│  ├── Objectness        │         │ ├── Motion           │
│  ├── Classification    │         │ ├── ROI Priority     │
│  ├── Box Regression    │         │ ├── Scene Complexity │
│  ├── Distance (zones)  │         │ └── Spatial Memory  │
│  ├── Urgency           │         │                       │
│  └── Uncertainty       │         │ Tier 3 Heads:        │
│                        │         │ ├── Scene Desc       │
│  Safety Decision Core  │         │ ├── Retrieval (adv)  │
│  "Is user safe now?"   │         │ ├── Therapy          │
│                        │         │ └── Fatigue          │
└───────────────────────┘         └───────────────────────┘
        ↓                                       ↓
┌─────────────────────────────────────────────────────────────┐
│         Output Scheduler + Cognitive Load Model            │
│         ├── Information Budget (per-second)                │
│         ├── Cool-down per object                           │
│         ├── Novelty Detection                               │
│         └── Operating Modes (Safe/Assist/Therapy)          │
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

### Tiered Head Architecture (Criticality Layers)

Heads are organized into **3 tiers** based on safety criticality:

#### **Tier 1: Safety-Critical (Never Disabled)**
- **Objectness**: Is there an object here?
- **Classification**: What object is this?
- **Box Regression**: Where is it?
- **Distance (coarse zones)**: Near/medium/far only
- **Urgency**: How dangerous?
- **Uncertainty**: How confident?

**Properties:**
- Highest loss priority in training
- Redundant validation
- Separate inference budget
- Runs every frame (<150ms target)
- Never blocked by other heads

#### **Tier 2: Navigation & Context (Can Degrade)**
- **Motion**: Object movement tracking
- **ROI Priority**: Region-of-interest prioritization
- **Scene Complexity**: Navigation difficulty
- **Spatial Memory**: Object tracking over time
- **Path Planning**: Navigation assistance

**Properties:**
- Can be throttled (every N frames)
- Can be delayed if Tier 1 needs resources
- Graceful degradation if disabled

#### **Tier 3: Enhancement & Therapy (Optional)**
- **Scene Description**: Natural language descriptions
- **Retrieval Augmentation**: Knowledge-enhanced context
- **Therapy**: Vision therapy exercises
- **Fatigue**: User fatigue detection

**Properties:**
- Optional (can be disabled)
- Asynchronous (background thread)
- Never blocks Tier 1 or Tier 2
- Advisory only (never drives safety decisions)

### Two-Stage Inference Pipeline

**Stage A: Fast Safety Pass** (<150ms, every frame)
- Minimal backbone + Tier 1 heads only
- Answers: "Is the user safe right now?"
- Early-exit after FPN for speed
- Feature caching for Stage B

**Stage B: Context Pass** (opportunistic)
- Runs only if Stage A is stable
- Uses cached features from Stage A
- Tier 2 & Tier 3 heads
- Can be skipped if latency is high

**Why This Matters:**
- Decouples safety from enhancement
- Reduces latency variance
- Makes debugging easier
- Enables graceful degradation

### Key Design Principles

1. **Safety-First**: Tier 1 heads never disabled, highest priority
2. **Tiered Criticality**: Not all heads are equal—safety > navigation > enhancement
3. **Two-Stage Inference**: Fast safety pass, opportunistic context pass
4. **Cognitive Load Management**: Information budgeting prevents overload
5. **Fail-Silent Modes**: Operating modes (Safe/Assist/Therapy) for predictable behavior
6. **Advisory Retrieval**: Retrieval enhances, never drives safety decisions
7. **Learnable Condition Adaptation**: FiLM layers adapt to user's vision condition
8. **Hybrid Distance**: Coarse zones (near/medium/far) only, not dense depth maps
9. **Motion as Temporal Anchor**: Motion features condition other heads (depth, contrast, OCR, fatigue)
10. **Simplified Fusion**: 2-tier hierarchy (cross-layer residual primary, weighted fusion secondary)
11. **Production-Ready Backbone**: Constrained cross-layer alpha, safe feature caching, GPU-parallel dynamic conv

---

## 🔍 Core Components Deep Dive

### 1. MaxSightCNN (`ml/models/maxsight_cnn.py`)

**Purpose**: Core multi-task vision model that powers all environmental understanding.

**Architecture**:
- **Backbone**: Hybrid CNN-ViT (ResNet50 + Vision Transformer) - combines CNN efficiency with ViT attention
- **Neck**: Simplified FPN (Feature Pyramid Network) - multi-scale feature extraction
- **Heads**: 20 specialized task-specific heads - organized by criticality tiers

**Why Hybrid CNN-ViT?**
- **CNN**: Provides local inductive bias and spatial precision
- **ViT**: Provides global context and long-range reasoning
- **Cross-Layer Interaction**: Learnable residual connections (CNN ↔ ViT) with sigmoid-constrained alpha
- **Simplified Fusion**: 2-tier hierarchy (cross-layer residual primary, weighted fusion secondary)

**Production-Ready Features**:
- **GPU-Parallel Dynamic Conv**: Grouped convolution trick (not per-sample loop)
- **Safe Feature Caching**: Frame ID-based keys (not mean hash)
- **Constrained Cross-Layer Alpha**: Sigmoid bounds prevent runaway amplification
- **Motion as Temporal Anchor**: Motion features condition depth, contrast, OCR, fatigue

**Input**: `[B, 3, 224, 224]` RGB images + optional audio features  
**Output**: Dictionary with 20+ task outputs (detections, urgency, distance, etc.)

**Key Features**:
- Anchor-free detection (FCOS-style) - simpler and more efficient than anchor-based
- Multi-scale feature extraction - detects objects of all sizes
- Audio-visual fusion - combines vision and sound for better understanding
- Condition-specific preprocessing - learnable FiLM adapters (not hard-coded)
- Global Confidence Aggregator - system-level uncertainty (not isolated head)

---

### 2. Tiered Head Architecture

Heads are organized into **3 tiers** based on safety criticality. This ensures safety-critical predictions are never blocked by enhancement features.

#### Tier 1: Safety-Critical Heads (Never Disabled)

| Head | Purpose | Output Shape | Why It Exists | Execution |
|------|---------|--------------|---------------|-----------|
| **Objectness** | Is there an object here? | `[B, 196]` | Filters background noise | Every frame |
| **Classification** | What object is this? | `[B, 196, 48]` | Identifies 48 environmental classes | Every frame |
| **Box Regression** | Where is the object? | `[B, 196, 4]` | Bounding box coordinates | Every frame |
| **Distance (zones)** | How far? (coarse only) | `[B, 3]` | Near/medium/far zones only | Every frame |
| **Urgency** | How dangerous? | `[B, 4]` | 4-level urgency (safe/caution/warning/danger) | Every frame |
| **Uncertainty** | Model confidence | `[B, 1]` | Critical for safety—suppresses unsafe outputs | Every frame |

**Properties:**
- Highest loss priority in training
- Redundant validation
- Separate inference budget
- Target: <150ms per frame
- Never blocked by Tier 2 or Tier 3

#### Tier 2: Navigation & Context Heads (Can Degrade)

| Head | Purpose | Output Shape | Why It Exists | Execution |
|------|---------|--------------|---------------|-----------|
| **Motion** | Object movement | `[B, 2, H, W]` | Tracks motion for predictive alerts | Every N frames |
| **ROI Priority** | Region prioritization | `[B, N]` | Identifies important regions | Every N frames |
| **Scene Complexity** | Navigation difficulty | `[B, 1]` | Overall scene complexity | Every N frames |
| **Spatial Memory** | Object tracking | Persistent | Maintains object map over time | Background |
| **Path Planning** | Navigation assistance | Path graph | Plans safe navigation paths | Background |

**Properties:**
- Can be throttled (every N frames)
- Can be delayed if Tier 1 needs resources
- Graceful degradation if disabled

#### Tier 3: Enhancement & Therapy Heads (Optional)

| Head | Purpose | Output Shape | Why It Exists | Execution |
|------|---------|--------------|---------------|-----------|
| **Scene Description** | Natural language | `[B, 512]` | Encodes scene for TTS | Background |
| **Retrieval Augmentation** | Knowledge enhancement | Advisory | Enhances descriptions, never drives safety | Background |
| **Therapy** | Vision therapy | Task config | Adaptive therapy exercises | Background |
| **Fatigue** | User fatigue | `[B, 1]` | Detects user fatigue (experimental) | Background |

**Properties:**
- Optional (can be disabled)
- Asynchronous (background thread)
- Never blocks Tier 1 or Tier 2
- **Advisory only** (never drives safety decisions)

**Why Tiered Architecture?**

Not all predictions are equal. Safety-critical predictions (Tier 1) must never be blocked by enhancement features (Tier 3). This architecture ensures:
- **Safety first**: Tier 1 always runs, never disabled
- **Graceful degradation**: If Tier 2/3 fail, Tier 1 continues
- **Resource management**: Tier 1 gets priority, Tier 2/3 are opportunistic
- **Predictable behavior**: Users know safety features always work

---

### 3. Learnable Condition Adapter (`ml/utils/preprocessing.py`)

**Purpose**: Adapts processing to user's vision condition using learnable embeddings, not hard-coded rules.

**Why Learnable Instead of Hard-Coded?**

Hard-coded preprocessing rules:
- Don't scale to new conditions
- Can backfire if assumptions are wrong
- Require manual tuning for each condition

Learnable condition adapters:
- Learn what actually helps users
- Adapt over time based on feedback
- Reduce manual heuristics

**How It Works**:
1. **Condition Embedding Vector**: Learned representation of each vision condition
2. **FiLM Layers** (Feature-wise Linear Modulation): Inject condition embedding into backbone
3. **Attention Modulation**: Condition-specific attention weights
4. **Fallback Heuristics**: Safety fallback if learning fails

**Architecture**:
```
Condition Embedding [C_dim] 
    ↓
FiLM Layers (modulate backbone features)
    ↓
Condition-Adapted Features
    ↓
Model Processing
```

**Benefits**:
- Learns optimal adaptations per condition
- Adapts to user feedback
- Reduces manual tuning
- Maintains safety fallbacks

This is more robust than hard-coded rules while maintaining safety.

---

### 5. OCR Integration (`ml/utils/ocr_integration.py`)

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

### 6. Description Generator (`ml/utils/description_generator.py`)

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

### 7. Spatial Memory (`ml/utils/spatial_memory.py`)

**Purpose**: Maintains a memory of objects and their positions over time.

**Why Spatial Memory?**

Users need to:
- Remember where objects were
- Track objects as they move
- Build a mental map of their environment
- Navigate based on remembered locations

Spatial memory enables these capabilities by maintaining a persistent object map.

---

### 8. Path Planner (`ml/utils/path_planner.py`)

**Purpose**: Plans navigation paths based on detected objects and spatial memory.

**Why Path Planning?**

Users need:
- Safe paths around obstacles
- Efficient routes to destinations
- Real-time navigation assistance
- Obstacle avoidance

Path planner combines detection, spatial memory, and navigation logic to provide these capabilities.

---

### 9. Therapy Integration (`ml/therapy/`)

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

### 10. Retrieval System (`ml/retrieval/`) - **Advisory Only**

**Purpose**: Multi-vector retrieval for knowledge-augmented scene understanding. **Advisory, not authoritative.**

**Why Advisory Retrieval?**

Retrieval is powerful but risky if it drives safety decisions. Retrieval can:
- Fail (network issues, index corruption)
- Be slow (latency spikes)
- Be wrong (similar scenes aren't identical)

**Architectural Rule:**
> **Retrieval outputs may only influence Tier 3 heads. No retrieval signal should affect urgency, distance, or path planning.**

**What Retrieval Does:**
- **Enhances descriptions**: Adds context from similar scenes
- **Suggests context**: Provides background information
- **Never drives safety**: Tier 1 heads are retrieval-independent

**What Retrieval Doesn't Do:**
- ❌ Affect urgency predictions
- ❌ Affect distance estimation
- ❌ Affect path planning
- ❌ Block safety decisions

**If Retrieval Fails:**
System behaves identically to "no internet"—Tier 1 and Tier 2 continue normally, only Tier 3 descriptions are affected.

**Components**:
- **Encoders**: Extract embeddings from images, audio, text, depth, scene graphs
- **Indexing**: FAISS-based indexing for fast similarity search
- **Two-Stage Retrieval**: Fast ANN search + multi-vector reranking
- **Knowledge Augmentation**: GNN-based knowledge graph integration (Tier 3 only)

**Why This Matters:**

Retrieval is enhancement, not core functionality. By making it advisory, we ensure:
- Safety decisions are never dependent on retrieval
- System degrades gracefully if retrieval fails
- Latency spikes in retrieval don't affect safety
- Users can trust that safety features always work

---

### 11. Output Scheduler with Cognitive Load Model (`ml/utils/output_scheduler.py`)

**Purpose**: Cross-modal output management with cognitive load budgeting.

**Why Cognitive Load Model?**

Users have limited cognitive bandwidth. Information overload causes:
- Missed critical alerts
- Confusion and frustration
- Reduced trust in the system

The scheduler manages **information budget**, not just priority:
- **Per-second information budget**: Maximum announcements per second
- **Cool-down per object**: Never announce same object twice in 5 seconds
- **Novelty detection**: Suppress redundant information
- **Motion-aware throttling**: Reduce non-urgent info when user is moving fast

**Operating Modes:**

1. **Safe Mode**: Only Tier 1 heads, minimal output, maximum safety
2. **Assist Mode**: Full navigation + context, balanced information
3. **Therapy Mode**: Rich feedback, slow pace, detailed descriptions

Mode switches triggered by:
- Uncertainty spikes (>0.7)
- Latency spikes (>500ms)
- User motion speed
- Battery state

**Why This Matters:**

Priority management is reactive. Cognitive load management is **proactive**—it prevents overload before it happens. This turns the scheduler from reactive → intelligent.

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

### Why Learnable Condition Adapters?

**Problem**: Hard-coded preprocessing rules don't scale and can backfire.

**Solution**: Learnable condition adapters using FiLM layers instead of hard-coded rules.

**Benefits**:
- Learns what actually helps users (data-driven)
- Adapts over time based on feedback
- Reduces manual heuristics
- Maintains safety fallbacks

**Trade-offs**:
- Requires training data per condition
- More complex than hard-coded rules
- Fallback heuristics still needed for safety

**Why This Matters**: Hard-coded rules assume we know what helps. Learning from data is more robust and adapts to real user needs.

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
- Additional inference overhead (but optional)
- Requires indexing infrastructure

**Why This Matters**: Retrieval is enhancement, not core. By making it advisory, we ensure safety decisions are never dependent on retrieval.

---

### Why Fail-Silent Modes?

**Problem**: Sometimes the safest action is silence, not more information.

**Solution**: Operating modes (Safe/Assist/Therapy) with different behavior profiles.

**Benefits**:
- **Predictable Behavior**: Users know what to expect in each mode
- **Safety-First**: Safe mode prioritizes safety over features
- **Context-Aware**: Mode switches based on uncertainty, latency, motion
- **Trust**: Users trust the system because it's predictable

**Trade-offs**:
- More complex state management
- Requires careful mode design

**Modes**:
- **Safe Mode**: Only Tier 1 heads, minimal output, maximum safety
- **Assist Mode**: Full navigation + context, balanced information
- **Therapy Mode**: Rich feedback, slow pace, detailed descriptions

### Why Kill Switches?

**Problem**: Some heads may fail or need to be disabled for performance

**Solution**: Runtime head enable/disable with graceful degradation

**Benefits**:
- Performance optimization
- Debugging capabilities
- Graceful degradation
- **Tier-based**: Can disable Tier 2/3 without affecting Tier 1

**Trade-offs**:
- Reduced functionality when heads disabled
- Requires careful tier assignment

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

## 📊 Performance Targets & Safety Metrics

### Standard Performance Metrics

- **Inference Latency**: 
  - Stage A (Safety Pass): <150ms (target: <100ms)
  - Stage B (Context Pass): <500ms (opportunistic)
- **Model Size**: <50MB (quantized)
- **Battery Drain**: <12% per hour normal use
- **Detection Accuracy**: >85% in varied environments
- **OCR Accuracy**: >90% text recognition
- **Sound Classification**: >80% accuracy

### Safety-Focused Metrics (More Important Than Accuracy)

- **False Reassurance Rate**: <1% (danger predicted as safe)
- **Alert Latency**: <200ms (time to first warning)
- **Information Overload Events**: <2 per minute (cognitive budget violations)
- **Silence Correctness**: >95% (when staying quiet was right)
- **Head Collapse Detection Time**: <10 seconds (detect failing heads)
- **Tier 1 Availability**: >99.9% (safety heads never disabled)
- **Uncertainty Calibration**: Well-calibrated (uncertainty correlates with actual error)

**Why Safety Metrics Matter:**

mAP and accuracy don't capture safety. A 95% accurate system that gives false reassurance is worse than an 85% accurate system that's safe. These metrics ensure the system is trustworthy, not just accurate.

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
- **[Maintenance Survival Map](docs/MAINTENANCE_SURVIVAL_MAP.md)** - One-page guide for long-term system health ⭐

### Maintenance & Health Checks

**Health Checks** (`ml/utils/monitoring.py`):
- `HealthChecker` class for daily Tier 1 head validation
- Run via `scripts/run_stress_tests.py` (includes health check)
- Checks: objectness, classification, box regression, distance, urgency, uncertainty
- Latency monitoring and model integrity checks

**Backup** (`scripts/train_maxsight.py`):
- Use `--backup` flag after training to backup models, code (git bundle), and data metadata
- Weekly automated backups recommended
- Backup location: `backups/YYYYMMDD/`

**Usage**:
```bash
# Health check (runs automatically with stress tests)
python scripts/run_stress_tests.py --checkpoint checkpoints/model.pt

# Backup after training
python scripts/train_maxsight.py --data-dir datasets --backup
```

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
