# MaxSight CNN Architecture Design

## Multi-Level Architecture Overview

The MaxSight CNN uses a multi-level architecture with shared feature extraction and specialized output heads for different tasks.

## Architecture Levels

### Level 1: Shared Backbone (Feature Extraction)

**Purpose:** Extract common visual features from input images

**Components:**
- ResNet-50 or EfficientNet backbone
- Feature Pyramid Network (FPN) for multi-scale features
- Shared convolutional layers

**Output:** Multi-scale feature maps at different resolutions

### Level 2: Specialized Output Heads

Each head processes shared features to produce task-specific outputs:

#### 2.1 Classification Head
- **Purpose:** Object detection and classification
- **Output:** Bounding boxes, class labels, confidence scores
- **Architecture:** Anchor-free detection (FCOS-style)
- **Classes:** 80 COCO classes + accessibility-specific classes

#### 2.2 Localization Head
- **Purpose:** Precise object localization
- **Output:** Refined bounding box coordinates
- **Architecture:** Regression layers with IoU prediction

#### 2.3 Description Head
- **Purpose:** Generate natural language scene descriptions
- **Output:** Text descriptions of detected objects and scenes
- **Architecture:** LSTM/Transformer-based text generation
- **Input:** Scene embedding + detected objects

#### 2.4 Urgency Head
- **Purpose:** Assess urgency/priority of detected objects
- **Output:** Urgency scores (0-3: low, medium, high, critical)
- **Architecture:** Multi-class classification head
- **Factors:** Object type, distance, motion, context

#### 2.5 Distance Head
- **Purpose:** Estimate distance to detected objects
- **Output:** Distance zones (near, medium, far) and precise distances
- **Architecture:** Regression head with zone classification
- **Methods:** Monocular depth estimation + object size cues

#### 2.6 Text Region Detection Head
- **Purpose:** Detect text regions in images
- **Output:** Text bounding boxes and confidence scores
- **Architecture:** Specialized detection head for text
- **Integration:** Feeds into OCR pipeline

#### 2.7 Audio Fusion Branch
- **Purpose:** Integrate audio features for multi-modal understanding
- **Input:** Visual features + audio features (if available)
- **Architecture:** Cross-modal attention mechanism
- **Output:** Enhanced scene understanding with audio context

### Level 3: Therapy-Specific Heads

#### 3.1 Contrast Head
- **Purpose:** Estimate local contrast for contrast sensitivity therapy
- **Output:** Contrast maps per image region
- **Architecture:** Edge-aware contrast estimation

#### 3.2 Depth Head
- **Purpose:** Depth estimation for depth perception therapy
- **Output:** Depth maps and zone classifications
- **Architecture:** Monocular depth estimation network

#### 3.3 Motion Head
- **Purpose:** Optical flow estimation for motion tracking therapy
- **Output:** Motion vectors and flow fields
- **Architecture:** Optical flow network

#### 3.4 ROI Priority Head
- **Purpose:** Rank regions of interest by importance
- **Output:** Priority scores for different image regions
- **Architecture:** Attention-based ranking network

#### 3.5 Uncertainty Head
- **Purpose:** Estimate prediction uncertainty
- **Output:** Uncertainty scores per detection
- **Architecture:** Bayesian or ensemble-based uncertainty estimation

#### 3.6 Fatigue Head
- **Purpose:** Monitor user eye fatigue during therapy
- **Input:** Eye tracking features + temporal context
- **Output:** Fatigue level estimates
- **Architecture:** LSTM-based temporal modeling

## Multi-Task Learning

All heads share the backbone and are trained jointly with a weighted multi-task loss:

```
Total Loss = w1 * Classification Loss + 
             w2 * Localization Loss + 
             w3 * Description Loss + 
             w4 * Urgency Loss + 
             w5 * Distance Loss + 
             w6 * Text Detection Loss + 
             w7 * Therapy Head Losses
```

## Condition-Specific Adaptations

The architecture adapts to different vision conditions:

1. **Preprocessing Stage:** Condition-specific image enhancement
2. **Feature Extraction:** Adaptive feature emphasis based on condition
3. **Head Prioritization:** Different heads prioritized per condition
4. **Output Scheduling:** Condition-aware output filtering and prioritization

## Model Specifications

- **Input Size:** 224x224 or 416x416 (configurable)
- **Backbone:** ResNet-50 or EfficientNet-B3
- **Total Parameters:** ~29M
- **Inference Speed:** <100ms on mobile (target)
- **Quantization:** INT8 quantization for mobile deployment

## Data Flow

1. **Input:** RGB image (optionally with audio features)
2. **Preprocessing:** Condition-specific enhancement
3. **Backbone:** Feature extraction
4. **Heads:** Parallel processing by all heads
5. **Post-processing:** NMS, filtering, scheduling
6. **Output:** Multi-modal results (visual, audio, haptic)

## Training Strategy

- **Multi-task learning:** All heads trained jointly
- **Curriculum learning:** Start with classification, add therapy heads gradually
- **Condition-specific fine-tuning:** Fine-tune on condition-specific datasets
- **Transfer learning:** Pre-trained on COCO, fine-tuned on accessibility data
