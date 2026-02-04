# MaxSight 3.0 - Complete Technical Specifications

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Input/Output Specifications](#inputoutput-specifications)
3. [Capability Tiers](#capability-tiers)
4. [Head Specifications](#head-specifications)
5. [Backbone Specifications](#backbone-specifications)
6. [Training Specifications](#training-specifications)
7. [Hardware Requirements](#hardware-requirements)
8. [Performance Targets](#performance-targets)
9. [Security Specifications](#security-specifications)
10. [Export Specifications](#export-specifications)

---

## Model Architecture

### Core Model: MaxSightCNN

**File**: `ml/models/maxsight_cnn.py`

**Base Architecture**:
- **Backbone**: ResNet50 (pretrained ImageNet) + Simplified FPN
- **Detection**: Anchor-free (FCOS-style)
- **Multi-task**: 30+ simultaneous task outputs
- **Two-stage inference**: Stage A (safety) + Stage B (context)

**Total Parameters**:
- **T0 (Baseline)**: ~98M parameters
- **T2 (Hybrid ViT)**: ~210M parameters
- **T5 (Temporal)**: ~320M parameters

---

## Input/Output Specifications

### Input Specifications

**Primary Input**:
- **Images**: `[B, 3, 224, 224]` RGB tensor
  - Batch size: Variable (B)
  - Channels: 3 (RGB)
  - Height: 224 pixels
  - Width: 224 pixels
  - Normalization: ImageNet mean/std
  - Data type: `torch.float32`

**Temporal Input** (T5 only):
- **Video**: `[B, T, 3, 224, 224]` RGB tensor
  - Batch size: Variable (B)
  - Temporal frames: Variable (T, typically 8-16)
  - Channels: 3 (RGB)
  - Spatial: 224x224

**Optional Inputs**:
- **Audio**: `[B, 128]` audio features (T4+)
- **User ID**: `[B]` user identifiers for personalization
- **Previous Temporal State**: Dict for temporal continuity (T5)

### Output Specifications

**Stage A Outputs** (Always produced, <150ms):
```python
{
    'objectness': [B, H*W],           # Object confidence scores
    'classification': [B, H*W, num_classes],  # Class logits (91 classes)
    'boxes': [B, H*W, 4],             # Bounding boxes (cx, cy, w, h)
    'distance_zones': [B, H*W, 3],    # Distance zone probabilities
    'urgency_scores': [B, 4],         # Scene-level urgency (safe/caution/warning/danger)
    'uncertainty': [B, H*W],          # Per-location uncertainty
    'text_logits': [B, H*W],          # Text detection logits
    'stage_a_completed': True,
    'stage_a_latency_ms': float
}
```

**Stage B Outputs** (Tier-dependent, opportunistic):
```python
{
    # Motion & Temporal (T5)
    'motion': [B, 2, H, W],           # Optical flow (x, y)
    'motion_features': [B, T, C, H, W],  # Temporal motion features
    'consistency': [B],                # Temporal consistency score
    'flicker': [B],                    # Flicker detection score
    
    # Depth (T2+)
    'depth_map': [B, H, W],           # Depth estimation
    'depth_uncertainty': [B, H, W],    # Depth uncertainty
    'distance_zones': [B, H*W, 3],    # Refined distance zones
    
    # Therapy State (T2+)
    'fatigue_score': [B],              # Fatigue level [0, 1]
    'blink_rate': [B],                 # Blinks per second
    'fixation_stability': [B],         # Fixation stability [0, 1]
    'contrast_map': [B, H, W],         # Contrast map
    'edge_map': [B, H, W],             # Edge detection map
    
    # Scene Understanding (T3+)
    'scene_description': str,          # Natural language description
    'scene_graph': Dict,               # Scene graph with relations
    'ocr_text': List[str],             # Detected text strings
    'ocr_boxes': [B, N, 4],           # OCR bounding boxes
    
    # Audio (T4+)
    'sound_events': [B, 15],          # Sound event probabilities
    'spatial_audio': [B, 4],          # Directional audio features
    
    # Retrieval (T4+)
    'retrieval_embeddings': Dict,     # Multi-vector embeddings
    'retrieval_results': List,        # Retrieved similar scenes
    
    # Personalization (T3+)
    'attention_weights': [B, 10],    # Personalized attention
    'verbosity_level': [B],           # Verbosity level [0, 3]
    
    'stage_b_completed': bool,
    'stage_b_latency_ms': float,
    'skip_stage_b_reason': Optional[str]
}
```

---

## Capability Tiers

### Tier 0: Baseline CNN (T0_BASELINE_CNN)

**Specifications**:
- **Backbone**: ResNet50 + FPN only
- **Parameters**: ~98M
- **Heads**: Tier 1 safety-critical only
- **Latency**: Stage A <150ms, Stage B disabled
- **Memory**: ~200-300 MB inference, ~500-700 MB training
- **Use Case**: Mobile devices, edge deployment

**Features**:
- Object detection (91 classes)
- Urgency classification (4 levels)
- Distance zones (3 zones)
- Uncertainty estimation

### Tier 1: Edge (T1_EDGE)

**Specifications**:
- **Backbone**: ResNet50 + FPN + SE/CBAM attention
- **Parameters**: ~120M
- **Heads**: Tier 1 + attention-enhanced features
- **Latency**: Stage A <150ms, Stage B ~50ms
- **Memory**: ~300-400 MB inference, ~700-900 MB training

**Additional Features**:
- SE (Squeeze-and-Excitation) attention
- CBAM (Convolutional Block Attention Module)

### Tier 2: Hybrid ViT (T2_HYBRID_VIT)

**Specifications**:
- **Backbone**: ResNet50 + FPN + Hybrid CNN-ViT
- **Parameters**: ~210M
- **Heads**: Tier 1 + Tier 2 (motion, depth, therapy state)
- **Latency**: Stage A <150ms, Stage B ~80ms
- **Memory**: ~1.5-2.0 GB inference, ~3.0-4.0 GB training

**Additional Features**:
- Hybrid CNN-ViT backbone
- Dynamic convolution
- Motion head
- Depth estimation
- Therapy state head (fatigue, contrast)

### Tier 3: Cross-Modal (T3_CROSS_MODAL)

**Specifications**:
- **Backbone**: ResNet50 + FPN + Hybrid CNN-ViT
- **Parameters**: ~250M
- **Heads**: Tier 1 + Tier 2 + Tier 3 (OCR, scene description, personalization)
- **Latency**: Stage A <150ms, Stage B ~150ms
- **Memory**: ~2.0-2.5 GB inference, ~4.0-5.0 GB training

**Additional Features**:
- Cross-task attention
- OCR head (text detection & recognition)
- Scene description head (natural language)
- Personalization head
- Scene graph encoder

### Tier 4: Cross-Modal Enhanced (T4_CROSS_MODAL)

**Specifications**:
- **Backbone**: ResNet50 + FPN + Hybrid CNN-ViT
- **Parameters**: ~280M
- **Heads**: Tier 1-3 + Tier 4 (audio, retrieval)
- **Latency**: Stage A <150ms, Stage B ~200ms
- **Memory**: ~2.5-3.0 GB inference, ~5.0-6.0 GB training

**Additional Features**:
- Cross-modal attention (audio-visual)
- Enhanced audio encoder
- Sound event classification
- Spatial sound mapping
- Multi-vector retrieval system
- Knowledge-augmented retrieval

### Tier 5: Temporal (T5_TEMPORAL)

**Specifications**:
- **Backbone**: ResNet50 + FPN + Hybrid CNN-ViT + Temporal
- **Parameters**: ~320M
- **Heads**: All tiers (Tier 1-5)
- **Latency**: Stage A <150ms, Stage B ~350ms
- **Memory**: ~3.0-4.0 GB inference, ~6.0-8.0 GB training

**Additional Features**:
- ConvLSTM temporal modeling
- TimeSformer temporal attention
- Temporal consistency detection
- Flicker detection
- Predictive alert head

---

## Head Specifications

### Tier 1 Heads (Safety-Critical, Always Enabled)

#### Objectness Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W]` confidence scores
- **Architecture**: Conv2d(256→256→1)
- **Purpose**: Object presence confidence

#### Classification Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W, 91]` class logits
- **Architecture**: Conv2d(256→256→91)
- **Classes**: 91 COCO + accessibility classes

#### Box Regression Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W, 4]` box coordinates (cx, cy, w, h)
- **Architecture**: Conv2d(256→256→4)
- **Format**: Center coordinates + width/height

#### Distance Zone Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W, 3]` zone probabilities
- **Architecture**: Conv2d(256→256→3)
- **Zones**: ['near', 'medium', 'far']

#### Urgency Head
- **Input**: `[B, 512]` scene features
- **Output**: `[B, 4]` urgency logits
- **Architecture**: Linear(512→256→4)
- **Levels**: ['safe', 'caution', 'warning', 'danger']

#### Uncertainty Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W]` uncertainty scores
- **Architecture**: Conv2d(256→256→1) + Sigmoid
- **Range**: [0, 1] (0 = certain, 1 = uncertain)

#### Text Detection Head
- **Input**: `[B, 256, H, W]` FPN features
- **Output**: `[B, H*W]` text probability
- **Architecture**: Conv2d(256→128→1) + Sigmoid
- **Purpose**: Text region detection for OCR

### Tier 2 Heads (Context Enhancement)

#### Motion Head
- **Input**: `[B, T, C, H, W]` temporal features
- **Output**: `[B, 2, H, W]` optical flow (x, y)
- **Architecture**: ConvLSTM + Conv2d
- **Purpose**: Motion tracking and prediction

#### Depth Head
- **Input**: `[B, 256, H, W]` FPN features + motion
- **Output**: `[B, H, W]` depth map
- **Architecture**: Conv2d + FPN fusion
- **Purpose**: Depth estimation for navigation

#### Therapy State Head
- **Input**: Multiple (eye, motion, depth, contrast features)
- **Output**: Dict with fatigue, depth, contrast outputs
- **Architecture**: Unified multi-branch network
- **Purpose**: Therapy state monitoring

### Tier 3 Heads (Scene Understanding)

#### OCR Head
- **Input**: `[B, 256, H, W]` text regions
- **Output**: `List[str]` recognized text + `[B, N, 4]` boxes
- **Architecture**: Transformer encoder-decoder
- **Purpose**: Text detection and recognition

#### Scene Description Head
- **Input**: `[B, 512]` global + `[B, R, 256]` region features
- **Output**: `str` natural language description
- **Architecture**: Transformer decoder
- **Purpose**: Natural language scene descriptions

#### Personalization Head
- **Input**: `[B, 512]` scene features + user_id
- **Output**: `Dict` with attention weights, verbosity
- **Architecture**: User embedding + attention
- **Purpose**: User-specific adaptations

### Tier 4 Heads (Multi-Modal)

#### Sound Event Head
- **Input**: `[B, 256]` audio features
- **Output**: `[B, 15]` sound event probabilities
- **Architecture**: Transformer encoder
- **Classes**: 15 sound event types

#### Retrieval Heads
- **Input**: Multiple (global, region, patch, depth, OCR, audio, scene graph)
- **Output**: `Dict` with multi-vector embeddings
- **Architecture**: Multi-encoder + projection to common space (256D)
- **Purpose**: Similar scene retrieval

### Tier 5 Heads (Temporal)

#### Predictive Alert Head
- **Input**: `[B, 512]` scene + `[B, 256]` motion features
- **Output**: `Dict` with hazard predictions
- **Architecture**: Motion predictor + hazard classifier
- **Purpose**: Hazard anticipation

---

## Backbone Specifications

### ResNet50 (Stage A, Always Enabled)

**Architecture**:
- **Input**: `[B, 3, 224, 224]`
- **Output**: Multi-scale features `[c2, c3, c4, c5]`
  - c2: `[B, 256, 56, 56]`
  - c3: `[B, 512, 28, 28]`
  - c4: `[B, 1024, 14, 14]`
  - c5: `[B, 2048, 7, 7]`
- **Pretrained**: ImageNet
- **Parameters**: ~25M

### Simplified FPN

**Architecture**:
- **Input**: ResNet50 features `[c2, c3, c4, c5]`
- **Output**: FPN features `[p2, p3, p4, p5]`
  - All at 256 channels
  - Spatial sizes: `[56, 28, 14, 7]`
- **Parameters**: ~2M

### Hybrid CNN-ViT (T2+, Stage B)

**Architecture**:
- **CNN Branch**: ResNet50 features
- **ViT Branch**: Patch embeddings (14x14 patches, 768D)
- **Fusion**: Weighted combination
- **Output**: `[B, 256, H, W]` fused features
- **Parameters**: ~80M

### Temporal Encoder (T5, Stage B)

**Architecture**:
- **ConvLSTM**: Multi-layer temporal processing
- **TimeSformer**: Divided space-time attention
- **Input**: `[B, T, C, H, W]` temporal features
- **Output**: `[B, C, H, W]` + temporal outputs
- **Parameters**: ~50M

---

## Training Specifications

### Dataset

**COCO Dataset**:
- **Classes**: 91 (80 COCO + 11 accessibility)
- **Images**: ~118k training, ~5k validation
- **Format**: COCO JSON annotations
- **Preprocessing**: Resize to 224x224, ImageNet normalization

### Training Configuration

**Optimizer**:
- **Type**: AdamW
- **Learning Rate**: 0.001 (with cosine annealing)
- **Weight Decay**: 0.0001
- **Beta1**: 0.9
- **Beta2**: 0.999

**Scheduler**:
- **Type**: CosineAnnealingLR
- **T_max**: num_epochs
- **eta_min**: 1e-6

**Loss Functions**:
- **Detection**: Focal Loss (classification) + IoU Loss (boxes)
- **Distance**: CrossEntropyLoss
- **Urgency**: CrossEntropyLoss
- **Depth**: L1Loss + Uncertainty loss
- **Motion**: L1Loss
- **OCR**: CTC Loss
- **Scene Description**: CrossEntropyLoss (token-level)

**Gradient Balancing**:
- **Method**: GradNorm (optional)
- **Purpose**: Balance gradients across tasks

### Batch Sizes

**By Tier**:
- **T0-T1**: Batch size 32 (faster convergence)
- **T2-T3**: Batch size 16 (moderate training time)
- **T4-T5**: Batch size 8 (longer training time)

**Memory Considerations**:
- Reduce batch size if OOM errors occur
- Enable gradient checkpointing for memory savings
- Use mixed precision (FP16) for 2x memory reduction

---

## Hardware Requirements

### Training

**Cloud GPU (Required for all tiers)**:
- **Minimum**: NVIDIA T4 (16GB VRAM)
- **Recommended**: NVIDIA A100 (40GB VRAM) or V100 (32GB VRAM)
- **CUDA**: 11.8+ or 12.1+
- **cuDNN**: 8.0+

**Memory Requirements**:
- **T0**: ~500-700 MB VRAM (batch size 32)
- **T2**: ~3.0-4.0 GB VRAM (batch size 16)
- **T5**: ~6.0-8.0 GB VRAM (batch size 8)

### Inference

**Local Development**:
- **Apple Silicon**: M1+ (MPS support)
- **CPU**: Any (slower, for testing only)
- **Memory**: 8GB+ RAM

**Production Deployment**:
- **Mobile**: iOS device with Neural Engine (A12+)
- **Edge**: Jetson Nano/Xavier (NVIDIA)
- **Cloud**: Any CUDA-capable GPU

---

## Performance Targets

### Latency Targets

**Stage A (Safety-Critical)**:
- **Target**: <150ms per frame
- **T0-T1**: ~60-80ms
- **T2-T3**: ~100-120ms
- **T4-T5**: ~120-150ms

**Stage B (Context)**:
- **T1**: ~50ms
- **T2**: ~80ms
- **T3**: ~150ms
- **T4**: ~200ms
- **T5**: ~350ms

**Total Latency**:
- **T0**: ~60ms (Stage A only)
- **T1**: ~130ms
- **T2**: ~180ms
- **T3**: ~250ms
- **T4**: ~310ms
- **T5**: ~470ms

### Throughput Targets

**FPS (Frames Per Second)**:
- **T0**: 10-15 FPS
- **T1**: 6-8 FPS
- **T2**: 4-6 FPS
- **T3**: 3-4 FPS
- **T4**: 2-3 FPS
- **T5**: 1-2 FPS

### Accuracy Targets

**Object Detection**:
- **mAP@0.5**: >0.40 (T0), >0.45 (T2), >0.50 (T5)
- **Classification Accuracy**: >0.70 (T0), >0.80 (T2), >0.85 (T5)

**Distance Estimation**:
- **MAE**: <0.5m for near objects, <2.0m for far objects

**Urgency Classification**:
- **Accuracy**: >0.85 (critical for safety)

---

## Security Specifications

### Authentication

**HMAC Tokens**:
- **Algorithm**: HMAC-SHA256
- **Token Format**: `<payload_base64>.<signature_base64>`
- **Expiration**: Configurable (default: 3600 seconds)
- **Secret**: Environment variable `MAXSIGHT_SECRET_KEY`

### Input Validation

**File Upload**:
- **Max Size**: 10MB (configurable)
- **Allowed Types**: JPEG, PNG, GIF, BMP, WEBP, TIFF
- **Validation**: Magic number detection

**Base64**:
- **Validation**: Format validation before decoding
- **Injection Prevention**: Reject invalid Base64 strings

### Security Headers

**HTTP Headers**:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'`
- `Strict-Transport-Security` (if HTTPS)

### Rate Limiting

**Per-Session**: Configurable (default: 10 req/sec)
**Global**: Configurable (default: 100 req/sec)

---

## Export Specifications

### CoreML Export

**Format**: `.mlmodel`
**Input**: `[1, 3, 224, 224]` RGB image
**Output**: Dictionary with all task outputs
**Optimization**: Neural Engine compatible
**Size**: ~200-400 MB (depending on tier)

### ExecuTorch Export

**Format**: `.pte`
**Input**: `[1, 3, 224, 224]` RGB image
**Output**: Dictionary with all task outputs
**Optimization**: Mobile-optimized
**Size**: ~150-300 MB (depending on tier)

### Quantization

**INT8 Quantization**:
- **Method**: Post-training dynamic quantization
- **Size Reduction**: ~4x
- **Accuracy Drop**: <3% mAP
- **Speedup**: 2-4x inference speed

---

## Class Specifications

### Object Classes

**Total**: 91 classes
- **COCO Base**: 80 classes
- **Accessibility**: 11+ additional classes

**Categories**:
- Doors & entrances (30+ classes)
- Vertical navigation (stairs, elevators, ramps)
- Traffic & safety signs
- Information signs & labels
- Accessibility infrastructure
- Safety & emergency equipment
- Mobility aids
- Building features
- Furniture & seating

### Urgency Levels

**4 Levels**:
1. **safe**: No immediate risk
2. **caution**: Low risk, be aware
3. **warning**: Moderate risk, take care
4. **danger**: High risk, immediate action needed

### Distance Zones

**3 Zones**:
1. **near**: <2 meters
2. **medium**: 2-5 meters
3. **far**: >5 meters

---

## Environment Variables

**Required**:
- `MAXSIGHT_SECRET_KEY`: Authentication secret (production)

**Optional**:
- `MAXSIGHT_SESSION_TIMEOUT`: Token TTL (default: 3600)
- `MAXSIGHT_CORS_ORIGINS`: CORS allowed origins
- `REDIS_URL`: Redis connection URL
- `DEBUG`: Debug mode (0 or 1)
- `HTTPS`: HTTPS enabled (true/false)
- `CLOUD_GPU_URL`: Cloud GPU endpoint
- `CLOUD_CREDENTIALS_PATH`: Cloud credentials path

---

## Version Information

**Model Version**: 3.0
**PyTorch**: 2.5.0+
**Python**: 3.12+
**CUDA**: 11.8+ or 12.1+ (for training)
**MPS**: Supported (Apple Silicon)

---

## Summary

**Core Model**: MaxSightCNN (98M-320M parameters)
**Input**: RGB images `[B, 3, 224, 224]` + optional audio
**Output**: 30+ task outputs (detections, depth, motion, etc.)
**Tiers**: 6 tiers (T0-T5) with increasing capabilities
**Latency**: <150ms Stage A, <500ms Stage B
**Training**: Cloud GPU required (CUDA)
**Deployment**: iOS (CoreML), Edge (ExecuTorch), Cloud (PyTorch)

