# MaxSight Training Pipeline - Data Flow Diagram

## Overall Training Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Dataset     │  (MaxSightDataset)
│  - Images    │  [B, 3, 224, 224]
│  - Audio     │  [B, 128] (optional)
│  - Annotations│  boxes, labels, urgency, distance
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ DataLoader   │  Batch preparation
│  - Collate   │  Padding, normalization,
│              │  distance masks, num_objects
│  - Shuffle   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                 │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────────┐
       │                                                              │
       ▼                                                              ▼
┌─────────────────┐                                         ┌─────────────────┐
│  Forward Pass   │ (with autocast if use_mixed_precision)  │  Loss Function  │
│  (MaxSightCNN)  │                                         │  (MaxSightLoss) │
└─────────────────┘                                         └─────────────────┘
       │                                                              │
       │                                                              │
       ▼                                                              │
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────┘
       │
       ├─── Input: [B, 3, 224, 224] images
       │
       ▼
┌─────────────────┐
│  ResNet50       │  Backbone feature extraction
│  Backbone       │  └─> [B, 256, 56, 56]  (layer1)
│                 │  └─> [B, 512, 28, 28]  (layer2)
│                 │  └─> [B, 1024, 14, 14] (layer3)
│                 │  └─> [B, 2048, 7, 7]   (layer4)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SimplifiedFPN  │  Multi-scale feature pyramid
│                 │  └─> [B, 256, 56, 56]
│                 │  └─> [B, 256, 28, 28]
│                 │  └─> [B, 256, 14, 14]
│                 │  └─> [B, 256, 7, 7]
└────────┬────────┘
         │
         ├─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────┐                              ┌─────────────────┐
│ Detection Heads │                              │ Scene Heads     │
│                 │                              │                 │
│ - Classification│ [B, N, num_classes]         │ - Scene Embed  │ [B, 512]
│ - Bounding Box  │ [B, N, 4]                   │ - Urgency       │ [B, 4]
│ - Objectness    │ [B, N]                       │ - Distance      │ [B, N, 3]
│ - Text Regions  │ [B, N]                       │                 │
└────────┬────────┘                              └────────┬────────┘
         │                                                 │
         └─────────────────┬───────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL OUTPUTS                               │
│  {                                                                  │
│    'classifications': [B, N, num_classes],                         │
│    'boxes': [B, N, 4],                                             │
│    'objectness': [B, N],                                           │
│    'text_regions': [B, N],                                         │
│    'scene_embedding': [B, 512],                                    │
│    'urgency_scores': [B, 4],                                       │
│    'distance_zones': [B, N, 3]                                     │
│  }                                                                  │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                            │
│  (MaxSightLoss)                                                     │
└─────────────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Classification│ │ Localization│ │ Objectness   │
│ Loss (Focal) │ │ Loss (L1)   │ │ Loss (BCE)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Total Loss          │
              │  = cls + box + obj   │
              └──────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKWARD PASS                                │
└─────────────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Gradient     │ │ Gradient     │ │ Optimizer    │
│ Accumulation │ │ Clipping     │ │ Step         │
│              │ │ (max_norm=1) │ │ (AdamW)      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  EMA Update          │
              │  (Exponential Moving │
              │   Average)           │
              └──────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         VALIDATION LOOP                             │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────────────┐
              │  Apply EMA Weights         │
              │  (optional, if ema_enabled)│
              └──────────┬─────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Forward Pass         │
              │  (no gradients)       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────────────────────┐
              │  Metrics Computation                 │
              │  - Detection: mAP, recall@k, precision │
              │  - Scene: urgency accuracy, distance  │
              │    zone error                         │
              └──────────┬────────────────────────────┘
                         │
                         ▼
              ┌────────────────────────────┐
              │  Restore Original Weights  │
              │  (optional, if ema_enabled)│
              └──────────┬─────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINTING                               │
│  - Save best model (lowest val_loss)                               │
│  - Save final model                                                 │
│  - Save training history                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Flows

### 1. Data Loading Flow

```
COCO/Annotations
       │
       ▼
┌──────────────┐
│ generate_    │  Convert COCO format to MaxSight format
│ annotations  │  - Map categories
│              │  - Assign urgency scores
│              │  - Estimate distance zones
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MaxSight     │  Load images + annotations
│ Dataset      │  - Apply condition-specific augmentations
│              │  - Load audio (if available)
│              │  - Format targets
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ DataLoader   │  Batch creation
│              │  - Collate function
│              │  - Padding to max_objects
└──────┬───────┘
       │
       ▼
    Batch:
    {
      'images': [B, 3, 224, 224],
      'labels': [B, max_objects],
      'boxes': [B, max_objects, 4],
      'urgency': [B],
      'distance': [B, max_objects],
      'num_objects': [B]
    }
```

### 2. Model Forward Pass Flow

```
Input Image [B, 3, 224, 224]
       │
       ▼
┌─────────────────┐
│  ResNet50       │
│  conv1 + bn1    │ ──> [B, 64, 112, 112]
│  maxpool        │ ──> [B, 64, 56, 56]
│  layer1         │ ──> [B, 256, 56, 56]
│  layer2         │ ──> [B, 512, 28, 28]
│  layer3         │ ──> [B, 1024, 14, 14]
│  layer4         │ ──> [B, 2048, 7, 7]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SimplifiedFPN  │
│  Lateral Convs  │ ──> Channel reduction (1x1)
│  Top-down Path  │ ──> Feature fusion
│  Output Convs   │ ──> Smoothing (3x3)
└────────┬────────┘
         │
         ├──> [B, 256, 56, 56]
         ├──> [B, 256, 28, 28]
         ├──> [B, 256, 14, 14]
         └──> [B, 256, 7, 7]
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐
│ Detection Path  │              │ Scene Path      │
│                 │              │                 │
│ Fusion Conv     │              │ GAP + Proj      │
│ Detection Head  │              │ Scene Embedding │
│ └─> Classification│            │  (visual 256 +  │
│ └─> Box Head    │              │   audio 128)    │
│ └─> Objectness  │              │ Urgency Head    │
│ └─> Text Head   │              │ Distance Head   │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         │                        Audio Branch (optional)
         │                                 │
         └─────────────────┬───────────────┘
                           │
                           ▼
                    Model Outputs
```

### 3. Loss Computation Flow

```
Model Outputs                    Ground Truth
       │                              │
       ├── classifications [B,N,C]   ├── labels [B,N]
       ├── boxes [B,N,4]              ├── boxes [B,N,4]
       ├── objectness [B,N]           └── num_objects [B]
       └── ...
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Target Assignment                       │
│  (assign_targets_to_anchors)                         │
│  - Match predictions to GT using IoU                 │
│  - Create positive/negative masks                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Loss Components                        │
└─────────────────────────────────────────────────────┘
       │
       ├──> Classification Loss (Focal Loss)
       │    - Positive samples: focal loss on matched GT
       │    - Negative samples: focal loss on background
       │
       ├──> Localization Loss (L1 + IoU)
       │    - Only on positive samples
       │    - L1 distance on box coordinates
       │    - IoU loss for better alignment
       │
       └──> Objectness Loss (BCE)
            - Binary classification: object vs background
            - Positive = matched to GT
            - Negative = unmatched predictions
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Total Loss                              │
│  total_loss = cls_loss + box_loss + obj_loss        │
│  Averaged over valid images (with positives)         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              Backward Pass
```

### 4. Training Step Flow

```
Batch Input
    │
    ▼
┌─────────────────┐
│ Forward Pass    │ ──> Model Outputs
│ (with autocast) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Loss Computation│ ──> Loss Dict
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Backward Pass   │ ──> Gradients
│ (scaled)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gradient        │ ──> Accumulate gradients
│ Accumulation    │     (if steps > 1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gradient        │ ──> Clip to max_norm=1.0
│ Clipping        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optimizer Step  │ <── LR schedule flow (updated before step)
│ (AdamW)         │ ──> Update weights
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EMA Update      │ ──> Update shadow weights
│ (if enabled)    │
└────────┬────────┘
         │
         ▼
    Next Batch
```

### 5. Learning Rate Schedule Flow

```
Training Step
    │
    ▼
┌─────────────────┐
│ Check Step      │
│ vs warmup_steps │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Warmup │ │ Cosine Decay│
│ Phase  │ │ Phase       │
│        │ │             │
│ Linear │ │ Cosine      │
│ 0 -> 1 │ │ 1 -> 0      │
└───┬────┘ └──────┬───────┘
    │             │
    └──────┬──────┘
           │
           ▼
    Update LR for
    each param group
    (backbone: 0.1x, heads: 1.0x)
```

## Key Data Dimensions

### Input Dimensions
- **Images**: `[batch_size, 3, 224, 224]` (RGB, normalized)
- **Audio**: `[batch_size, 128]` (MFCC features, optional)
- **Labels**: `[batch_size, max_objects]` (class indices)
- **Boxes**: `[batch_size, max_objects, 4]` (center_x, center_y, w, h, normalized)

### Intermediate Dimensions
- **ResNet Features**: 
  - Layer1: `[B, 256, 56, 56]`
  - Layer2: `[B, 512, 28, 28]`
  - Layer3: `[B, 1024, 14, 14]`
  - Layer4: `[B, 2048, 7, 7]`
- **FPN Features**: `[B, 256, H, W]` at 4 scales
- **Scene Features**: `[B, 384]` (256 visual + 128 audio)

### Output Dimensions
- **Classifications**: `[batch_size, num_queries, num_classes]`
- **Boxes**: `[batch_size, num_queries, 4]`
- **Objectness**: `[batch_size, num_queries]`
- **Text Regions**: `[batch_size, num_queries]`
- **Scene Embedding**: `[batch_size, 512]`
- **Urgency Scores**: `[batch_size, 4]`
- **Distance Zones**: `[batch_size, num_queries, 3]`

## Condition-Specific Adaptations

```
Condition Mode
    │
    ├──> 'glaucoma' ──> Peripheral weight boost
    ├──> 'amd' ──> Center weight boost
    ├──> 'color_blindness' ──> Color prediction head
    ├──> 'cataracts' ──> Contrast enhancement
    ├──> 'diabetic_retinopathy' ──> Edge enhancement
    ├──> 'retinitis_pigmentosa' ──> Brightness boost
    └──> Others ──> Preprocessing augmentations
```

