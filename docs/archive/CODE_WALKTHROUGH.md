# MaxSight 3.0 - Complete Code Walkthrough

**Step-by-step explanation of every component and how they work together**

---

## Table of Contents

1. [Quick Start Code Walkthrough](#1-quick-start-code-walkthrough)
2. [Model Creation & Initialization](#2-model-creation--initialization)
3. [Forward Pass Flow](#3-forward-pass-flow)
4. [Stage A: Safety-Critical Pass](#4-stage-a-safety-critical-pass)
5. [Stage B: Context Pass](#5-stage-b-context-pass)
6. [Training Pipeline](#6-training-pipeline)
7. [Export & Deployment](#7-export--deployment)

---

## 1. Quick Start Code Walkthrough

### Step 1: Installation Commands

```bash
# Line 1: Clone repository
git clone <repository-url>
# What it does: Downloads the entire MaxSight codebase to your local machine
# Result: Creates '2026-Prototype' directory with all code

# Line 2: Navigate to project
cd 2026-Prototype
# What it does: Changes your current directory to the project root
# Result: You're now in the project directory

# Line 3: Create virtual environment
python3.12 -m venv venv
# What it does: Creates an isolated Python environment (Python 3.12)
# Why: Prevents dependency conflicts with other projects
# Result: Creates 'venv' directory with Python interpreter

# Line 4: Activate virtual environment
source venv/bin/activate
# What it does: Activates the virtual environment (macOS/Linux)
# Why: Ensures you use the project's Python and packages
# Result: Your shell prompt shows '(venv)' prefix

# Line 5: Upgrade pip
pip install --upgrade pip
# What it does: Updates pip to latest version
# Why: Ensures you can install latest packages
# Result: pip is now up-to-date

# Line 6: Install dependencies
pip install -r requirements.txt
# What it does: Reads requirements.txt and installs all listed packages
# Packages installed: torch, torchvision, numpy, PIL, etc.
# Result: All dependencies are now available in your venv

# Line 7: Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
# What it does: 
#   - Imports PyTorch library
#   - Prints PyTorch version number
#   - Checks if MPS (Apple Silicon GPU) is available
# Expected output: "PyTorch 2.5.1, MPS: True" (on Apple Silicon)
```

### Step 2: Basic Model Usage

```python
# File: ml/models/maxsight_cnn.py
# Function: create_model()

# Step 1: Import the model creation function
from ml.models.maxsight_cnn import create_model

# What this does:
#   - Imports the 'create_model' function from maxsight_cnn.py
#   - This function is a convenience wrapper around MaxSightCNN.__init__

# Step 2: Create a model instance
model = create_model()

# What this does (line by line inside create_model):
#   1. Checks if tier_config is provided (it's None by default)
#   2. Creates a TierConfig for T0_BASELINE_CNN (default tier)
#   3. Calls MaxSightCNN.__init__() with default parameters:
#      - num_classes=48 (COCO classes + accessibility classes)
#      - num_urgency_levels=4 (safe, caution, warning, danger)
#      - num_distance_zones=3 (near, medium, far)
#      - use_audio=False (audio disabled by default)
#      - fpn_channels=256 (FPN output channels)
#      - tier_config=TierConfig.for_tier(CapabilityTier.T0_BASELINE_CNN)

# Step 3: Set model to evaluation mode
model.eval()

# What this does:
#   - Disables dropout layers (if any)
#   - Sets BatchNorm to use running statistics (not batch statistics)
#   - Ensures consistent inference behavior
#   - Required before running inference

# Step 4: Create dummy input
dummy_image = torch.randn(2, 3, 224, 224)

# What this does:
#   - Creates a random tensor with shape [2, 3, 224, 224]
#   - 2 = batch size (2 images)
#   - 3 = channels (RGB)
#   - 224, 224 = height, width (standard ImageNet size)
#   - Values are random numbers from normal distribution (mean=0, std=1)

# Step 5: Run inference
with torch.no_grad():
    outputs = model(dummy_image)

# What this does:
#   - torch.no_grad(): Disables gradient computation (saves memory, faster)
#   - model(dummy_image): Calls model.forward(dummy_image)
#   - This triggers the complete forward pass (Stage A + Stage B)
#   - Returns a dictionary with 30+ outputs

# Step 6: Inspect outputs
for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.shape}")
    else:
        print(f"    {k}: {type(v).__name__}")

# What this does:
#   - Iterates through all outputs in the dictionary
#   - For tensors: prints the shape (e.g., "objectness: torch.Size([2, 3136])")
#   - For non-tensors: prints the type (e.g., "detections: list")
```

---

## 2. Model Creation & Initialization

### MaxSightCNN.__init__() - Line by Line

```python
# File: ml/models/maxsight_cnn.py
# Class: MaxSightCNN
# Method: __init__()

def __init__(
    self,
    num_classes: int = len(COCO_CLASSES),  # Default: 48 classes
    num_urgency_levels: int = 4,           # safe, caution, warning, danger
    num_distance_zones: int = 3,           # near, medium, far
    use_audio: bool = False,               # Audio fusion disabled by default
    condition_mode: Optional[str] = None,  # Vision condition (glaucoma, AMD, etc.)
    fpn_channels: int = 256,               # FPN output channels
    tier_config: Optional['TierConfig'] = None  # Tier configuration
):
```

**Step-by-step initialization:**

#### Step 1: Store Configuration
```python
self.num_classes = num_classes  # 48
self.num_urgency_levels = num_urgency_levels  # 4
self.num_distance_zones = num_distance_zones  # 3
self.use_audio = use_audio  # False
self.condition_mode = condition_mode  # None
self.fpn_channels = fpn_channels  # 256
```

**What this does:**
- Stores all configuration parameters as instance variables
- These are used throughout the forward pass

#### Step 2: Load ResNet50 Backbone (ImageNet Pretrained)
```python
# Line 384-386
try:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
except AttributeError:
    resnet = models.resnet50(pretrained=True)
```

**What this does:**
- Loads ResNet50 pretrained on ImageNet (classification task)
- `IMAGENET1K_V2` = ImageNet-1k version 2 weights
- Fallback to `pretrained=True` for older PyTorch versions
- **Result**: ResNet50 with learned features (edges, textures, object parts)

#### Step 3: Extract ResNet50 Layers
```python
# Lines 388-395
self.conv1 = resnet.conv1      # First 7x7 conv layer
self.bn1 = resnet.bn1          # Batch normalization
self.relu = resnet.relu        # ReLU activation
self.maxpool = resnet.maxpool  # Max pooling
self.layer1 = resnet.layer1    # ResNet block 1 (output: 256 channels)
self.layer2 = resnet.layer2    # ResNet block 2 (output: 512 channels)
self.layer3 = resnet.layer3    # ResNet block 3 (output: 1024 channels)
self.layer4 = resnet.layer4    # ResNet block 4 (output: 2048 channels)
```

**What this does:**
- Extracts individual layers from ResNet50
- These layers will be called sequentially in the forward pass
- **Output channels**: C2=256, C3=512, C4=1024, C5=2048

#### Step 4: Create FPN (Feature Pyramid Network)
```python
# Line 398
self.fpn = SimplifiedFPN([256, 512, 1024, 2048], fpn_channels)
```

**What this does:**
- Creates a Feature Pyramid Network
- **Input**: List of channel counts from ResNet layers [256, 512, 1024, 2048]
- **Output**: All FPN levels have `fpn_channels=256` channels
- **Purpose**: Creates multi-scale features (P2, P3, P4, P5) for detecting objects of different sizes
- **Note**: FPN is initialized from scratch (not COCO-pretrained)

**SimplifiedFPN architecture:**
```python
# Inside SimplifiedFPN.__init__()
self.lateral_convs = nn.ModuleList([
    # 1x1 convs to normalize channel counts
    nn.Sequential(
        nn.Conv2d(256, 256, 1),   # For C2
        nn.Conv2d(512, 256, 1),   # For C3
        nn.Conv2d(1024, 256, 1),  # For C4
        nn.Conv2d(2048, 256, 1),  # For C5
    )
])

self.fpn_convs = nn.ModuleList([
    # 3x3 convs to smooth features
    nn.Sequential(
        nn.Conv2d(256, 256, 3, padding=1),
        # ... (4 times, one for each level)
    )
])
```

**FPN Forward Pass:**
```python
# Top-down path with lateral connections
# P5 (highest level) → P4 → P3 → P2 (lowest level)
# Each level: lateral connection + top-down connection + 3x3 conv
```

#### Step 5: Create Detection Heads (Tier 1: Safety-Critical)
```python
# Lines 462-493
# Detection Fusion (combines P3, P4, P5)
self.detection_fusion = nn.Sequential(
    nn.Conv2d(fpn_channels * 3, fpn_channels, 1),  # Fuse 3 scales
    nn.BatchNorm2d(fpn_channels),
    nn.ReLU(inplace=False)
)

# Objectness Head (is there an object?)
self.obj_head = nn.Sequential(
    nn.Conv2d(fpn_channels, 256, 3, padding=1),
    nn.ReLU(inplace=False),
    nn.Conv2d(256, 1, 1)  # Output: [B, 1, H, W]
)

# Classification Head (what object?)
self.cls_head = nn.Sequential(
    nn.Conv2d(fpn_channels, 256, 3, padding=1),
    nn.ReLU(inplace=False),
    nn.Conv2d(256, num_classes, 1)  # Output: [B, num_classes, H, W]
)

# Box Regression Head (where is it?)
self.box_head = nn.Sequential(
    nn.Conv2d(fpn_channels, 256, 3, padding=1),
    nn.ReLU(inplace=False),
    nn.Conv2d(256, 4, 1)  # Output: [B, 4, H, W] (cx, cy, w, h)
)
```

**What this does:**
- Creates heads for object detection (Tier 1: safety-critical)
- All heads share the same FPN features
- **Objectness**: Binary classification (object vs. background)
- **Classification**: Multi-class classification (48 classes)
- **Box Regression**: Bounding box coordinates (center-x, center-y, width, height)

#### Step 6: Create Distance & Urgency Heads
```python
# Distance Zones Head (how far?)
self.distance_head = nn.Sequential(
    nn.Linear(fpn_channels, 128),
    nn.ReLU(),
    nn.Linear(128, num_distance_zones)  # Output: [B, 3] (near, medium, far)
)

# Urgency Head (how dangerous?)
self.urgency_head = nn.Sequential(
    nn.Linear(fpn_channels, 128),
    nn.ReLU(),
    nn.Linear(128, num_urgency_levels)  # Output: [B, 4] (safe, caution, warning, danger)
)
```

**What this does:**
- **Distance**: Predicts distance zone for each object
- **Urgency**: Predicts urgency level (safety-critical)

#### Step 7: Initialize Weights
```python
# Line 834
self._initialize_weights()
```

**What this does:**
- Initializes all layers (except pretrained ResNet50)
- Uses Kaiming initialization for conv layers
- Uses small random weights for linear layers
- Sets BatchNorm to identity (weight=1, bias=0)

---

## 3. Forward Pass Flow

### model.forward() - Complete Flow

```python
# File: ml/models/maxsight_cnn.py
# Method: forward()

def forward(
    self,
    images: torch.Tensor,                    # [B, 3, 224, 224]
    audio_features: Optional[torch.Tensor] = None  # [B, 128] (optional)
) -> Dict[str, Any]:
```

**Step-by-step execution:**

#### Step 1: Input Validation
```python
batch_size = images.shape[0]  # Get batch size
H, W = images.shape[2], images.shape[3]  # Get height, width
```

**What this does:**
- Extracts batch size and image dimensions
- Used for reshaping outputs later

#### Step 2: Stage A Backbone (ALWAYS ResNet50 + FPN)
```python
# Line 881-931: _forward_stage_a_backbone()
fpn_features, fused_features, scene_context = self._forward_stage_a_backbone(images)
```

**What this does:**
- **CRITICAL**: Always uses ResNet50 + FPN (safety guarantee)
- Never uses hybrid backbone or temporal processing
- Returns:
  - `fpn_features`: List of [P2, P3, P4, P5] features
  - `fused_features`: Combined P3+P4+P5 features for detection
  - `scene_context`: Scene-level context features

**Inside _forward_stage_a_backbone():**

```python
# Step 2a: ResNet50 Forward
x = self.conv1(images)      # [B, 3, 224, 224] → [B, 64, 112, 112]
x = self.bn1(x)
x = self.relu(x)
x = self.maxpool(x)        # [B, 64, 112, 112] → [B, 64, 56, 56]

c2 = self.layer1(x)        # [B, 64, 56, 56] → [B, 256, 56, 56]
c3 = self.layer2(c2)       # [B, 256, 56, 56] → [B, 512, 28, 28]
c4 = self.layer3(c3)       # [B, 512, 28, 28] → [B, 1024, 14, 14]
c5 = self.layer4(c4)       # [B, 1024, 14, 14] → [B, 2048, 7, 7]

# Step 2b: FPN Forward
p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
# Output shapes:
#   P2: [B, 256, 56, 56]   (highest resolution, small objects)
#   P3: [B, 256, 28, 28]
#   P4: [B, 256, 14, 14]
#   P5: [B, 256, 7, 7]     (lowest resolution, large objects)

# Step 2c: Fuse Features for Detection
p3_resized = F.interpolate(p3, size=p4.shape[2:])  # Resize to match P4
p5_resized = F.interpolate(p5, size=p4.shape[2:])  # Resize to match P4
fused_features = torch.cat([p3_resized, p4, p5_resized], dim=1)  # [B, 768, 14, 14]
fused_features = self.detection_fusion(fused_features)  # [B, 256, 14, 14]
```

#### Step 3: Stage A Heads (Tier 1: Safety-Critical)
```python
# Objectness (is there an object?)
objectness = self.obj_head(fused_features)  # [B, 1, 14, 14]
objectness = objectness.contiguous().reshape(batch_size, -1)  # [B, 196]

# Classification (what object?)
cls_logits = self.cls_head(fused_features)  # [B, 48, 14, 14]
cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, self.num_classes)  # [B, 196, 48]

# Box Regression (where is it?)
box_preds = self.box_head(fused_features)  # [B, 4, 14, 14]
box_preds = box_preds.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 4)  # [B, 196, 4]

# Distance Zones (how far?)
dist_input = scene_context  # [B, 256]
distances = self.distance_head(dist_input)  # [B, 3]

# Urgency (how dangerous?)
urgency_scores = self.urgency_head(scene_context)  # [B, 4]
```

**What this does:**
- All Tier 1 heads run on every frame
- **Objectness**: Binary classification per spatial location
- **Classification**: Multi-class classification per spatial location
- **Box Regression**: Bounding box coordinates per spatial location
- **Distance**: Scene-level distance zones
- **Urgency**: Scene-level urgency scores

#### Step 4: Decision Point (Skip Stage B?)
```python
# Check if we should skip Stage B
skip_stage_b = (
    stage_a_latency_ms > 200.0 or  # Too slow
    max_uncertainty > 0.7           # Too uncertain
)
```

**What this does:**
- **Safety guarantee**: If Stage A is too slow or uncertain, skip Stage B
- Ensures safety-critical predictions are never blocked
- Stage B is opportunistic (nice-to-have, not required)

#### Step 5: Stage B Backbone (If Not Skipped)
```python
if not skip_stage_b:
    stage_b_features, temporal_outputs = self._forward_stage_b_backbone(
        images, fused_features, temporal_mode=False
    )
```

**What this does:**
- Only runs if Stage A completed successfully
- Can use Hybrid CNN-ViT (T2+) or Temporal (T5+)
- Processes raw images (not Stage A features) for hybrid backbone

**Inside _forward_stage_b_backbone():**

```python
# If Hybrid Backbone Enabled (T2+)
if self.hybrid_backbone is not None:
    hybrid_features = self.hybrid_backbone(images)  # Process raw images
    # Fuse with Stage A features
    stage_b_features = self._stage_b_channel_adapter(hybrid_features) + fused_features

# If Temporal Enabled (T5+)
if self.temporal_encoder is not None:
    temporal_features = self.temporal_encoder(fused_features)
    stage_b_features = stage_b_features + temporal_features
```

#### Step 6: Stage B Heads (Tier 2 & Tier 3)
```python
if not skip_stage_b:
    # Tier 2: Navigation & Context
    motion = self.motion_head(stage_b_features)  # [B, 2, H, W]
    therapy_state = self.therapy_state_head(...)  # Dict with fatigue, depth, contrast
    
    # Tier 3: Enhancement & Therapy
    scene_graph = self.scene_graph_encoder(...)  # Spatial/semantic relations
    ocr_results = self.ocr_head(stage_b_features)  # Text detection
    scene_description = self.scene_description_head(...)  # Natural language
```

**What this does:**
- Tier 2 heads: Navigation and context (can be throttled)
- Tier 3 heads: Enhancement features (optional, background)

#### Step 7: Output Assembly
```python
outputs = {
    # Tier 1: Safety-Critical (always present)
    'objectness': objectness,
    'classifications': cls_logits,
    'boxes': box_preds,
    'distance_zones': distances,
    'urgency_scores': urgency_scores,
    'uncertainty': uncertainty,
    
    # Tier 2: Navigation & Context (if Stage B ran)
    'motion': motion if not skip_stage_b else None,
    'therapy_state': therapy_state if not skip_stage_b else None,
    
    # Tier 3: Enhancement (if Stage B ran)
    'scene_graph': scene_graph if not skip_stage_b else None,
    'ocr_results': ocr_results if not skip_stage_b else None,
    'scene_description': scene_description if not skip_stage_b else None,
    
    # Metadata
    'stage_a_completed': True,
    'stage_b_completed': not skip_stage_b,
    'skip_stage_b_reason': skip_reason if skip_stage_b else None
}
```

**What this does:**
- Assembles all outputs into a dictionary
- Tier 1 outputs are always present
- Tier 2/3 outputs are None if Stage B was skipped
- Includes metadata about execution

---

## 4. Stage A: Safety-Critical Pass

### Complete Stage A Flow

```python
def _forward_stage_a_backbone(self, images: torch.Tensor):
    """
    Stage A backbone: ALWAYS ResNet50 + FPN (safety guarantee).
    
    CRITICAL: This method is HARD-CODED to use ResNet50+FPN only.
    Hybrid backbone is NEVER used here - it's only available in Stage B.
    """
```

**Line-by-line execution:**

#### Line 893-902: ResNet50 Forward
```python
# ResNet50 forward (ALWAYS - no tier-dependent switching)
x = self.conv1(images)        # 7x7 conv, stride 2: [B,3,224,224] → [B,64,112,112]
x = self.bn1(x)               # BatchNorm: normalizes activations
x = self.relu(x)              # ReLU: non-linearity
x = self.maxpool(x)           # MaxPool: [B,64,112,112] → [B,64,56,56]

c2 = self.layer1(x)           # ResNet block 1: [B,64,56,56] → [B,256,56,56]
c3 = self.layer2(c2)          # ResNet block 2: [B,256,56,56] → [B,512,28,28]
c4 = self.layer3(c3)          # ResNet block 3: [B,512,28,28] → [B,1024,14,14]
c5 = self.layer4(c4)          # ResNet block 4: [B,1024,14,14] → [B,2048,7,7]
```

**What each layer does:**
- **conv1**: Initial feature extraction (edges, textures)
- **layer1-4**: Progressive feature abstraction (object parts → objects)
- **Output**: Multi-scale features at different resolutions

#### Line 905: FPN Forward
```python
p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
```

**What FPN does:**
1. **Lateral Connections**: 1x1 convs normalize channels to 256
2. **Top-Down Path**: High-level features flow down to low levels
3. **Feature Fusion**: Each level combines lateral + top-down features
4. **Smoothing**: 3x3 convs smooth the fused features

**Output shapes:**
- P2: [B, 256, 56, 56] - High resolution, small objects
- P3: [B, 256, 28, 28] - Medium resolution
- P4: [B, 256, 14, 14] - Low resolution, large objects
- P5: [B, 256, 7, 7] - Lowest resolution

#### Line 907-912: Optional Attention (T1+)
```python
if self.fpn_attention is not None:
    p2 = self.fpn_attention(p2)
    p3 = self.fpn_attention(p3)
    p4 = self.fpn_attention(p4)
    p5 = self.fpn_attention(p5)
```

**What this does:**
- Applies SE (Squeeze-and-Excitation) or CBAM attention
- **SE**: Channel-wise attention (which channels are important)
- **CBAM**: Channel + spatial attention (which channels + where)
- **Purpose**: Focus on important features, suppress noise

#### Line 914-920: Scene Context
```python
p2_pooled = self.gap(p2).flatten(1)  # Global Average Pooling: [B,256,56,56] → [B,256]
p3_pooled = self.gap(p3).flatten(1)  # [B,256,28,28] → [B,256]
p4_pooled = self.gap(p4).flatten(1)  # [B,256,14,14] → [B,256]
p5_pooled = self.gap(p5).flatten(1)  # [B,256,7,7] → [B,256]

scene_feats = torch.cat([p2_pooled, p3_pooled, p4_pooled, p5_pooled], dim=1)  # [B,1024]
scene_context = self.scene_proj(scene_feats)  # [B,256]
```

**What this does:**
- **Global Average Pooling**: Reduces spatial dimensions to 1x1
- **Concatenation**: Combines features from all FPN levels
- **Projection**: Linear layer reduces to 256 dimensions
- **Purpose**: Scene-level context for distance/urgency heads

#### Line 922-929: Fused Features for Detection
```python
p3_resized = F.interpolate(p3, size=p4.shape[2:], mode='bilinear')  # [B,256,28,28] → [B,256,14,14]
p5_resized = F.interpolate(p5, size=p4.shape[2:], mode='bilinear')  # [B,256,7,7] → [B,256,14,14]
p4 = p4.contiguous()  # Ensure contiguous memory

fused_features = torch.cat([p3_resized, p4, p5_resized], dim=1)  # [B,768,14,14]
fused_features = self.detection_fusion(fused_features)  # [B,256,14,14]
```

**What this does:**
- **Interpolation**: Resizes P3 and P5 to match P4's spatial size (14x14)
- **Concatenation**: Combines P3, P4, P5 features (768 channels total)
- **Fusion**: 1x1 conv reduces to 256 channels
- **Purpose**: Multi-scale features for detection heads

---

## 5. Stage B: Context Pass

### Stage B Backbone Flow

```python
def _forward_stage_b_backbone(
    self,
    images: torch.Tensor,              # Raw images [B, 3, 224, 224]
    stage_a_features: torch.Tensor,    # Stage A fused features [B, 256, 14, 14]
    temporal_mode: bool = False
):
```

**Step-by-step:**

#### Step 1: Hybrid CNN-ViT Backbone (T2+)
```python
if self.hybrid_backbone is not None:
    # Process raw images (not Stage A features)
    hybrid_features = self.hybrid_backbone(images)
    
    # Adapt channels and fuse with Stage A features
    hybrid_adapted = self._stage_b_channel_adapter(hybrid_features)
    stage_b_features = hybrid_adapted + fused_features  # Weighted fusion
```

**What Hybrid Backbone does:**
1. **CNN Path**: ResNet50 + FPN (same as Stage A)
2. **ViT Path**: Vision Transformer processes image patches
3. **Fusion**: Combines CNN and ViT features
4. **Output**: Enhanced features with global context

#### Step 2: Temporal Processing (T5+)
```python
if self.temporal_encoder is not None:
    # Process Stage A features temporally
    temporal_features = self.temporal_encoder(stage_a_features)
    stage_b_features = stage_b_features + temporal_features
```

**What Temporal Encoder does:**
1. **ConvLSTM**: Processes features across time (motion tracking)
2. **TimeSformer**: Long-range temporal dependencies
3. **Output**: Motion-aware features

---

## 6. Training Pipeline

### Complete Training Flow

```python
# File: scripts/train_maxsight.py

# Step 1: Load Dataset
dataset = MaxSightDataset(
    data_dir='datasets/coco',
    annotation_file='annotations.json'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Create Model
model = create_model(tier_config=TierConfig.for_tier(CapabilityTier.T2_HYBRID_VIT))
model.train()  # Enable training mode

# Step 3: Setup Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Step 4: Training Loop
for epoch in range(100):
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = compute_multi_head_loss(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**What each step does:**

#### Step 1: Dataset Loading
```python
dataset = MaxSightDataset(data_dir='datasets/coco')
```
- Loads COCO images and annotations
- Applies condition-specific augmentations
- Returns (image, target) pairs

#### Step 2: Model Creation
```python
model = create_model(tier_config=TierConfig.for_tier(CapabilityTier.T2_HYBRID_VIT))
```
- Creates T2 tier model (210M parameters)
- Includes Hybrid CNN-ViT backbone
- All heads enabled

#### Step 3: Optimizer Setup
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```
- AdamW optimizer (Adam with weight decay)
- Learning rate: 0.001
- Updates all model parameters

#### Step 4: Training Loop
```python
for epoch in range(100):  # 100 epochs
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Forward pass: model processes batch
        outputs = model(images)
        
        # Loss computation: compare predictions to targets
        loss = compute_multi_head_loss(outputs, targets)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
```

**What happens in each iteration:**
1. **Forward**: Model processes batch, produces predictions
2. **Loss**: Compare predictions to ground truth
3. **Backward**: Compute gradients (how to update weights)
4. **Step**: Update weights using gradients

---

## 7. Export & Deployment

### Model Export Flow

```python
# File: ml/training/export.py

# Step 1: Load Trained Model
model = create_model()
checkpoint = torch.load('checkpoints/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Step 2: Export to CoreML (iOS)
export_to_coreml(model, 'exports/model.mlpackage')

# Step 3: Export to ExecuTorch (Mobile)
export_to_executorch(model, 'exports/model.pte')

# Step 4: Export to ONNX (Cross-platform)
export_to_onnx(model, 'exports/model.onnx')
```

**What each export does:**

#### CoreML Export
```python
def export_to_coreml(model, output_path):
    # Convert PyTorch model to CoreML format
    # Optimized for Apple Silicon (M1, M2, M3)
    # Can run on iOS devices
```

#### ExecuTorch Export
```python
def export_to_executorch(model, output_path):
    # Convert to ExecuTorch format
    # Optimized for mobile inference
    # Supports quantization (INT8)
```

#### ONNX Export
```python
def export_to_onnx(model, output_path):
    # Convert to ONNX format
    # Cross-platform (iOS, Android, Web)
    # Can be loaded in other frameworks
```

---

## Summary

**Complete Flow:**
1. **Input**: Images [B, 3, 224, 224]
2. **Stage A**: ResNet50 + FPN → Tier 1 heads (safety-critical)
3. **Decision**: Skip Stage B if latency >200ms or uncertainty >0.7
4. **Stage B**: Hybrid CNN-ViT + Temporal → Tier 2/3 heads (context)
5. **Output**: Dictionary with 30+ task outputs

**Key Guarantees:**
- Stage A always uses ResNet50+FPN (safety)
- Stage B is opportunistic (can be skipped)
- Tier 1 heads always run (safety-critical)
- Tier 2/3 heads are optional (enhancement)

**Training:**
- ResNet50: Pretrained on ImageNet
- FPN: Trained from scratch on COCO
- All heads: Trained on COCO + accessibility data

---

**Next Steps:**
- See [FUNCTION_FLOW_ANALYSIS.md](FUNCTION_FLOW_ANALYSIS.md) for detailed flow diagrams
- See [DEVICE_SELECTION_POLICY.md](DEVICE_SELECTION_POLICY.md) for device selection
- See [MPS_COMPATIBILITY.md](MPS_COMPATIBILITY.md) for Apple Silicon development

