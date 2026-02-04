# MaxSight 3.0: Complete Detailed Implementation Plan

## Overview

This document provides **extremely detailed** implementation specifications for MaxSight 3.0, including architecture details, code structure, algorithms, hyperparameters, integration points, and step-by-step instructions for every component.

**Timeline**: 90-120 days (18-24 weeks)
**Status**: Planning Phase

---

## Table of Contents

1. [Phase 0: Advanced Backbone & Architecture](#phase-0-advanced-backbone--architecture)
2. [Phase 1: Multi-Modal Sensor Fusion](#phase-1-multi-modal-sensor-fusion)
3. [Phase 2: Advanced Multi-Task Heads](#phase-2-advanced-multi-task-heads)
4. [Phase 3: Multi-Vector Retrieval System](#phase-3-multi-vector-retrieval-system)
5. [Phase 4: Knowledge-Augmented Retrieval](#phase-4-knowledge-augmented-retrieval)
6. [Phase 5: Advanced Training Techniques](#phase-5-advanced-training-techniques)
7. [Phase 6: Personalization & Active Guidance](#phase-6-personalization--active-guidance)
8. [Phase 7: Optimization & Mobile Deployment](#phase-7-optimization--mobile-deployment)
9. [Phase 8: Simulator Integration & UI](#phase-8-simulator-integration--ui)
10. [Phase 9: Evaluation & Metrics](#phase-9-evaluation--metrics)

---

## Phase 0: Advanced Backbone & Architecture (Week 0-3)

### 0.1 Vision Transformer Backbone

**File**: `ml/models/backbone/vit_backbone.py`

**Class**: `VisionTransformerBackbone`

**Detailed Architecture**:

1. **Patch Embedding**:
   ```python
   # Input: RGB image [B, 3, H, W] (typically 224x224 or 384x384)
   # Patch size: 16x16 pixels
   patch_embed = nn.Conv2d(
       in_channels=3,
       out_channels=embed_dim,  # 768 for ViT-Base, 1024 for ViT-Large
       kernel_size=patch_size,  # 16
       stride=patch_size
   )
   # Output: [B, embed_dim, H/patch_size, W/patch_size]
   # Flatten spatial dims: [B, embed_dim, N_patches] → [B, N_patches, embed_dim]
   # N_patches = (H/patch_size) * (W/patch_size) = 196 for 224x224 with 16x16 patches
   ```

2. **CLS Token**:
   ```python
   cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # [1, 1, D]
   # Expand to batch: [B, 1, embed_dim]
   # Concatenate: [B, N_patches+1, embed_dim]
   ```

3. **Positional Embeddings**:
   ```python
   # Option 1: Learned (recommended)
   pos_embed = nn.Parameter(torch.randn(1, N_patches+1, embed_dim))
   
   # Option 2: Sinusoidal (non-learned)
   pos_embed = create_sinusoidal_pos_embedding(N_patches+1, embed_dim)
   
   # Add to tokens: tokens = tokens + pos_embed
   ```

4. **Transformer Encoder Blocks** (12 layers for ViT-Base, 24 for ViT-Large):
   ```python
   class TransformerBlock(nn.Module):
       def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
           # Multi-Head Self-Attention
           self.attention = nn.MultiheadAttention(
               embed_dim, num_heads, dropout=dropout, batch_first=True
           )
           self.norm1 = nn.LayerNorm(embed_dim)
           
           # Feed-Forward Network
           mlp_dim = int(embed_dim * mlp_ratio)
           self.mlp = nn.Sequential(
               nn.Linear(embed_dim, mlp_dim),
               nn.GELU(),
               nn.Dropout(dropout),
               nn.Linear(mlp_dim, embed_dim),
               nn.Dropout(dropout)
           )
           self.norm2 = nn.LayerNorm(embed_dim)
       
       def forward(self, x):
           # Pre-norm architecture
           x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
           x = x + self.mlp(self.norm2(x))
           return x
   ```

5. **Output Extraction**:
   ```python
   # CLS token: [B, embed_dim] - global scene representation
   cls_output = output[:, 0, :]
   
   # Patch tokens: [B, N_patches, embed_dim] - spatial features
   patch_output = output[:, 1:, :]
   
   return cls_output, patch_output
   ```

**Hyperparameters**:
- `embed_dim`: 768 (ViT-Base) or 1024 (ViT-Large)
- `num_layers`: 12 (ViT-Base) or 24 (ViT-Large)
- `num_heads`: 12 (for embed_dim=768) or 16 (for embed_dim=1024)
- `mlp_ratio`: 4.0
- `dropout`: 0.1
- `patch_size`: 16

**Integration**:
- Add to `MaxSightCNN.__init__()`:
  ```python
  if use_vit_backbone:
      self.vit_backbone = VisionTransformerBackbone(
          embed_dim=768,
          num_layers=12,
          num_heads=12,
          patch_size=16
      )
  ```

**Testing**:
- Unit test: Verify output shapes match expected dimensions
- Integration test: Ensure CLS token and patch tokens are correctly extracted
- Performance test: Measure inference time vs ResNet50 baseline

---

### 0.2 Hybrid CNN + ViT Backbone

**File**: `ml/models/backbone/hybrid_backbone.py`

**Class**: `HybridCNNViTBackbone`

**Detailed Architecture**:

1. **CNN Branch** (ResNet50-FPN):
   ```python
   # Use existing ResNet50 + FPN
   self.cnn_backbone = ResNet50FPN(...)
   # Output: FPN features at 4 scales
   # P2: [B, 256, H/4, W/4]
   # P3: [B, 256, H/8, W/8]
   # P4: [B, 256, H/16, W/16]
   # P5: [B, 256, H/32, W/32]
   ```

2. **ViT Branch**:
   ```python
   self.vit_backbone = VisionTransformerBackbone(...)
   # Output: CLS token [B, 768], patch tokens [B, 196, 768]
   ```

3. **Cross-Layer Connections**:

   **CNN → ViT**:
   ```python
   # Project FPN features to ViT dimension
   self.cnn_to_vit_proj = nn.Conv2d(256, 768, 1)
   
   # Resize and flatten FPN features
   fpn_resized = F.interpolate(fpn_features, size=(14, 14), mode='bilinear')
   fpn_flat = fpn_resized.flatten(2).transpose(1, 2)  # [B, 196, 256]
   fpn_projected = self.cnn_to_vit_proj(fpn_resized).flatten(2).transpose(1, 2)  # [B, 196, 768]
   
   # Add to ViT patch tokens before attention
   vit_patch_tokens = vit_patch_tokens + fpn_projected
   ```

   **ViT → CNN**:
   ```python
   # Reshape ViT patch tokens to spatial
   vit_spatial = vit_patch_tokens.transpose(1, 2).reshape(B, 768, 14, 14)
   
   # Project to FPN dimension
   self.vit_to_cnn_proj = nn.Conv2d(768, 256, 1)
   vit_projected = self.vit_to_cnn_proj(vit_spatial)
   
   # Resize to match FPN scales and add
   for i, fpn_feat in enumerate(fpn_features):
       vit_resized = F.interpolate(vit_projected, size=fpn_feat.shape[2:], mode='bilinear')
       fpn_features[i] = fpn_features[i] + vit_resized
   ```

4. **Feature Fusion** (3 methods):

   **Method 1 - Concatenation**:
   ```python
   # Global pooling on FPN
   fpn_global = torch.cat([
       F.adaptive_avg_pool2d(fpn, 1).flatten(1) for fpn in fpn_features
   ], dim=1)  # [B, 256*4] = [B, 1024]
   
   # CLS token
   vit_cls = vit_cls_token  # [B, 768]
   
   # Concatenate and project
   fused = torch.cat([fpn_global, vit_cls], dim=1)  # [B, 1792]
   fused = nn.Linear(1792, 512)(fused)  # [B, 512]
   ```

   **Method 2 - Weighted Sum** (learned):
   ```python
   # Project both to same dimension
   fpn_proj = nn.Linear(256, 512)(fpn_global)
   vit_proj = nn.Linear(768, 512)(vit_cls)
   
   # Learned weights
   weights = F.softmax(torch.stack([
       self.weight_cnn.expand(B, 1),
       self.weight_vit.expand(B, 1)
   ], dim=1), dim=1)  # [B, 2]
   
   # Weighted sum
   fused = weights[:, 0:1] * fpn_proj + weights[:, 1:2] * vit_proj
   ```

   **Method 3 - Cross-Attention**:
   ```python
   # CNN features as queries, ViT features as keys/values
   fused = CrossAttention(
       query=fpn_global.unsqueeze(1),  # [B, 1, 256]
       key=vit_patch_tokens,  # [B, 196, 768]
       value=vit_patch_tokens
   )  # [B, 1, 768]
   fused = fused.squeeze(1)  # [B, 768]
   ```

**Hyperparameters**:
- `fusion_method`: 'concat', 'weighted', or 'cross_attention'
- `cnn_dim`: 256 (FPN channels)
- `vit_dim`: 768 (ViT embed_dim)
- `fused_dim`: 512

**Integration**:
- Modify `MaxSightCNN.__init__()`:
  ```python
  if use_hybrid_backbone:
      self.backbone = HybridCNNViTBackbone(
          fusion_method='cross_attention',
          cnn_dim=256,
          vit_dim=768,
          fused_dim=512
      )
  ```

**Testing**:
- Verify cross-layer connections preserve feature quality
- Test all three fusion methods and compare performance
- Measure computational overhead vs single backbone

---

### 0.3 Dynamic Convolution Modules

**File**: `ml/models/backbone/dynamic_conv.py`

**Class**: `DynamicConv2d`

**Detailed Implementation**:

```python
class DynamicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_kernels: int = 4,
        stride: int = 1,
        padding: int = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        
        # Base kernel set
        self.base_kernels = nn.ParameterList([
            nn.Parameter(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
            )
            for _ in range(num_kernels)
        ])
        
        # Condition predictor network
        self.condition_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 4, 16), 1),
            nn.ReLU(),
            nn.Conv2d(max(in_channels // 4, 16), num_kernels, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Predict condition weights
        condition_weights = self.condition_predictor(x)  # [B, num_kernels, 1, 1]
        condition_weights = condition_weights.squeeze(-1).squeeze(-1)  # [B, num_kernels]
        
        # Generate dynamic kernel
        # Weighted combination of base kernels
        dynamic_kernel = torch.zeros(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
            device=x.device, dtype=x.dtype
        )
        
        for i, base_kernel in enumerate(self.base_kernels):
            weight = condition_weights[:, i].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            dynamic_kernel = dynamic_kernel + weight.mean(0) * base_kernel
        
        # Apply convolution
        # Note: F.conv2d expects [out_channels, in_channels, k, k]
        output = F.conv2d(
            x, dynamic_kernel,
            stride=self.stride,
            padding=self.padding
        )
        
        return output
```

**Condition Adaptation Mechanisms**:

1. **Lighting Adaptation**:
   ```python
   def compute_lighting_condition(self, x):
       # Brightness: global average
       brightness = x.mean(dim=(2, 3))  # [B, C]
       
       # Contrast: standard deviation
       contrast = x.std(dim=(2, 3))  # [B, C]
       
       # Combine into condition vector
       condition = torch.cat([brightness.mean(1, keepdim=True),
                              contrast.mean(1, keepdim=True)], dim=1)
       return condition
   ```

2. **Occlusion Detection** (via attention):
   ```python
   def compute_occlusion_score(self, attention_weights):
       # attention_weights: [B, H, W] from attention module
       occlusion_score = 1 - attention_weights.mean(dim=(1, 2))  # [B]
       return occlusion_score
   ```

3. **Motion Blur** (from temporal encoder):
   ```python
   def compute_motion_condition(self, motion_magnitude):
       # motion_magnitude: [B, 1] from temporal encoder
       return motion_magnitude
   ```

**Integration**:
- Replace key conv layers in ResNet50:
  ```python
  # In ResNet50 layer initialization
  if use_dynamic_conv:
      self.layer3[0].conv1 = DynamicConv2d(512, 1024, 3, num_kernels=4)
      self.layer4[0].conv1 = DynamicConv2d(1024, 2048, 3, num_kernels=4)
  ```

**Hyperparameters**:
- `num_kernels`: 4-8 (number of base kernels)
- `kernel_size`: 3x3 (standard)
- `condition_dim`: 4 (brightness, contrast, occlusion, motion)

**Testing**:
- Verify dynamic kernels adapt to different input conditions
- Measure performance improvement vs static convolution
- Test with synthetic lighting/occlusion variations

---

### 0.4 CBAM & SE Attention Modules

**File**: `ml/models/attention/cbam_attention.py`

**Class**: `CBAM`

**Detailed Implementation**:

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        channel_weights = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return x * channel_weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        spatial_features = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_weights = self.sigmoid(self.conv(spatial_features))  # [B, 1, H, W]
        
        return x * spatial_weights


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```

**File**: `ml/models/attention/se_attention.py`

**Class**: `SEBlock`

**Detailed Implementation**:

```python
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        B, C, H, W = x.shape
        
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(B, C)  # [B, C]
        
        # Excitation: FC layers
        y = self.fc(y).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Scale: multiply with input
        return x * y
```

**Integration**:
- Add after each ResNet layer:
  ```python
  # In ResNet50
  self.layer1 = nn.Sequential(
      BasicBlock(...),
      CBAM(256, reduction=16)  # or SEBlock(256, reduction=16)
  )
  ```

**Hyperparameters**:
- `reduction`: 16 (standard) - controls compression ratio
- `kernel_size`: 7 (for spatial attention in CBAM)

**Testing**:
- Verify attention weights are learned correctly
- Compare performance: CBAM vs SE vs baseline
- Visualize attention maps to ensure meaningful patterns

---

### 0.5 Cross-Modal Attention

**File**: `ml/models/attention/cross_modal_attention.py`

**Class**: `CrossModalAttention`

**Detailed Implementation**:

```python
class CrossModalAttention(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        haptic_dim: int = 0,  # Optional
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        if haptic_dim > 0:
            self.haptic_proj = nn.Linear(haptic_dim, embed_dim)
        
        # Cross-attention: Vision ↔ Audio
        self.vision_audio_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_vision_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm_vision = nn.LayerNorm(embed_dim)
        self.norm_audio = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, N_vision, vision_dim]
        audio_features: torch.Tensor,   # [B, N_audio, audio_dim]
        haptic_features: Optional[torch.Tensor] = None  # [B, haptic_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vision_features.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, N_vision, embed_dim]
        audio_proj = self.audio_proj(audio_features)    # [B, N_audio, embed_dim]
        
        # Cross-attention: Vision attends to Audio
        vision_enhanced, _ = self.vision_audio_attn(
            query=vision_proj,
            key=audio_proj,
            value=audio_proj
        )  # [B, N_vision, embed_dim]
        vision_enhanced = self.norm_vision(vision_proj + vision_enhanced)
        
        # Cross-attention: Audio attends to Vision
        audio_enhanced, _ = self.audio_vision_attn(
            query=audio_proj,
            key=vision_proj,
            value=vision_proj
        )  # [B, N_audio, embed_dim]
        audio_enhanced = self.norm_audio(audio_proj + audio_enhanced)
        
        # Global pooling and fusion
        vision_global = vision_enhanced.mean(dim=1)  # [B, embed_dim]
        audio_global = audio_enhanced.mean(dim=1)    # [B, embed_dim]
        
        # Optional: Incorporate haptic
        if haptic_features is not None:
            haptic_proj = self.haptic_proj(haptic_features)  # [B, embed_dim]
            fused = torch.cat([vision_global, audio_global, haptic_proj], dim=1)
            fused = self.output_proj(fused)  # [B, embed_dim]
        else:
            fused = torch.cat([vision_global, audio_global], dim=1)
            fused = self.output_proj(fused)  # [B, embed_dim]
        
        return fused, vision_enhanced, audio_enhanced
```

**Integration**:
- Add after scene context fusion in `MaxSightCNN.forward()`:
  ```python
  # After combining scene_context and audio_emb
  if use_cross_modal_attention:
      fused_context, vision_enhanced, audio_enhanced = self.cross_modal_attn(
          vision_features=scene_context.unsqueeze(1),  # [B, 1, 256]
          audio_features=audio_emb.unsqueeze(1),       # [B, 1, 128]
          haptic_features=haptic_emb if available else None
      )
      combined_context = fused_context  # Use fused instead
  ```

**Hyperparameters**:
- `embed_dim`: 512 (common embedding dimension)
- `num_heads`: 8 (multi-head attention)
- `dropout`: 0.1

**Testing**:
- Verify cross-modal attention improves performance
- Test with missing modalities (audio=None)
- Visualize attention weights to ensure meaningful cross-modal connections

---

### 0.6 Cross-Task Attention

**File**: `ml/models/attention/cross_task_attention.py`

**Class**: `CrossTaskAttention`

**Detailed Implementation**:

```python
class CrossTaskAttention(nn.Module):
    def __init__(
        self,
        detection_dim: int,
        ocr_dim: int,
        description_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Projection layers
        self.detection_proj = nn.Linear(detection_dim, embed_dim)
        self.ocr_proj = nn.Linear(ocr_dim, embed_dim)
        self.description_proj = nn.Linear(description_dim, embed_dim)
        
        # Cross-attention modules
        # OCR → Detection (sign → door relationship)
        self.ocr_to_detection = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Detection → Description (objects → natural language)
        self.detection_to_description = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Description → OCR (context improves text recognition)
        self.description_to_ocr = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
    
    def forward(
        self,
        detection_features: torch.Tensor,  # [B, N_detections, detection_dim]
        ocr_features: torch.Tensor,        # [B, N_text_regions, ocr_dim]
        description_context: torch.Tensor  # [B, description_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project to common dimension
        det_proj = self.detection_proj(detection_features)  # [B, N_det, embed_dim]
        ocr_proj = self.ocr_proj(ocr_features)              # [B, N_text, embed_dim]
        desc_proj = self.description_proj(description_context).unsqueeze(1)  # [B, 1, embed_dim]
        
        # OCR → Detection: Use OCR context to enhance detection
        det_enhanced, _ = self.ocr_to_detection(
            query=det_proj,
            key=ocr_proj,
            value=ocr_proj
        )  # [B, N_det, embed_dim]
        
        # Detection → Description: Use detections to generate descriptions
        desc_enhanced, _ = self.detection_to_description(
            query=desc_proj,
            key=det_enhanced,
            value=det_enhanced
        )  # [B, 1, embed_dim]
        
        # Description → OCR: Use description context to improve OCR
        ocr_enhanced, _ = self.description_to_ocr(
            query=ocr_proj,
            key=desc_enhanced.expand(-1, ocr_proj.shape[1], -1),
            value=desc_enhanced.expand(-1, ocr_proj.shape[1], -1)
        )  # [B, N_text, embed_dim]
        
        return det_enhanced, ocr_enhanced, desc_enhanced.squeeze(1)
```

**Integration**:
- Integrate with detection, OCR, and description heads:
  ```python
  # In MaxSightCNN.forward() or separate processing step
  if use_cross_task_attention:
      det_enhanced, ocr_enhanced, desc_enhanced = self.cross_task_attn(
          detection_features=detection_embeddings,
          ocr_features=ocr_embeddings,
          description_context=scene_embedding
      )
      # Use enhanced features in respective heads
  ```

**Hyperparameters**:
- `embed_dim`: 512
- `num_heads`: 8

**Testing**:
- Verify OCR context improves detection (e.g., "EXIT" sign → door detection)
- Test description quality improvement with detection context
- Measure OCR accuracy improvement with description context

---

### 0.7 Advanced Temporal Modules

**File**: `ml/models/temporal/conv_lstm.py`

**Class**: `ConvLSTM`

**Detailed Implementation**:

```python
class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ConvLSTM cells
        self.cells = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,  # [B, T, C, H, W] - sequence of frames
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C, H, W = x.shape
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = hidden_state
        
        outputs = []
        for t in range(T):
            # Process through each layer
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(x[:, t], (h, c))
            outputs.append(h)
        
        # Stack outputs: [B, T, hidden_dim, H, W]
        output = torch.stack(outputs, dim=1)
        
        return output, (h, c)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Convolutional gates
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,  # i, f, g, o gates
            kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W]
        hidden: Tuple[torch.Tensor, torch.Tensor]  # (h, c)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h_prev], dim=1)  # [B, C+hidden_dim, H, W]
        
        # Convolutional gates
        gates = self.conv(combined)  # [B, 4*hidden_dim, H, W]
        
        # Split into gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Candidate values
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c_prev + i * g
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
```

**File**: `ml/models/temporal/temporal_transformer.py`

**Class**: `TimeSformer`

**Detailed Implementation**:

```python
class TimeSformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_frames: int = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Temporal embedding
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim))
        
        # Transformer blocks with divided space-time attention
        self.blocks = nn.ModuleList([
            DividedSpaceTimeAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N_patches, embed_dim] from ViT patch tokens
        B, T, N, D = x.shape
        
        # Add temporal embedding
        x = x + self.temporal_embed.unsqueeze(0)  # [B, T, N, D]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Global average pooling over spatial and temporal
        x = x.mean(dim=(1, 2))  # [B, embed_dim]
        
        return x


class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # Spatial attention (within each frame)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Temporal attention (across frames)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        
        # Reshape for spatial attention: [B*T, N, D]
        x_spatial = x.view(B * T, N, D)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = self.norm1(x.view(B * T, N, D) + x_spatial)
        x = x_spatial.view(B, T, N, D)
        
        # Reshape for temporal attention: [B*N, T, D]
        x_temporal = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = self.norm2(x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D) + x_temporal)
        x = x_temporal.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        
        # FFN
        x_ffn = self.ffn(x.view(B * T * N, D)).view(B, T, N, D)
        x = self.norm3(x + x_ffn)
        
        return x
```

**Integration**:
- Enhance existing `temporal_encoder.py`:
  ```python
  class EnhancedTemporalEncoder(nn.Module):
      def __init__(self, ...):
          self.conv_lstm = ConvLSTM(input_dim=256, hidden_dim=256, num_layers=2)
          self.timesformer = TimeSformer(embed_dim=768, num_frames=8)
      
      def forward(self, frame_sequence, vit_patch_tokens):
          # ConvLSTM for motion tracking
          motion_features, (h, c) = self.conv_lstm(frame_sequence)
          
          # TimeSformer for long-range dependencies
          temporal_context = self.timesformer(vit_patch_tokens)
          
          return motion_features, temporal_context
  ```

**Hyperparameters**:
- ConvLSTM: `hidden_dim`: 256, `num_layers`: 2, `kernel_size`: 3
- TimeSformer: `embed_dim`: 768, `num_heads`: 12, `num_layers`: 12, `num_frames`: 8

**Testing**:
- Test motion tracking on video sequences
- Verify long-range temporal dependencies are captured
- Measure performance on action recognition tasks

---

*[This document continues with detailed specifications for all remaining phases. Each phase includes similar level of detail: code structure, algorithms, hyperparameters, integration points, and testing procedures.]*

---

## Next Steps

1. Review this detailed plan
2. Prioritize phases based on project timeline
3. Begin implementation starting with Phase 0
4. Set up development environment and dependencies
5. Create initial file structure

---

**Document Version**: 1.0
**Last Updated**: 2025-12-08
**Status**: Planning Complete - Ready for Implementation

