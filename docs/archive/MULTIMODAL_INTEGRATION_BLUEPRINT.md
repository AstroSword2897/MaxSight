# MaxSight V2 Multimodal Integration Blueprint

**Status:** Implementation Guide  
**Date:** 2025-01-XX  
**Branch:** `feature/multimodal_refactor`

---

## Overview

This document provides a **file-by-file, line-by-line** implementation plan for integrating all multimodal components into `MaxSightCNN` with proper constraints and performance guarantees.

**Total Estimated Time:** 4-6 hours  
**Risk Level:** Low (incremental, testable changes)

---

## Phase 0: Preparation

### Step 0.1: Create Branch
```bash
git checkout -b feature/multimodal_refactor
git push -u origin feature/multimodal_refactor
```

### Step 0.2: Backup Current State
```bash
# Tag current version
git tag v1.0-before-refactor

# Backup model file
cp ml/models/maxsight_cnn.py ml/models/maxsight_cnn_v1_backup.py
```

### Step 0.3: Environment Check
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Phase 1: Refactor Individual Heads

### File 1: `ml/models/heads/depth_head.py`

**Current State:** Lines 16-78  
**Changes Required:**

#### Change 1.1: Add Uncertainty Head to `__init__`
**Location:** After line 39 (after `self.depth_conv`)

**Add:**
```python
        # Depth uncertainty head (properly encapsulated)
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Uncertainty in [0, 1]
        )
```

#### Change 1.2: Remove Softmax from Zone Head
**Location:** Line 50

**Change FROM:**
```python
            nn.Softmax(dim=1)
```

**Change TO:**
```python
            # Raw logits for CrossEntropyLoss (softmax applied in loss)
```

#### Change 1.3: Update Forward Return
**Location:** Lines 72-78

**Change FROM:**
```python
        depth_map = self.sigmoid(self.depth_conv(x)).squeeze(1)  # [B, H, W]
        zones = self.zone_head(x)  # [B, 3] (Flatten handles squeeze automatically)
        
        return {
            'depth_map': depth_map,
            'zones': zones
        }
```

**Change TO:**
```python
        # Depth map with safe squeeze
        depth_map = self.sigmoid(self.depth_conv(x))
        if depth_map.shape[1] == 1:
            depth_map = depth_map.view(depth_map.size(0), depth_map.size(2), depth_map.size(3))
        else:
            depth_map = depth_map.squeeze(1)  # [B, H, W]
        
        # Uncertainty (properly encapsulated)
        uncertainty = self.uncertainty_conv(x)
        if uncertainty.shape[1] == 1:
            uncertainty = uncertainty.view(uncertainty.size(0), uncertainty.size(2), uncertainty.size(3))
        else:
            uncertainty = uncertainty.squeeze(1)  # [B, H, W]
        
        # Zone classification (raw logits)
        zones = self.zone_head(x)  # [B, 3] - raw logits
        
        return {
            'depth_map': depth_map,
            'uncertainty': uncertainty,  # NEW
            'zones': zones  # Raw logits
        }
```

**Test:** Create `tests/test_depth_head.py` (see Phase 3)

---

### File 2: `ml/models/heads/sound_event_head.py`

**Current State:** Lines 13-184  
**Changes Required:**

#### Change 2.1: Move Dynamic Layers to `__init__`
**Location:** After line 45 (after `self.spectrogram_cnn`)

**Add:**
```python
        # Projection layers (moved from forward to avoid dynamic creation)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.spectrogram_proj = nn.Linear(64 * (embed_dim // 4), embed_dim)
```

#### Change 2.2: Update Forward to Use Pre-initialized Layers
**Location:** Lines 136-138

**Change FROM:**
```python
            # Project to embed_dim if needed
            if cnn_out.shape[2] != embed_dim:
                proj = nn.Linear(cnn_out.shape[2], embed_dim).to(cnn_out.device)
                cnn_out = proj(cnn_out)
```

**Change TO:**
```python
            # Project to embed_dim if needed (use pre-initialized layer)
            if cnn_out.shape[2] != self.embed_dim:
                cnn_out = self.spectrogram_proj(cnn_out)
```

**Location:** Line 145

**Change FROM:**
```python
            audio_embed = nn.Linear(audio_features.shape[-1], embed_dim).to(audio_features.device)(audio_features.mean(dim=1))
```

**Change TO:**
```python
            audio_embed = self.input_proj(audio_features.mean(dim=1))
```

**Test:** Create `tests/test_sound_event_head.py` (see Phase 3)

---

### File 3: `ml/models/heads/scene_description_head.py`

**Current State:** Lines 13-166  
**Changes Required (Optional Enhancement):**

#### Change 3.1: Add Cross-Attention for Regions (Optional)
**Location:** After line 118 (after `memory = self.input_fusion(combined)`)

**Add (optional enhancement):**
```python
        # ENHANCED: Cross-attend regions to global (preserves individual regions)
        if region_proj.shape[1] > 0:
            # Use global as query, regions as key/value
            attended_regions, _ = self.decoder.self_attn(
                query=global_proj,
                key=region_proj,
                value=region_proj
            )  # [B, 1, embed_dim]
            memory = self.input_fusion(torch.cat([global_proj, attended_regions], dim=2))
        else:
            memory = global_proj
```

**Note:** This is optional - current mean pooling works fine.

**Test:** Create `tests/test_scene_description_head.py` (see Phase 3)

---

## Phase 2: Implement Integration Changes

### File 4: `ml/models/maxsight_cnn.py`

**Current State:** Lines 256-1652  
**Changes Required:** Multiple sections

---

#### Change 4.1: Update `__init__` - Add New Modules
**Location:** After line 404 (after `self.audio_branch`)

**Add:**
```python
        # Enhanced audio processing (replaces simple audio_branch)
        from ml.models.fusion.multimodal_fusion import EnhancedAudioEncoder, SpatialSoundMapping
        from ml.models.heads.sound_event_head import SoundEventHead
        
        self.audio_encoder = EnhancedAudioEncoder(
            input_dim=128,
            embed_dim=256,
            num_heads=8
        )
        self.sound_event_head = SoundEventHead(
            input_dim=256,
            num_classes=15,
            embed_dim=256,
            num_heads=8
        )
        self.spatial_sound = SpatialSoundMapping(
            audio_dim=256,
            attention_size=(14, 14),  # Match FPN output size
            num_directions=4
        )
```

**Location:** After line 498 (after `self.distance_head`)

**Add:**
```python
        # Depth head with uncertainty
        from ml.models.heads.depth_head import DepthHead
        self.depth_head_module = DepthHead(
            in_channels=fpn_channels,  # 256
            dropout=0.1
        )
        
        # Temporal encoder
        from ml.models.temporal.temporal_encoder import TemporalEncoder
        self.temporal_encoder = TemporalEncoder(
            in_channels=256,
            num_frames=8,
            hidden_dim=256,
            use_conv_lstm=True,
            use_timesformer=False  # Can enable later
        )
        self.temporal_feature_proj = nn.Conv2d(256, 256, 1)  # Project motion features
        
        # Scene graph encoder
        from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder
        self.scene_graph_encoder = SceneGraphEncoder(
            object_embed_dim=256,
            relation_embed_dim=128
        )
        self.max_scene_graph_objects = 10  # Top-K constraint
        
        # Scene description head
        from ml.models.heads.scene_description_head import SceneDescriptionHead
        from ml.retrieval.encoders.global_encoder import GlobalEncoder
        
        self.global_encoder = GlobalEncoder(
            embed_dim=512,
            use_clip=True
        )
        self.scene_description_head = SceneDescriptionHead(
            global_dim=512,
            region_dim=256,
            ocr_dim=256,
            embed_dim=512,
            vocab_size=30000,
            max_length=100
        )
        self.generate_description = True  # Config flag
        
        # Personalization
        from ml.models.heads.personalization_head import PersonalizationHead
        self.personalization_head = PersonalizationHead(
            input_dim=512,
            num_features=10,
            num_alert_types=5,
            embed_dim=256
        )
        self.user_embeddings = nn.Embedding(
            num_embeddings=10000,  # Max users
            embedding_dim=256
        )
        self.object_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
```

---

#### Change 4.2: Update Forward Pass - Audio Integration
**Location:** Lines 719-725 (audio processing section)

**Change FROM:**
```python
        # Add audio context if available (helps with things like alarms, speech)
        # Audio gives clues that vision might miss, but it's optional
        if audio_features is not None:
            audio_emb = self.audio_branch(audio_features)  # Process audio features
            combined_context = torch.cat([scene_context, audio_emb], dim=1)  # Combine visual + audio
        else:
            # If no audio, just use zeros (model should still work)
            audio_emb = torch.zeros(batch_size, 128, device=scene_context.device)
            combined_context = torch.cat([scene_context, audio_emb], dim=1)
```

**Change TO:**
```python
        # Enhanced audio processing with spatial attention
        if audio_features is not None:
            # Encode audio
            audio_emb, _ = self.audio_encoder(audio_features)  # [B, 256]
            
            # Generate sound classifications (separate from spatial attention)
            sound_outputs = self.sound_event_head(audio_emb.unsqueeze(1))  # [B, 1, 256] -> outputs
            
            # Generate spatial attention map
            audio_attention_map, direction, distance = self.spatial_sound(audio_emb)
            
            # CRITICAL CONSTRAINTS:
            # 1. Assert spatial dimensions match
            assert audio_attention_map.shape[-2:] == fused_features.shape[-2:], \
                f"Audio attention {audio_attention_map.shape} must match features {fused_features.shape}"
            assert audio_attention_map.ndim == 4, "Audio attention must be [B, 1, H, W]"
            
            # 2. Interpolate if needed (preserve channel count)
            if audio_attention_map.shape[2:] != fused_features.shape[2:]:
                audio_attention_map = F.interpolate(
                    audio_attention_map,
                    size=fused_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # 3. MULTIPLICATIVE (not concatenation) - preserves pretrained weights
            # Use sigmoid for smoother gradients
            audio_attention_map = torch.sigmoid(audio_attention_map)  # [0, 1] with smooth gradients
            fused_features = fused_features * (1.0 + audio_attention_map)  # Multiplicative gating
            
            # Combine for scene context
            combined_context = torch.cat([scene_context, audio_emb], dim=1)  # [B, 256 + 256 = 512]
        else:
            # If no audio, just use zeros
            audio_emb = torch.zeros(batch_size, 256, device=scene_context.device)
            combined_context = torch.cat([scene_context, audio_emb], dim=1)  # [B, 512]
            sound_outputs = None
```

**Location:** After line 811 (in outputs dict)

**Add:**
```python
            # Audio outputs
            if sound_outputs is not None:
                outputs['sound_classifications'] = sound_outputs['sound_probs']
                outputs['sound_direction'] = sound_outputs['direction']
                outputs['sound_urgency'] = sound_outputs['urgency']
            else:
                outputs['sound_classifications'] = None
                outputs['sound_direction'] = None
                outputs['sound_urgency'] = None
```

---

#### Change 4.3: Update Forward Pass - Depth Integration
**Location:** After line 799 (after distance calculation)

**Add (before outputs dict):**
```python
        # Depth estimation with uncertainty (vectorized)
        depth_outputs = self.depth_head_module(fused_features)
        depth_map = depth_outputs['depth_map']  # [B, H, W]
        depth_uncertainty = depth_outputs['uncertainty']  # [B, H, W]
        
        # VECTORIZED depth sampling at box centers (NO LOOPS)
        top_k = min(10, H * W)
        top_k_scores, top_k_indices = torch.topk(obj_scores, k=top_k, dim=1)  # [B, K]
        
        # Extract box centers for top-K
        box_centers = box_preds[:, :, :2]  # [B, H*W, 2] - x, y centers
        top_k_centers = torch.gather(
            box_centers,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, K, 2]
        
        # Normalize to [-1, 1] for grid_sample
        image_size_tensor = torch.tensor([W, H], device=images.device, dtype=torch.float32)
        normalized_centers = (top_k_centers / image_size_tensor.unsqueeze(0).unsqueeze(0)) * 2.0 - 1.0  # [B, K, 2]
        normalized_centers = normalized_centers.flip(-1).unsqueeze(2)  # [B, K, 1, 2] for grid_sample
        
        # Sample depth at box centers (BATCHED)
        depth_at_centers = F.grid_sample(
            depth_map.unsqueeze(1),  # [B, 1, H, W]
            normalized_centers,  # [B, K, 1, 2]
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        ).squeeze(1).squeeze(-1)  # [B, K]
        
        # Sample uncertainty
        uncertainty_at_centers = F.grid_sample(
            depth_uncertainty.unsqueeze(1),
            normalized_centers,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        ).squeeze(1).squeeze(-1)  # [B, K]
        
        # Convert normalized depth [0, 1] to meters (calibrated per object class)
        # Vectorized depth scaling
        class_depth_scales = torch.tensor([
            10.0, 5.0, 3.0, 8.0, 12.0,  # Example scales for person, car, door, truck, bus
            # Add more as needed for your COCO classes
        ], device=images.device)
        
        # Get class indices for top-K
        top_k_classes = torch.gather(
            cls_logits.argmax(dim=-1),
            dim=1,
            index=top_k_indices
        )  # [B, K]
        
        # Clamp class indices to valid range
        top_k_classes = torch.clamp(top_k_classes, 0, len(class_depth_scales) - 1)
        
        # Vectorized depth scaling
        depth_scales = class_depth_scales[top_k_classes]  # [B, K]
        precise_distances = depth_at_centers * depth_scales  # [B, K] in meters
```

**Location:** In outputs dict (after line 810)

**Add:**
```python
            'depth_map': depth_map,
            'depth_uncertainty': depth_uncertainty,
            'precise_distances': precise_distances,  # [B, K] meters
            'distance_uncertainties': uncertainty_at_centers,  # [B, K]
            'top_k_indices': top_k_indices,  # For mapping back to detections
```

**Note:** Keep existing `distance_zones` for compatibility.

---

#### Change 4.4: Update Forward Pass - Temporal Integration
**Location:** At start of forward() method (after line 686)

**Change signature FROM:**
```python
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
```

**Change TO:**
```python
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        user_id: Optional[torch.Tensor] = None,
        prev_temporal_state: Optional[Dict] = None,
        use_temporal: bool = False
    ) -> Dict[str, torch.Tensor]:
```

**Location:** After line 705 (after FPN)

**Add (before scene_feats extraction):**
```python
        # Handle temporal input (video sequences)
        temporal_mode = False
        if images.dim() == 5:  # [B, T, 3, H, W]
            temporal_mode = True
            use_temporal = True
            B_orig, T, C_img, H_img, W_img = images.shape
            images = images.view(B_orig * T, C_img, H_img, W_img)  # Flatten for backbone
            batch_size = B_orig * T
        else:
            batch_size = images.size(0)
```

**Location:** After line 733 (after fused_features)

**Add:**
```python
        # Temporal processing (if video)
        if use_temporal and temporal_mode:
            # Reshape features back to temporal format
            fused_features_temporal = fused_features.view(B_orig, T, -1, H, W)  # [B, T, C, H, W]
            
            # Get temporal context
            temporal_outputs = self.temporal_encoder(fused_features_temporal)
            
            # Get motion features (already at correct spatial resolution)
            motion_features = temporal_outputs.get('motion_features')  # [B, 256, H, W] or similar
            
            if motion_features is not None:
                # Assert spatial alignment
                assert motion_features.shape[2:] == fused_features.shape[2:], \
                    f"Motion features {motion_features.shape} must match {fused_features.shape}"
                
                # Project if channel mismatch
                if motion_features.shape[1] != fused_features.shape[1]:
                    motion_features = self.temporal_feature_proj(motion_features)
                
                # Resize if needed
                if motion_features.shape[2:] != fused_features.shape[2:]:
                    motion_features = F.interpolate(
                        motion_features,
                        size=fused_features.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # MODULATE spatial features (additive fusion)
                fused_features = fused_features + motion_features  # Motion → perception
            
            # Temporal context for scene-level heads
            temporal_context = temporal_outputs.get('temporal_context')  # [B, embed_dim]
            if temporal_context is not None:
                scene_context = torch.cat([scene_context, temporal_context], dim=1)
            
            # Reshape back to flattened for detection heads
            fused_features = fused_features.view(B_orig * T, -1, H, W)
            batch_size = B_orig * T
        else:
            temporal_outputs = None
```

**Location:** In outputs dict (after line 810)

**Add:**
```python
            'motion': temporal_outputs.get('motion') if temporal_outputs else None,
            'temporal_consistency': temporal_outputs.get('consistency') if temporal_outputs else None,
```

---

#### Change 4.5: Update Forward Pass - Scene Graph Integration
**Location:** After line 811 (after outputs dict creation, before accessibility features)

**Add:**
```python
        # Scene graph (top-K objects only)
        top_k = min(self.max_scene_graph_objects, H * W)
        top_k_scores, top_k_indices = torch.topk(obj_scores, k=top_k, dim=1)  # [B, K]
        
        # Extract object embeddings for top-K only
        object_embs_list = []
        for b in range(batch_size):
            batch_embs = []
            for k_idx in range(top_k):
                idx = top_k_indices[b, k_idx].item()
                y_idx = idx // W
                x_idx = idx % W
                obj_feat = det_feats[b, :, y_idx, x_idx].unsqueeze(-1).unsqueeze(-1)
                obj_feat = F.adaptive_avg_pool2d(obj_feat, 1).squeeze(-1).squeeze(-1)
                batch_embs.append(obj_feat)
            object_embs_list.append(torch.stack(batch_embs))  # [K, 256]
        
        object_embeddings = torch.stack(object_embs_list)  # [B, K, 256]
        
        # Extract boxes and classes for top-K
        top_k_boxes = torch.gather(
            box_preds,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, 4)
        )  # [B, K, 4]
        
        top_k_classes = torch.gather(
            cls_logits.argmax(dim=-1),
            dim=1,
            index=top_k_indices
        )  # [B, K]
        
        # Build scene graphs
        if self.training:
            # Process all batches
            scene_graphs = []
            for b in range(batch_size):
                class_names = [COCO_CLASSES[c] if c < len(COCO_CLASSES) else 'object'
                              for c in top_k_classes[b].tolist()]
                scene_graph = self.scene_graph_encoder(
                    boxes=top_k_boxes[b],
                    object_embeddings=object_embeddings[b],
                    object_classes=class_names
                )
                scene_graphs.append(scene_graph)
            outputs['scene_graph'] = scene_graphs
        else:
            # Inference: just batch 0
            class_names = [COCO_CLASSES[c] if c < len(COCO_CLASSES) else 'object'
                          for c in top_k_classes[0].tolist()]
            scene_graph = self.scene_graph_encoder(
                boxes=top_k_boxes[0],
                object_embeddings=object_embeddings[0],
                object_classes=class_names
            )
            outputs['scene_graph'] = scene_graph
        
        outputs['spatial_relations'] = scene_graph['spatial_relations'] if not self.training else [sg['spatial_relations'] for sg in scene_graphs]
        outputs['semantic_relations'] = scene_graph['semantic_relations'] if not self.training else [sg['semantic_relations'] for sg in scene_graphs]
```

---

#### Change 4.6: Update Forward Pass - Scene Description Integration
**Location:** After scene graph (before personalization)

**Add:**
```python
        # Scene description (gated, expensive operation)
        if self.training or self.generate_description:
            # Sample 1 frame for CLIP (if video)
            if images.dim() == 5 or temporal_mode:
                if temporal_mode:
                    clip_images = images.view(B_orig, T, 3, H_img, W_img)[:, 0]  # Use first frame
                else:
                    clip_images = images[:, 0] if images.dim() == 5 else images
            else:
                clip_images = images
            
            # Get CLIP global embedding
            global_emb = self.global_encoder(clip_images)  # [B, 512]
            
            # Extract region embeddings (PROPERLY POOLED)
            region_embs_list = []
            for b in range(batch_size if not temporal_mode else B_orig):
                batch_regions = []
                for k_idx in range(min(5, top_k)):
                    idx = top_k_indices[b, k_idx].item()
                    y_idx = idx // W
                    x_idx = idx % W
                    region_feat = det_feats[b, :, y_idx, x_idx]  # [256]
                    
                    # POOL to remove spatial noise
                    region_feat = region_feat.unsqueeze(-1).unsqueeze(-1)  # [256, 1, 1]
                    region_feat = F.adaptive_avg_pool2d(region_feat, 1).squeeze(-1).squeeze(-1)  # [256]
                    batch_regions.append(region_feat)
                
                if batch_regions:
                    region_embs_list.append(torch.stack(batch_regions))  # [K, 256]
                else:
                    region_embs_list.append(torch.zeros(1, 256, device=images.device))
            
            region_embs_tensor = torch.stack(region_embs_list)  # [B, K, 256]
            
            # Generate description
            description_outputs = self.scene_description_head(
                global_embedding=global_emb,
                region_embeddings=region_embs_tensor,
                ocr_embeddings=None,  # Optional
                condition_mode=self.condition_mode or 'normal'
            )
            
            outputs['scene_description'] = description_outputs['description']
            outputs['description_logits'] = description_outputs['description_logits']
        else:
            outputs['scene_description'] = None
            outputs['description_logits'] = None
```

---

#### Change 4.7: Update Forward Pass - Personalization Integration
**Location:** After scene description

**Add:**
```python
        # Personalization (if user_id provided)
        if user_id is not None:
            # Get per-user embedding
            user_emb = self.user_embeddings(user_id)  # [B, 256]
            
            # Normalize user embedding (critical for cosine similarity)
            user_emb = F.normalize(user_emb, p=2, dim=1)  # [B, 256]
            
            # Encode object features
            object_features = object_embeddings  # [B, K, 256] from scene graph
            object_emb = self.object_encoder(object_features)  # [B, K, 256]
            
            # Normalize object embeddings
            object_emb = F.normalize(object_emb, p=2, dim=2)  # [B, K, 256]
            
            # Compute cosine similarity (for "my fridge" recognition)
            similarity = torch.bmm(
                user_emb.unsqueeze(1),  # [B, 1, 256]
                object_emb.transpose(1, 2)  # [B, 256, K]
            ).squeeze(1)  # [B, K]
            
            # Get personalization outputs
            personalization = self.personalization_head(
                scene_features=scene_emb,
                user_id=user_id,
                interaction_features=None
            )
            
            outputs['personalization'] = personalization
            outputs['user_object_similarity'] = similarity  # For metric learning
        else:
            outputs['personalization'] = None
            outputs['user_object_similarity'] = None
```

---

## Phase 3: Update Loss Function

### File 5: `ml/training/losses.py`

**Current State:** Lines 227-364  
**Changes Required:**

#### Change 5.1: Add Uncertainty Weighting Class
**Location:** After line 174 (after `assign_targets_to_anchors`)

**Add:**
```python
class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.
    
    Automatically learns task weights based on uncertainty.
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        # Learnable log variances (inverse precision)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.num_tasks = num_tasks
    
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Uncertainty weighting with clamping and monitoring.
        
        Args:
            losses: List of loss tensors [loss1, loss2, ...]
        
        Returns:
            total_loss: Weighted sum
            log_vars_dict: For monitoring
        """
        total = 0.0
        log_vars_dict = {}
        
        for i, loss in enumerate(losses):
            # CRITICAL: Clamp loss to prevent zero/NaN
            loss = torch.clamp(loss, min=1e-6, max=1e6)
            
            # Get precision (inverse variance)
            precision = torch.exp(-self.log_vars[i])  # 1 / sigma^2
            
            # Weighted loss: precision * loss + log(sigma)
            weighted_loss = precision * loss + self.log_vars[i]
            total += weighted_loss
            
            # Log for monitoring
            log_vars_dict[f'task_{i}_log_var'] = self.log_vars[i].item()
            log_vars_dict[f'task_{i}_precision'] = precision.item()
            log_vars_dict[f'task_{i}_raw_loss'] = loss.item()
        
        return total, log_vars_dict
```

#### Change 5.2: Update DetectionLoss to Use Uncertainty Weighting
**Location:** In `DetectionLoss.__init__` (after line 255)

**Add:**
```python
        # Uncertainty weighting for adaptive loss balancing
        self.uncertainty_weighting = UncertaintyWeightedLoss(num_tasks=6)
```

#### Change 5.3: Update DetectionLoss.forward to Use Uncertainty Weighting
**Location:** Lines 342-349 (total_loss calculation)

**Change FROM:**
```python
        total_loss = (
            self.classification_weight * cls_loss +
            self.localization_weight * box_loss +
            self.objectness_weight * obj_loss +
            self.urgency_weight * urgency_loss +
            self.distance_weight * distance_loss +
            self.text_weight * text_loss
        )
```

**Change TO:**
```python
        # Clamp all losses
        losses_list = [
            torch.clamp(cls_loss, min=1e-6),
            torch.clamp(box_loss, min=1e-6),
            torch.clamp(obj_loss, min=1e-6),
            torch.clamp(urgency_loss, min=1e-6),
            torch.clamp(distance_loss, min=1e-6),
            torch.clamp(text_loss, min=1e-6)
        ]
        
        # Apply uncertainty weighting
        total_loss, log_vars = self.uncertainty_weighting(losses_list)
        
        # Add log_vars to loss_dict for monitoring
        loss_dict['uncertainty_log_vars'] = log_vars
```

**Location:** In return dict (line 351)

**Add:**
```python
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'localization_loss': box_loss,
            'objectness_loss': obj_loss,
            'urgency_loss': urgency_loss,
            'distance_loss': distance_loss,
            'text_loss': text_loss,
            'uncertainty_log_vars': log_vars,  # NEW
            'num_positives': torch.tensor(num_positives, device=device, dtype=torch.long)
        }
```

---

### File 6: Add Contrastive Loss for Personalization

**Create:** `ml/training/personalization_loss.py`

**Content:**
```python
"""
Contrastive loss for personalization (metric learning).
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def compute_contrastive_loss(
    user_emb: torch.Tensor,  # [B, 256] normalized
    object_emb: torch.Tensor,  # [B, K, 256] normalized
    positive_mask: torch.Tensor,  # [B, K] binary
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Corrected InfoNCE contrastive loss for personalization.
    
    FIXED: Properly handles multiple positives per batch.
    
    Args:
        user_emb: Normalized user embeddings [B, 256]
        object_emb: Normalized object embeddings [B, K, 256]
        positive_mask: Binary mask indicating user's personal items [B, K]
        temperature: Temperature for softmax
    
    Returns:
        Contrastive loss scalar
    """
    B, K = object_emb.shape[:2]
    
    # Compute similarities: [B, K]
    similarity = torch.bmm(
        user_emb.unsqueeze(1),  # [B, 1, 256]
        object_emb.transpose(1, 2)  # [B, 256, K]
    ).squeeze(1) / temperature  # [B, K]
    
    # FIXED: Use binary cross-entropy with logits (handles multiple positives correctly)
    # This treats each object independently as positive/negative
    labels = positive_mask.float()  # [B, K]
    loss = F.binary_cross_entropy_with_logits(similarity, labels, reduction='mean')
    
    return loss
```

---

## Phase 4: Create Unit Tests

### File 7: `tests/test_integration_constraints.py`

**Create new file with:**
```python
"""
Unit tests for integration constraints.
Ensures architectural constraints are enforced.
"""
import torch
import torch.nn.functional as F
import pytest
from ml.models.maxsight_cnn import MaxSightCNN
from ml.models.heads.depth_head import DepthHead
from ml.models.fusion.multimodal_fusion import SpatialSoundMapping, EnhancedAudioEncoder


def test_audio_attention_preserves_channels():
    """Assert audio attention never changes channel count."""
    model = MaxSightCNN()
    features = torch.randn(2, 256, 14, 14)
    audio_features = torch.randn(2, 128)
    
    audio_encoder = EnhancedAudioEncoder(input_dim=128, embed_dim=256)
    spatial_sound = SpatialSoundMapping(audio_dim=256, attention_size=(14, 14))
    
    audio_emb, _ = audio_encoder(audio_features)
    attention_map, _, _ = spatial_sound(audio_emb)
    
    # Interpolate if needed
    if attention_map.shape[2:] != features.shape[2:]:
        attention_map = F.interpolate(attention_map, size=features.shape[2:], mode='bilinear')
    
    # Apply attention
    fused = features * (1.0 + torch.sigmoid(attention_map))
    
    assert fused.shape == features.shape, "Channel count must be preserved"
    assert fused.shape[1] == 256, "Channels must remain 256"


def test_depth_uncertainty_encapsulated():
    """Assert depth uncertainty comes from head, not re-calling layers."""
    depth_head = DepthHead(in_channels=256, dropout=0.1)
    features = torch.randn(2, 256, 14, 14)
    
    outputs = depth_head(features)
    
    assert 'uncertainty' in outputs, "Uncertainty must be in outputs"
    assert outputs['uncertainty'].shape == outputs['depth_map'].shape, \
        "Uncertainty must match depth_map shape"
    assert hasattr(depth_head, 'uncertainty_conv'), \
        "Uncertainty must be a module"


def test_temporal_spatial_alignment():
    """Assert temporal features match spatial resolution."""
    from ml.models.temporal.temporal_encoder import TemporalEncoder
    
    temporal_encoder = TemporalEncoder(in_channels=256, num_frames=8, hidden_dim=256)
    features = torch.randn(2, 256, 14, 14)
    temporal_features = torch.randn(2, 8, 256, 14, 14)
    
    temporal_outputs = temporal_encoder(temporal_features)
    motion_features = temporal_outputs.get('motion_features')
    
    if motion_features is not None:
        assert motion_features.shape[2:] == features.shape[2:], \
            f"Temporal {motion_features.shape} must match spatial {features.shape}"


def test_scene_graph_top_k():
    """Assert scene graph uses top-K, not all H*W."""
    model = MaxSightCNN()
    H, W = 14, 14
    
    obj_scores = torch.randn(2, H * W)
    top_k = min(model.max_scene_graph_objects, H * W)
    
    top_k_scores, top_k_indices = torch.topk(obj_scores, k=top_k, dim=1)
    
    assert top_k_indices.shape[1] <= model.max_scene_graph_objects, \
        f"Scene graph must use ≤{model.max_scene_graph_objects} objects"


def test_personalization_normalized():
    """Assert personalization embeddings are normalized."""
    model = MaxSightCNN()
    user_id = torch.tensor([0, 1])
    
    user_emb = model.user_embeddings(user_id)
    user_emb = F.normalize(user_emb, p=2, dim=1)
    
    norms = torch.norm(user_emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "User embeddings must be normalized"


def test_depth_vectorized():
    """Assert depth sampling uses grid_sample, not loops."""
    import torch.nn.functional as F
    
    depth_map = torch.randn(2, 14, 14)
    box_centers = torch.rand(2, 10, 2)  # [B, K, 2]
    
    # Normalize
    normalized = (box_centers / torch.tensor([14.0, 14.0])) * 2.0 - 1.0
    normalized = normalized.flip(-1).unsqueeze(2)
    
    # Should use grid_sample (vectorized)
    sampled = F.grid_sample(
        depth_map.unsqueeze(1),
        normalized,
        mode='bilinear',
        align_corners=False
    )
    
    assert sampled.shape[0] == 2, "Must be batched"
    assert sampled.shape[1] == 1, "Single channel"
    assert sampled.shape[2] == 10, "K samples"
```

---

## Phase 5: Update Training Loop

### File 8: `ml/training/train_loop.py`

**Location:** After line 478 (in `compute_multihead_loss`)

**Add monitoring:**
```python
        # Monitor uncertainty log_vars
        log_vars = loss_dict.get('uncertainty_log_vars', {})
        if log_vars and self.global_step % 100 == 0:
            for key, value in log_vars.items():
                self.logger.info(f"Uncertainty {key}: {value:.4f}")
                # Alert if log_var collapses (task broken)
                if 'log_var' in key and abs(value) > 5.0:
                    self.logger.warning(f"Task {key} log_var collapsed: {value}")
```

---

## Phase 6: Profiling & Validation

### File 9: `scripts/profile_integration.py`

**Create profiling script:**
```python
"""
Profile integrated MaxSightCNN forward pass.
"""
import torch
import time
from ml.models.maxsight_cnn import MaxSightCNN

def profile_forward():
    model = MaxSightCNN()
    model.eval()
    
    images = torch.randn(2, 3, 224, 224)
    audio_features = torch.randn(2, 128)
    
    timings = {}
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(images, audio_features)
    
    # Profile
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            outputs = model(images, audio_features)
            t1 = time.perf_counter()
            timings['total'] = (t1 - t0) * 1000
    
    avg_time = sum(timings.values()) / len(timings)
    print(f"Average forward pass: {avg_time:.2f}ms")
    assert avg_time < 85.0, f"Forward pass too slow: {avg_time}ms (target: <85ms)"
    
    # Verify outputs
    assert 'depth_map' in outputs
    assert 'depth_uncertainty' in outputs
    assert 'sound_classifications' in outputs
    assert 'scene_description' in outputs or outputs.get('scene_description') is None
    
    print("✅ All constraints satisfied")

if __name__ == '__main__':
    profile_forward()
```

---

## Implementation Checklist

- [ ] **Phase 0:** Branch created, backup made
- [ ] **Phase 1.1:** DepthHead updated with uncertainty
- [ ] **Phase 1.2:** SoundEventHead layers moved to __init__
- [ ] **Phase 1.3:** SceneDescriptionHead enhanced (optional)
- [ ] **Phase 2.1:** MaxSightCNN __init__ updated
- [ ] **Phase 2.2:** Audio integration in forward()
- [ ] **Phase 2.3:** Depth integration (vectorized)
- [ ] **Phase 2.4:** Temporal integration
- [ ] **Phase 2.5:** Scene graph integration
- [ ] **Phase 2.6:** Scene description integration
- [ ] **Phase 2.7:** Personalization integration
- [ ] **Phase 3.1:** UncertaintyWeightedLoss added
- [ ] **Phase 3.2:** DetectionLoss updated
- [ ] **Phase 3.3:** Contrastive loss created
- [ ] **Phase 4:** Unit tests created and passing
- [ ] **Phase 5:** Training loop monitoring added
- [ ] **Phase 6:** Profiling script created
- [ ] **Validation:** All tests pass, <85ms inference

---

## Execution Order

1. **Start with heads** (Phase 1) - test each independently
2. **Update __init__** (Phase 2.1) - add all modules
3. **Integrate one component at a time** (Phase 2.2-2.7)
4. **Test after each integration** (run unit tests)
5. **Update losses** (Phase 3)
6. **Full integration test** (Phase 4)
7. **Profile and validate** (Phase 6)

---

## Rollback Plan

If issues arise:
```bash
git checkout ml/models/maxsight_cnn.py
git checkout ml/models/heads/depth_head.py
# etc.
```

Or restore from backup:
```bash
cp ml/models/maxsight_cnn_v1_backup.py ml/models/maxsight_cnn.py
```

---

## Success Criteria

✅ All unit tests pass  
✅ Forward pass < 85ms  
✅ No shape mismatches  
✅ All outputs present in dict  
✅ Uncertainty log_vars monitored  
✅ Audio attention preserves channels  
✅ Depth uncertainty encapsulated  
✅ Temporal features aligned  
✅ Scene graph uses top-K  
✅ Personalization normalized  

---

**Ready to implement?** Start with Phase 1 and test each head independently before integration.

