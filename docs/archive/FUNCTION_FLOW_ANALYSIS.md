# MaxSightCNN Function Flow Analysis

## Complete Forward Pass Flow

### Input Processing
1. **Input**: `images` [B, 3, 224, 224] or [B, T, 3, 224, 224] for video
2. **Temporal Detection**: If `images.dim() == 5`, flatten to [B*T, 3, 224, 224]
3. **Audio**: Optional `audio_features` [B, 128]

---

## STAGE A: Fast Safety Pass (<150ms target)

### 1. Backbone: `_forward_stage_a_backbone()`
**ALWAYS ResNet50 + FPN** (regardless of tier)

**Flow:**
```
images [B, 3, 224, 224]
  ↓
ResNet50:
  - conv1 + bn1 + relu + maxpool
  - layer1 → c2
  - layer2 → c3
  - layer3 → c4
  - layer4 → c5
  ↓
FPN:
  - [c2, c3, c4, c5] → [p2, p3, p4, p5]
  ↓
Optional FPN Attention (T1+)
  ↓
Scene Context:
  - Global Average Pooling on each FPN level
  - Concatenate → scene_context [B, 256]
  ↓
Fused Features:
  - Resize p3, p5 to p4 size
  - Concatenate [p3_resized, p4, p5_resized]
  - detection_fusion conv → fused_features [B, 256, H, W]
```

**Outputs:**
- `fpn_features`: [p2, p3, p4, p5]
- `fused_features`: [B, 256, H, W]
- `scene_context`: [B, 256]

### 2. Audio Processing (if provided)
```
audio_features [B, 128]
  ↓
audio_encoder → audio_emb [B, 256]
  ↓
sound_event_head → sound_outputs
spatial_sound → audio_attention_map [B, 1, H, W]
  ↓
Apply to fused_features:
  fused_features = fused_features * (1.0 + sigmoid(audio_attention_map))
```

### 3. Condition-Specific Enhancements
- **Blur** (cataracts): `contrast_enhance(fused_features)`
- **Spotty** (diabetic retinopathy): `edge_enhance(fused_features)`
- **Night** (retinitis pigmentosa): `fused_features * brightness_enhance`
- **Inconsistent** (CVI/amblyopia): Weighted FPN attention

### 4. Detection Heads (Safety-Critical)
```
fused_features [B, 256, H, W]
  ↓
detection_head → det_feats [B, 256, H, W]
  ↓
Parallel Heads:
  - cls_head → cls_logits [B, H, W, 91]
  - box_head → box_preds [B, H, W, 4]
  - obj_head → obj_logits [B, H, W]
  - text_head → text_logits [B, H, W]
  ↓
Reshape to [B, H*W, C]:
  - classifications [B, H*W, 91]
  - boxes [B, H*W, 4]
  - objectness [B, H*W]
  - text_scores [B, H*W]
```

### 5. Scene-Level Heads
```
combined_context [B, 512] (scene_context + audio_emb)
  ↓
scene_embedding → scene_emb [B, 256]
urgency_head → urgency [B, 4]
  ↓
distance_head → distance_zones [B, H*W, 3]
uncertainty_head → uncertainty [B, 1]
```

### 6. Stage A → Stage B Decision Point

**Skip Conditions:**
1. **High Latency**: `stage_a_latency_ms > 200ms` → `skip_stage_b = True`
2. **High Uncertainty**: `uncertainty > 0.7` → `skip_stage_b = True`
3. **Invalid Scene Graph**: `edge_index/edge_attr mismatch` → `skip_stage_b = True`

**If `skip_stage_b = True`:**
- Return Stage A outputs only
- Set `stage_b_completed = False`
- Set `skip_stage_b_reason = 'high_latency' | 'high_uncertainty' | 'unknown'`

---

## STAGE B: Context Pass (Opportunistic, Tier-Dependent)

### 7. Backbone: `_forward_stage_b_backbone()`

**Inputs:**
- `images`: Raw images [B, 3, 224, 224] (for Hybrid backbone)
- `stage_a_features`: Fused features [B, 256, H, W] (for temporal processing)

**Flow:**
```
Start: stage_b_features = stage_a_features

Option 1: Hybrid Backbone (T2+)
  images [B, 3, 224, 224]
    ↓
  hybrid_backbone → hybrid_fused, aux_features
    ↓
  Extract hybrid_p4 from FPN features
    ↓
  Resize to match stage_b_features spatial dims
    ↓
  Channel adapter (1x1 conv) if needed
    ↓
  Fuse: stage_b_features = stage_b_features + 0.3 * hybrid_p4

Option 2: Temporal Processing (T5+)
  stage_a_features [B, 256, H, W]
    ↓
  Reshape to [B_orig, T, 256, H, W]
    ↓
  temporal_encoder → temporal_outputs
    ↓
  Extract motion_features
    ↓
  Reshape to [B_orig*T, C, H, W]
    ↓
  Project channels if needed
    ↓
  Resize to match stage_b_features
    ↓
  Fuse: stage_b_features = stage_b_features + motion_features
```

**Outputs:**
- `stage_b_features`: [B, 256, H, W]
- `temporal_outputs`: Dict with motion_features, consistency, flicker (if temporal)

### 8. Stage B Heads (Context-Rich)

#### Motion Head (T2+)
```
stage_b_features [B, 256, H, W]
  ↓
motion_head → motion [B, 2, H, W] (optical flow)
```

#### Therapy State Head (T2+)
```
Inputs:
  - eye_features: From FPN
  - motion_features: From temporal or motion head
  - depth_features: From Stage A depth
  - fpn_features: [p2, p3, p4, p5]
  - contrast_features: From fused_features
  ↓
therapy_state_head → {
  'fatigue_score': [B],
  'blink_rate': [B],
  'fixation_stability': [B],
  'depth_map': [B, H, W],
  'uncertainty': [B, H, W],
  'zones': [B, H*W, 3],
  'contrast_map': [B, H, W],
  'edge_map': [B, H, W]
}
```

#### Scene Graph Encoder (T3+)
```
top_k_boxes [B, K, 4]
object_embeddings [B, K, C]
class_names_batch [B, K]
  ↓
scene_graph_encoder.extract_relations() → {
  'edge_index': [2, E],
  'edge_attr': [E, C],
  'relations': List[SceneRelation],
  'object_embeddings': [N, C],
  'batch': [N]
}
  ↓
gnn_encoder → graph_embedding [B, 512]
```

#### OCR Head (T3+)
```
fpn_features [p2, p3, p4, p5]
  ↓
ocr_head → {
  'text_boxes': [B, N, 4],
  'text_scores': [B, N],
  'text_classes': [B, N],
  'recognized_text': List[List[str]]
}
```

#### Scene Description Head (T3+)
```
scene_emb [B, 256]
  ↓
scene_description_head → descriptions List[str]
```

#### Sound Event Head (T4+)
```
audio_emb [B, 256]
  ↓
sound_event_head → {
  'event_classes': [B, num_events],
  'direction': [B, 2],
  'distance': [B]
}
```

#### Predictive Alert Head (T2+)
```
scene_context + motion_features
  ↓
predictive_alert_head → {
  'hazard_probability': [B],
  'navigation_guidance': [B, 3]
}
```

### 9. Retrieval System (T4+, Async, Non-Blocking)

**AsyncRetrievalWorker** (separate thread):
```
Inputs:
  - images [B, 3, 224, 224]
  - audio_features [B, 128] (optional)
  - text_snippets List[List[str]] (optional)
  - scene_graph Dict (optional)
  ↓
Parallel Encoders:
  - global_encoder (CLIP/DINOv2) → global_emb [B, 512]
  - region_extractor → region_emb [B, R, 256], region_boxes [B, R, 4]
  - patch_extractor → patch_emb [B, P, 768]
  - depth_extractor → depth_emb [B, H*W, 256]
  - ocr_encoder → ocr_emb [B, T, 384]
  - audio_encoder → audio_emb [B, A, 256]
  - scene_graph_encoder → sg_emb [B, N, 256]
  ↓
Project to common_dim (256):
  - All embeddings → [B, *, 256]
  ↓
FAISS Index Search (async)
  ↓
Return: retrieval_results Dict
```

**Note**: Retrieval is non-blocking - forward pass continues even if retrieval is slow.

---

## Output Assembly

### Final Output Dictionary

**Stage A Outputs (Always Present):**
- `objectness`: [B, H*W]
- `classifications`: [B, H*W, 91]
- `boxes`: [B, H*W, 4]
- `distance_zones`: [B, H*W, 3]
- `urgency_scores`: [B, 4]
- `uncertainty`: [B, 1]

**Stage B Outputs (If not skipped):**
- `motion`: [B, 2, H, W]
- `therapy_state`: Dict with fatigue, depth, contrast
- `scene_graph`: Dict with edge_index, edge_attr, relations
- `ocr`: Dict with text_boxes, recognized_text
- `scene_description`: List[str]
- `sound_events`: Dict with event_classes, direction
- `predictive_alerts`: Dict with hazard_probability

**Metadata:**
- `stage_a_completed`: bool (always True)
- `stage_b_completed`: bool (True if not skipped)
- `skip_stage_b_reason`: str | None ('high_latency' | 'high_uncertainty' | 'unknown' | None)
- `stage_a_latency_ms`: float | None

---

## Key Architectural Guarantees

1. **Stage A Always ResNet50+FPN**: No hybrid backbone, no temporal processing
2. **Stage B Uses Raw Images**: Hybrid backbone processes raw images, not Stage A features
3. **Temporal Only in Stage B**: Temporal processing uses Stage A features as input
4. **Retrieval is Async**: Non-blocking, advisory only
5. **Safety First**: Stage A completes before Stage B decision
6. **Fail-Safe**: High latency/uncertainty → skip Stage B, return Stage A only

---

## Decision Points Summary

| Decision Point | Condition | Action |
|----------------|-----------|--------|
| Temporal Mode | `images.dim() == 5` | Flatten to [B*T, 3, 224, 224] |
| Audio Processing | `audio_features is not None AND use_audio` | Encode and apply attention |
| Hybrid Backbone | `tier >= T2 AND use_hybrid` | Process raw images, fuse with Stage A |
| Temporal Processing | `tier >= T5 AND use_temporal AND temporal_mode` | Process Stage A features |
| Skip Stage B | `latency > 200ms OR uncertainty > 0.7` | Return Stage A only |
| Scene Graph | `tier >= T3 AND top_k_boxes available` | Extract relations, encode with GNN |
| Retrieval | `tier >= T4 AND use_retrieval` | Async worker, non-blocking |

---

## Performance Characteristics

**Stage A (Always):**
- Backbone: ResNet50 + FPN (~50ms on GPU)
- Heads: 6 safety-critical heads (~30ms)
- **Total: ~80-100ms** (target <150ms)

**Stage B (If not skipped):**
- Hybrid Backbone: +20-40ms (T2+)
- Temporal: +50-100ms (T5+)
- Heads: 5-8 context heads (~50ms)
- **Total: +70-190ms** (opportunistic)

**Full Pipeline:**
- **T0-T1**: ~80-100ms (Stage A only)
- **T2-T3**: ~150-200ms (Stage A + Hybrid + Heads)
- **T4-T5**: ~200-300ms (Stage A + Hybrid + Temporal + All Heads)

