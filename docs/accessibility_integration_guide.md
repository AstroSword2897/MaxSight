# MaxSight Accessibility Integration Guide
## Quick Start: Adding Features to Current Architecture

This guide shows how to integrate the accessibility features from the blueprint into your existing MaxSight codebase.

---

## Current Architecture Overview

Your `MaxSightCNN` model currently outputs:
- `classifications`: [B, 196, num_classes]
- `boxes`: [B, 196, 4]
- `objectness`: [B, 196]
- `text_regions`: [B, 196]
- `scene_embedding`: [B, 512]
- `urgency_scores`: [B, 4]
- `distance_zones`: [B, 196, 3]

---

## Integration Strategy

### Step 1: Add Feature Flags

Modify `MaxSightCNN.__init__()`:

```python
def __init__(
    self,
    num_classes: int = len(COCO_CLASSES),
    num_urgency_levels: int = 4,
    num_distance_zones: int = 3,
    use_audio: bool = True,
    condition_mode: Optional[str] = None,
    fpn_channels: int = 256,
    detection_threshold: float = 0.5,
    # NEW: Accessibility feature flags
    enable_priority_scoring: bool = True,
    enable_functional_vision: bool = False,
    enable_behavior_impact: bool = False,
    enable_social_understanding: bool = False,
    enable_spatial_mapping: bool = False,
):
```

### Step 2: Add Heads Incrementally

Start with **Priority Scoring** (highest impact, easiest to add):

```python
# In __init__()
if enable_priority_scoring:
    self.priority_head = nn.Sequential(
        nn.Conv2d(256, 128, 3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, 1),
        nn.Sigmoid()  # Output 0-1, scale to 0-100 in post-processing
    )
```

### Step 3: Extend Forward Pass

```python
# In forward(), after existing heads:
if self.enable_priority_scoring:
    priority_logits = self.priority_head(det_feats)
    priority_logits = priority_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W)
    outputs['priority_scores'] = priority_logits * 100  # Scale to 0-100
```

### Step 4: Update Loss Function

Add to `MaxSightLoss` in `ml/training/losses.py`:

```python
def compute_priority_loss(self, predictions, targets):
    """Compute priority scoring loss"""
    if 'priority_scores' not in predictions:
        return torch.tensor(0.0, device=predictions['classifications'].device)
    
    pred_priority = predictions['priority_scores']  # [B, H*W]
    gt_priority = targets['priority']  # [B, H*W] or [B, max_objects]
    
    # Match priorities to detections (similar to box matching)
    # ... matching logic ...
    
    return F.mse_loss(matched_pred, matched_gt)
```

---

## Implementation Order (Recommended)

### Phase 1: Foundation (Do First)
1. ✅ **Priority Scoring** - Add priority_head, update get_detections()
2. ✅ **User Profile System** - Create `ml/utils/user_profiles.py`
3. ✅ **Scene Type Classifier** - Extend existing scene_embedding

### Phase 2: Functional Vision
4. **Contrast Sensitivity** - Add contrast_head
5. **Glare Detection** - Add glare_head
6. **Field-of-View** - Add field_of_view_head

### Phase 3: Behavior Impact
7. **Findability Scoring** - Add findability_head
8. **Recognizability** - Add recognizability_head
9. **Navigation Difficulty** - Add navigation_difficulty_head

### Phase 4: Environmental Context
10. **Lighting Classifier** - Add lighting_classifier
11. **Hazard Heatmap** - Add hazard_heatmap_head
12. **Movement Analysis** - Add movement_analyzer (requires temporal)

### Phase 5: Social Understanding
13. **Body Orientation** - Add body_orientation_head
14. **Gesture Classification** - Add gesture_classifier
15. **Interaction Intent** - Add interaction_intent_head

### Phase 6: Spatial Mapping
16. **Semantic Segmentation** - Add semantic_segmentation_head
17. **Pathfinding** - Add pathfinding algorithm

---

## Code Structure

```
ml/
├── models/
│   └── maxsight_cnn.py          # Add new heads here
├── training/
│   └── losses.py                # Add new loss terms
├── utils/
│   ├── user_profiles.py         # NEW: User profile management
│   ├── audio_output.py          # NEW: Spatial audio beacons
│   ├── haptic_output.py         # NEW: Vibration patterns
│   ├── narration.py             # NEW: Scene summaries
│   └── pathfinding.py           # NEW: Pathfinding algorithm
└── data/
    └── accessibility/           # NEW: Accessibility datasets
        ├── priority_labels/
        ├── functional_vision/
        └── ...
```

---

## Quick Integration Example

### Adding Priority Scoring (5 minutes)

1. **Add head to model:**
```python
# In MaxSightCNN.__init__()
self.priority_head = nn.Sequential(
    nn.Conv2d(256, 128, 3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 1, 1),
    nn.Sigmoid()
)
```

2. **Add to forward pass:**
```python
# In forward(), after det_feats
priority_scores = self.priority_head(det_feats)
priority_scores = priority_scores.permute(0, 2, 3, 1).reshape(batch_size, H*W)
outputs['priority_scores'] = priority_scores * 100
```

3. **Update get_detections():**
```python
# In get_detections(), add priority to each detection
for idx in nms_indices:
    # ... existing code ...
    priority = float(priority_scores[idx].item() * 100)
    img_detections.append({
        # ... existing fields ...
        'priority': priority
    })
```

4. **Add priority-based filtering:**
```python
def filter_by_priority(detections, user_profile):
    alert_freq = user_profile.get('alert_frequency', 'medium')
    threshold = {'low': 70, 'medium': 40, 'high': 0}[alert_freq]
    return [d for d in detections if d['priority'] >= threshold]
```

---

## Testing Strategy

### Unit Tests
- Test each new head outputs correct shapes
- Test priority filtering logic
- Test user profile loading/saving

### Integration Tests
- Test end-to-end: image → outputs → detections → filtered detections
- Test user profile integration
- Test cross-sensory output generation

### Validation
- Compare priority scores with expert labels
- Validate scene type classification accuracy
- Test pathfinding on known layouts

---

## Data Requirements

### Minimum Viable Dataset
- **Priority Labels:** 1000 images with priority scores per detection
- **Scene Types:** 500 images per scene type (10 types = 5000 images)
- **Functional Vision:** 500 images with contrast/glare labels

### Data Collection
1. Use existing COCO dataset + add accessibility labels
2. Collect failure cases (hard cases dataset)
3. Synthetic augmentation for edge cases

---

## Performance Considerations

### Model Size
- Each new head adds ~100KB-1MB
- Total accessibility features: ~5-10MB additional
- Still under 50MB target with INT8 quantization

### Inference Speed
- Priority scoring: +2-3ms
- Scene type: +1ms
- Functional vision: +5-10ms total
- **Target:** Keep total inference <500ms

### Training Time
- Priority scoring: +1 hour training time
- Full accessibility suite: +4-6 hours training time

---

## Next Steps

1. **Review blueprint** - Confirm feature set
2. **Start with Phase 1** - Priority scoring + user profiles
3. **Collect data** - Begin labeling priority scores
4. **Iterate** - Add features incrementally

---

**Status:** Ready for implementation  
**Last Updated:** 2026

