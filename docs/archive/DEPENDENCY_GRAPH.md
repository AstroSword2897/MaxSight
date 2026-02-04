# MaxSight Dependency Graph and Versioning

**Last Updated:** December 2024  
**Version:** 1.0.0

---

## Overview

This document tracks dependencies between components, execution order, and versioning to ensure reproducible training and inference pipelines.

---

## Component Dependencies

### Model Architecture Dependencies

```
MaxSightCNN
├── Backbone (ResNet50)
│   └── Output: Multi-scale features [C2, C3, C4, C5]
│
├── FPN (Feature Pyramid Network)
│   └── Dependencies: [C2, C3, C4, C5]
│   └── Output: FPN features [P2, P3, P4, P5]
│
├── Scene Context
│   └── Dependencies: [P2, P3, P4, P5]
│   └── Output: scene_context [B, 256]
│
├── Detection Features
│   └── Dependencies: [P3, P4, P5]
│   └── Output: det_feats [B, 256, H, W]
│
└── Multi-Head Architecture
    ├── Classification Head
    │   └── Dependencies: [det_feats]
    │   └── Output: classifications [B, 196, 80]
    │
    ├── Box Regression Head
    │   └── Dependencies: [det_feats]
    │   └── Output: boxes [B, 196, 4]
    │
    ├── Objectness Head
    │   └── Dependencies: [det_feats]
    │   └── Output: objectness [B, 196]
    │
    ├── Text Region Head
    │   └── Dependencies: [det_feats]
    │   └── Output: text_regions [B, 196]
    │
    ├── Scene Embedding
    │   └── Dependencies: [scene_context]
    │   └── Output: scene_embedding [B, 512]
    │
    ├── Shared Scene Embedding
    │   └── Dependencies: [scene_context]
    │   └── Output: shared_scene_embedding [B, 256]
    │
    ├── Urgency Head
    │   └── Dependencies: [shared_scene_embedding]
    │   └── Output: urgency_scores [B, 4]
    │
    ├── Distance Head
    │   └── Dependencies: [shared_scene_embedding, boxes]
    │   └── Output: distance_zones [B, 196, 3]
    │
    ├── Contrast Head
    │   └── Dependencies: [shared_scene_embedding]
    │   └── Output: contrast_sensitivity [B, 1]
    │
    ├── Glare Head
    │   └── Dependencies: [shared_scene_embedding]
    │   └── Output: glare_risk_level [B, 4]
    │
    ├── Findability Head
    │   └── Dependencies: [det_feats]
    │   └── Output: object_findability [B, 196]
    │
    ├── Navigation Difficulty Head
    │   └── Dependencies: [shared_scene_embedding]
    │   └── Output: navigation_difficulty [B, 1]
    │
    └── Uncertainty Head
        └── Dependencies: [shared_scene_embedding]
        └── Output: uncertainty [B, 1]
```

---

## Execution Order

Heads must be executed in dependency order:

1. **Backbone** → Multi-scale features
2. **FPN** → FPN features
3. **Scene Context** → Scene embedding
4. **Detection Features** → Detection features
5. **Shared Scene Embedding** → Shared embedding (for accessibility heads)
6. **Core Heads** (parallel):
   - Classification
   - Box Regression
   - Objectness
   - Text Region
7. **Scene-Level Heads** (parallel, depend on shared_scene_embedding):
   - Urgency
   - Contrast
   - Glare
   - Navigation Difficulty
   - Uncertainty
8. **Dependent Heads** (depend on previous outputs):
   - Distance (depends on boxes + shared_scene_embedding)
   - Findability (depends on det_feats)

---

## System-Level Dependencies

### Processing Pipeline

```
Input Image
    ↓
Preprocessing (ImagePreprocessor)
    ↓ (depends on: condition_mode)
    ↓
Model Inference (MaxSightCNN)
    ↓ (outputs: detections, urgency, distance, scene_embedding)
    ↓
OCR Integration
    ↓ (depends on: model.text_regions)
    ↓ (outputs: text_results)
    ↓
Description Generator
    ↓ (depends on: detections, urgency_scores, text_results)
    ↓ (outputs: scene_description)
    ↓
Spatial Memory
    ↓ (depends on: detections)
    ↓ (outputs: spatial_context)
    ↓
Path Planner
    ↓ (depends on: spatial_context)
    ↓ (outputs: path_info)
    ↓
Output Scheduler
    ↓ (depends on: detections, urgency_scores, uncertainty)
    ↓ (outputs: scheduled_outputs)
    ↓
Therapy Integration
    ↓ (depends on: detections, session_manager)
    ↓ (outputs: therapy_feedback)
    ↓
Overlay Engine
    ↓ (depends on: detections, urgency_scores, text_regions)
    ↓ (outputs: overlay_image)
    ↓
Voice Feedback
    ↓ (depends on: detections, urgency_scores, scene_description)
    ↓ (outputs: voice_announcements)
    ↓
Haptic Feedback
    ↓ (depends on: detections, urgency_scores, path_info)
    ↓ (outputs: haptic_patterns)
```

---

## Versioning

### Model Version: 1.0.0

**Component Versions:**
- Backbone: 1.0.0 (ResNet50)
- FPN: 1.0.0 (SimplifiedFPN)
- Classification Head: 1.0.0
- Box Regression Head: 1.0.0
- Objectness Head: 1.0.0
- Text Region Head: 1.0.0
- Urgency Head: 1.0.0
- Distance Head: 1.0.0
- Accessibility Heads: 1.0.0

**Dependency Versions:**
- PyTorch: >=2.9.1
- torchvision: >=0.24.1
- scipy: >=1.11.0 (for Hungarian matching)
- scikit-learn: >=1.3.0 (for OCR clustering)

---

## Error Propagation and Fallbacks

### Fallback Strategy

1. **Head Failure**: If a head fails, use default outputs (zeros)
2. **Dependency Missing**: If dependency missing, skip dependent head or use fallback
3. **High Uncertainty**: If uncertainty > threshold, use conservative outputs
4. **Timeout**: If head exceeds timeout, use fallback or skip

### Fallback Order

1. **Try primary execution**
2. **If error**: Use fallback function (if provided)
3. **If fallback fails**: Use default outputs
4. **If all fail**: Return minimal safe outputs

---

## Real-Time Constraints

### Latency Targets

- **Core Heads Only**: <200ms
- **Core + Text**: <250ms
- **Core + Urgency**: <300ms
- **Core + Distance**: <350ms
- **All Heads**: <500ms (target)
- **All Heads + Accessibility**: <600ms (acceptable)

### Head Latency Breakdown (Estimated)

| Head | Estimated Latency | Critical |
|------|------------------|----------|
| Classification | 50ms | ✅ Yes |
| Box Regression | 40ms | ✅ Yes |
| Objectness | 30ms | ✅ Yes |
| Text Region | 20ms | ⚠️ Optional |
| Urgency | 15ms | ⚠️ Optional |
| Distance | 25ms | ⚠️ Optional |
| Contrast | 10ms | ⚠️ Optional |
| Glare | 10ms | ⚠️ Optional |
| Findability | 20ms | ⚠️ Optional |
| Navigation Difficulty | 10ms | ⚠️ Optional |
| Uncertainty | 10ms | ⚠️ Optional |

**Total (all heads)**: ~240ms (estimated, actual depends on hardware)

---

## Recommendations

### For Production Deployment

1. **Enable only required heads** based on use case
2. **Use quantization** (INT8) to reduce latency
3. **Benchmark on target hardware** (mobile device)
4. **Monitor uncertainty** and use fallbacks when high
5. **Cache shared embeddings** to avoid recomputation

### For Training

1. **Train all heads together** for best accuracy
2. **Use dependency order** for loss computation
3. **Validate dependencies** before training
4. **Version configurations** for reproducibility

---

## Dependency Validation

Use `DependencyGraph.validate_dependencies()` to check that all required dependencies are available before execution.

Example:
```python
from ml.config import DependencyGraph

graph = DependencyGraph()
outputs = {'model': {'detections': [...], 'urgency_scores': [...]}}
validation = graph.validate_dependencies(outputs)

# Check if all components can run
if all(validation.values()):
    # All dependencies satisfied
    pass
else:
    # Some dependencies missing
    missing = [comp for comp, valid in validation.items() if not valid]
    print(f"Missing dependencies: {missing}")
```

---

**Last Updated:** December 2024  
**Maintainer:** Engineering Team

