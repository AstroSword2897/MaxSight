# Multimodal Integration Quick Checklist

**Quick reference for implementation progress**

---

## Phase 0: Preparation ✅
- [ ] Create branch `feature/multimodal_refactor`
- [ ] Tag current version `v1.0-before-refactor`
- [ ] Backup `maxsight_cnn.py`
- [ ] Verify PyTorch version

---

## Phase 1: Refactor Heads

### DepthHead (`ml/models/heads/depth_head.py`)
- [ ] Add `uncertainty_conv` to `__init__` (after line 39)
- [ ] Remove Softmax from zone_head (line 50)
- [ ] Update forward() to return uncertainty (lines 72-78)
- [ ] Test: `python -m pytest tests/test_depth_head.py -v`

### SoundEventHead (`ml/models/heads/sound_event_head.py`)
- [ ] Move `input_proj` to `__init__` (after line 45)
- [ ] Move `spectrogram_proj` to `__init__` (after line 45)
- [ ] Update forward() to use pre-initialized layers (lines 136-145)
- [ ] Test: `python -m pytest tests/test_sound_event_head.py -v`

### SceneDescriptionHead (optional)
- [ ] Add cross-attention for regions (optional enhancement)

---

## Phase 2: Integration

### MaxSightCNN (`ml/models/maxsight_cnn.py`)

#### __init__ Updates
- [ ] Add audio modules (after line 404)
- [ ] Add depth_head_module (after line 498)
- [ ] Add temporal_encoder (after line 498)
- [ ] Add scene_graph_encoder (after line 498)
- [ ] Add global_encoder + scene_description_head (after line 498)
- [ ] Add personalization modules (after line 498)

#### Forward Pass Updates
- [ ] Update forward signature (add user_id, prev_temporal_state, use_temporal)
- [ ] Replace audio_branch with enhanced audio (lines 719-725)
- [ ] Add depth integration (after line 799)
- [ ] Add temporal integration (after line 705)
- [ ] Add scene graph integration (after line 811)
- [ ] Add scene description integration (after scene graph)
- [ ] Add personalization integration (after scene description)

---

## Phase 3: Loss Updates

### Losses (`ml/training/losses.py`)
- [ ] Add `UncertaintyWeightedLoss` class (after line 174)
- [ ] Add uncertainty_weighting to DetectionLoss.__init__ (after line 255)
- [ ] Update DetectionLoss.forward() to use uncertainty weighting (lines 342-349)

### Personalization Loss
- [ ] Create `ml/training/personalization_loss.py`
- [ ] Add `compute_contrastive_loss` function

---

## Phase 4: Testing

### Unit Tests (`tests/test_integration_constraints.py`)
- [ ] Create test file
- [ ] Add `test_audio_attention_preserves_channels`
- [ ] Add `test_depth_uncertainty_encapsulated`
- [ ] Add `test_temporal_spatial_alignment`
- [ ] Add `test_scene_graph_top_k`
- [ ] Add `test_personalization_normalized`
- [ ] Add `test_depth_vectorized`
- [ ] Run: `python -m pytest tests/test_integration_constraints.py -v`

---

## Phase 5: Training Loop

### Train Loop (`ml/training/train_loop.py`)
- [ ] Add uncertainty log_vars monitoring (after line 478)

---

## Phase 6: Profiling

### Profiling Script
- [ ] Create `scripts/profile_integration.py`
- [ ] Run profiling: `python scripts/profile_integration.py`
- [ ] Verify < 85ms inference time

---

## Validation

- [ ] All unit tests pass
- [ ] Forward pass < 85ms
- [ ] No shape mismatches
- [ ] All outputs present
- [ ] Uncertainty log_vars logged
- [ ] Audio attention preserves channels
- [ ] Depth uncertainty encapsulated
- [ ] Temporal features aligned
- [ ] Scene graph uses top-K
- [ ] Personalization normalized

---

## Quick Commands

```bash
# Run all tests
python -m pytest tests/test_integration_constraints.py -v

# Profile forward pass
python scripts/profile_integration.py

# Check for linting errors
python -m pylint ml/models/maxsight_cnn.py

# Run specific test
python -m pytest tests/test_integration_constraints.py::test_audio_attention_preserves_channels -v
```

---

**Status:** Ready to start Phase 1

