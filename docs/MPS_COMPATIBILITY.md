# MPS Compatibility Guide for MaxSight 3.0

## Overview

MaxSight 3.0 includes **MPS-stable mode** for local development on Apple Silicon. This mode sacrifices some learning capability to avoid PyTorch MPS backward pass crashes.

## When to Use MPS-Stable Mode

### ✅ Use MPS-Stable Mode When:
- **Local development** on Apple Silicon (M1/M2/M3)
- **Forward pass testing** and debugging
- **Small-scale training** (1-2 scenes per batch)
- **Prototyping** before moving to cloud GPU

### ❌ Don't Use MPS-Stable Mode When:
- **Production training** (use cloud GPU instead)
- **Full gradient learning** on edges is required
- **Large batch training** (use cloud GPU)
- **Performance-critical** inference

## What MPS-Stable Mode Does

### SceneGraphEncoder (`mps_stable=True`)

**Trade-off:** Edge attributes (`edge_attr`) are detached from the computation graph.

**Impact:**
- ✅ Forward pass works correctly
- ✅ Relations and edge_index are correct
- ❌ Edge embeddings don't learn (frozen)
- ✅ Avoids MPS backward crashes

**Code:**
```python
scene_graph_encoder = SceneGraphEncoder(
    object_embed_dim=256,
    relation_embed_dim=128,
    num_spatial_relations=6,
    num_semantic_relations=10,
    mps_stable=True  # Enable MPS workaround
)
```

### GNNEncoder (`mps_stable=True`)

**Trade-off:** Batched graph pooling uses CPU fallback for `index_add`.

**Impact:**
- ✅ Forward pass works correctly
- ✅ Graph embeddings are correct
- ⚠️ Slower (CPU fallback for pooling)
- ✅ Avoids MPS kernel crashes

**Code:**
```python
gnn_encoder = GNNEncoder(
    node_dim=256,
    edge_dim=128,
    hidden_dim=256,
    num_layers=3,
    output_dim=512,
    mps_stable=True  # Enable MPS workaround
)
```

## Auto-Detection

MaxSightCNN **automatically detects MPS** and enables `mps_stable` mode:

```python
# In MaxSightCNN.__init__
device_type = "mps" if torch.backends.mps.is_available() else "cpu"
mps_stable = (device_type == "mps")  # Auto-enable on MPS
```

This means:
- **Apple Silicon**: Automatically uses MPS-stable mode
- **CUDA/CPU**: Uses full gradient mode (no workarounds)

## Manual Override

If you want to **force full gradients** on MPS (not recommended):

```python
# In MaxSightCNN.__init__, override auto-detection
self.scene_graph_encoder = SceneGraphEncoder(
    ...,
    mps_stable=False  # Force full gradients (may crash on backward)
)
```

## Performance Comparison

| Mode | Device | Edge Learning | Backward Stability | Speed |
|------|--------|---------------|-------------------|-------|
| MPS-Stable | MPS | ❌ No | ✅ Stable | Medium |
| Full Gradients | MPS | ✅ Yes | ❌ Crashes | N/A |
| Full Gradients | CUDA | ✅ Yes | ✅ Stable | Fast |
| Full Gradients | CPU | ✅ Yes | ✅ Stable | Very Slow |

## Recommended Workflow

### 1. Local Development (Apple Silicon)
```python
# MaxSightCNN auto-detects MPS and enables mps_stable
model = MaxSightCNN(...)  # Automatically uses MPS-stable mode
# Forward passes work, backward may be limited
```

### 2. Cloud GPU Training
```python
# On CUDA, mps_stable is automatically False
model = MaxSightCNN(...).cuda()  # Full gradients, no workarounds
# Full training with edge learning
```

### 3. Hybrid Approach
```python
# Develop locally with MPS-stable, train on cloud GPU
# 1. Local: Test forward passes, debug architecture
# 2. Cloud: Full training with gradients
```

## Known MPS Limitations

1. **`index_add` crashes** on large batched tensors → CPU fallback in MPS-stable mode
2. **Residual + LayerNorm** can cause backward crashes → Edge detach in MPS-stable mode
3. **`grid_sample` backward** not implemented → Already handled in depth head (skipped on MPS)
4. **Large pairwise computations** (O(N²)) → Use small batches (1-2 scenes)

## Migration Path

When moving from local MPS to cloud GPU:

1. **No code changes needed** - `mps_stable` auto-detects device
2. **Model checkpoints** are compatible (edge embeddings may be frozen in MPS checkpoints)
3. **Training resumes** with full gradients on CUDA

## Summary

- **MPS-stable mode** = Local development on Apple Silicon
- **Full gradients** = Cloud GPU training
- **Auto-detection** = No manual configuration needed
- **Trade-off** = Edge learning vs. backward stability

For production training, **always use cloud GPU** (CUDA) with `mps_stable=False`.

