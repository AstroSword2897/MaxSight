# Loss Diagnostics Script Analysis

## Critical Issues Found

### 1. **Loss Function Interface Mismatch** ❌ CRITICAL

**Problem**: Individual loss functions in `ml/training/losses.py` (`ObjectnessLoss`, `ClassificationLoss`, etc.) expect `(predictions, targets)` where both are **tensors**, but `GradNormMultiHeadLoss.compute_head_losses()` calls them with `(outputs, targets)` where `outputs` is a **dictionary**.

**Location**: `GradNormMultiHeadLoss.compute_head_losses()` line 386
```python
loss_dict = loss_fn(outputs, targets)  # outputs is Dict[str, Tensor]
```

**Verification**: Tested - `ObjectnessLoss` fails when called with dict, works with tensors.

**Root Cause**: There are TWO loss function systems:
1. `ml/training/losses.py` - Tensor-based losses (ObjectnessLoss, ClassificationLoss, etc.)
2. `ml/training/head_losses.py` - Dict-based losses (HeadLoss base class)

**Fix Required**: The script MUST use wrapper functions that:
1. Extract correct keys from `outputs` dictionary
2. Call tensor-based loss functions
3. Return `{'loss': tensor}` dictionary format

**Example Fix**:
```python
class WrappedObjectnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = ObjectnessLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = outputs.get('objectness')
        target = targets.get('objectness')
        if predictions is None or target is None:
            device = next(iter(outputs.values())).device
            return {'loss': torch.tensor(0.0, device=device)}
        loss = self.loss_fn(predictions, target)
        return {'loss': loss}
```

### 2. **Model Output Key Mismatch** ❌

**Problem**: Model outputs use different keys than targets:
- Model: `'classifications'`, `'boxes'`, `'objectness'`, `'distance_zones'`, `'urgency_scores'`
- Targets: `'labels'`, `'boxes'`, `'objectness'`, `'distance'`, `'urgency'`

**Location**: Script creates targets with `'labels'` but model outputs `'classifications'`

**Fix Required**: Either:
1. Map model outputs to match target keys, OR
2. Map target keys to match model outputs, OR
3. Update loss functions to handle both key formats

### 3. **Loss Function Return Type Mismatch** ⚠️

**Problem**: Individual loss functions return `torch.Tensor`, but `GradNormMultiHeadLoss.compute_head_losses()` expects a dictionary with `'loss'` key.

**Location**: `ml/training/task_balancing.py` line 386-391
```python
loss_dict = loss_fn(outputs, targets)
head_loss_dicts[head_name] = loss_dict  # Expects dict with 'loss' key
```

**But**: Individual loss functions return `torch.Tensor` directly.

**Fix Required**: Either:
1. Wrap loss functions to return `{'loss': tensor}`, OR
2. Update `GradNormMultiHeadLoss` to handle tensor returns

### 4. **Target Shape Mismatch** ⚠️

**Problem**: Script creates targets with shape `(num_preds,)` but model outputs have shape `[B, H*W, ...]` where `H*W` is dynamic (typically 14*14 = 196 for 224x224 input).

**Location**: `SyntheticMaxSightDataset.__getitem__()` creates `num_preds = 100` but model outputs depend on input size.

**Fix Required**: Match target shapes to model output shapes dynamically.

### 5. **Monitor Update Format** ✅ (Actually OK)

**Status**: The monitor update format is correct. `PerHeadLossMonitor.update()` expects `Dict[str, torch.Tensor]` and the script provides that.

### 6. **Gradient Collection Timing** ✅ (Actually OK)

**Status**: The script calls `total_loss.backward()` before collecting gradients, which is correct.

### 7. **Missing Import** ⚠️

**Problem**: Script uses `from ml.training.losses import *` which is not ideal but should work. However, need to verify all loss classes are exported.

**Fix Required**: Use explicit imports or verify `__all__` in `ml/training/losses.py`.

## Required Fixes

### Fix 1: Create Loss Function Wrappers

```python
class WrappedObjectnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = ObjectnessLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = outputs.get('objectness')  # Extract from outputs
        target = targets.get('objectness')
        if predictions is None or target is None:
            device = next(iter(outputs.values())).device
            return {'loss': torch.tensor(0.0, device=device)}
        
        loss = self.loss_fn(predictions, target)
        return {'loss': loss}
```

### Fix 2: Fix Target Key Mapping

```python
# In SyntheticMaxSightDataset
targets = {
    "objectness": ...,
    "labels": ...,  # Change to match model output
    "boxes": ...,
    "distance": ...,  # Model outputs 'distance_zones'
    "urgency": ...,   # Model outputs 'urgency_scores'
}
```

### Fix 3: Match Target Shapes to Model Outputs

```python
# After model forward pass
outputs = model(images)
B, H, W = outputs['objectness'].shape[:3]  # Get actual shape
num_locations = H * W

# Create targets matching model output shapes
targets = {
    "objectness": torch.randint(0, 2, (B, num_locations)).float(),
    "labels": torch.randint(0, num_classes, (B, num_locations)),
    "boxes": torch.rand(B, num_locations, 4),
    "distance": torch.randint(0, 3, (B, num_locations)),
    "urgency": torch.randint(0, 4, (B,)),
}
```

## Recommendations

1. **Use existing `collect_loss_data.py`**: The repository already has a working loss collection script that handles these issues correctly.

2. **Fix the provided script**: Apply the fixes above to make it compatible with the repository structure.

3. **Test with actual model outputs**: Verify the script works with real model outputs, not just synthetic data.

## Summary

The script has good structure but needs fixes for:
- Loss function interface compatibility
- Model output key mapping
- Target shape matching
- Loss function return type handling

The repository's existing `scripts/collect_loss_data.py` already handles these issues correctly and should be used instead, or the provided script should be updated to match the repository's patterns.

