# Quantization Code Fixes

## Issues in Your Snippet

Your code snippet had several syntax errors and API issues. Here's what was wrong and how to fix it:

### 1. Function Signature Error

**Wrong:**
```python
def _fuse_maxsight_modules(model: nn.Module -> nn.Module) -> nn.Module:
```

**Correct:**
```python
def _fuse_maxsight_modules(model: nn.Module) -> nn.Module:
```

The `->` is only used in return type annotations, not in parameter types.

### 2. Missing Colon and List Initialization

**Wrong:**
```python
fuse_list[]
```

**Correct:**
```python
fuse_list = []
```

### 3. Incorrect API Usage

**Wrong:**
```python
torch.quantization.fuse_modules(model, inplace=True)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization(model, inplace=False)
model_int8 = torch.quantization.convert(model_prepared, inplace=False)
```

**Correct:**
```python
import torch.ao.quantization as quantization

quantization.fuse_modules(model, [fuse_pattern], inplace=True)
model.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model, inplace=False)
model_int8 = quantization.convert(model_prepared, inplace=False)
```

**Key changes:**
- Use `torch.ao.quantization` (modern API) instead of deprecated `torch.quantization`
- `fuse_modules` requires a list of patterns: `[fuse_pattern]`
- Use `quantization.prepare()` not `torch.quantization()`
- Use `quantization.convert()` not `torch.quantization.convert()`

### 4. Variable Name Typo

**Wrong:**
```python
if fuse_models:  # Should be fuse_modules
```

**Correct:**
```python
if fuse_modules:
```

### 5. Missing Parameter in Function Signature

**Wrong:**
```python
def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10
    backend: str = 'fbgemm',  # Missing comma!
    fuse_modules: bool = True
) -> nn.Module:
```

**Correct:**
```python
def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,  # Comma added
    backend: str = 'fbgemm',
    fuse_modules: bool = True
) -> nn.Module:
```

### 6. Incorrect Fuse Modules Call

**Wrong:**
```python
model = torch.quantization.fuse_modules(model, inplace=True)
```

**Correct:**
```python
model = _fuse_maxsight_modules(model)  # Use the helper function
```

The `fuse_modules` function requires specific patterns, not the whole model.

### 7. Indentation Issues

Your calibration loop had inconsistent indentation. Make sure all code inside the `with torch.no_grad():` block is properly indented.

## Corrected Version

Here's the corrected version that matches your existing production code:

```python
import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from typing import Optional
from copy import deepcopy
import warnings


def _fuse_maxsight_modules(model: nn.Module) -> nn.Module:
    """Fuse Conv+BN+ReLU patterns in MaxSight CNN."""
    fuse_list = []
    
    def is_fusable_conv_bn_relu(seq: nn.Sequential, start_idx: int = 0) -> bool:
        if len(seq) < start_idx + 3:
            return False
        return (isinstance(seq[start_idx], nn.Conv2d) and
                isinstance(seq[start_idx + 1], nn.BatchNorm2d) and
                isinstance(seq[start_idx + 2], (nn.ReLU, nn.ReLU6)))
    
    # Find fusable patterns
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            if is_fusable_conv_bn_relu(module, 0):
                fuse_pattern = [f"{name}.0", f"{name}.1", f"{name}.2"]
                fuse_list.append(fuse_pattern)
    
    # ResNet backbone patterns
    if hasattr(model, 'conv1') and hasattr(model, 'bn1') and hasattr(model, 'relu'):
        fuse_list.append(['conv1', 'bn1', 'relu'])
    
    # Fuse all patterns
    fused_count = 0
    for fuse_pattern in fuse_list:
        try:
            quantization.fuse_modules(model, [fuse_pattern], inplace=True)
            fused_count += 1
        except Exception:
            continue
    
    if fused_count > 0:
        print(f"✓ Fused {fused_count} Conv+BN+ReLU patterns")
    else:
        warnings.warn("No modules were fused.")
    
    return model


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,
    backend: str = 'qnnpack',
    fuse_modules: bool = True
) -> nn.Module:
    """Quantize model to int8."""
    model = deepcopy(model)
    model.eval()
    
    torch.backends.quantized.engine = backend
    
    # Fuse modules
    if fuse_modules:
        try:
            model = _fuse_maxsight_modules(model)
        except Exception as e:
            warnings.warn(f"Fusion failed: {e}")
    
    # Set quantization config
    model.qconfig = quantization.get_default_qconfig(backend)
    
    # Per-channel weight quantization for qnnpack (ARM/iOS)
    if backend == 'qnnpack' and model.qconfig is not None:
        model.qconfig.weight = quantization.default_per_channel_weight_observer
    
    # Prepare model
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibration
    print(f"Calibrating with {num_calibration_batches} batches...")
    if calibration_data is None:
        print("Warning: Using synthetic calibration data")
        calibration_data = [(torch.randn(1, 3, 224, 224),) for _ in range(num_calibration_batches)]
    
    batch_count = 0
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('images') or batch.get('input')
            else:
                inputs = batch
            
            if inputs.device.type != 'cpu':
                inputs = inputs.cpu()
            
            try:
                model_prepared(inputs)
                batch_count += 1
                if batch_count >= num_calibration_batches:
                    break
            except Exception as e:
                warnings.warn(f"Batch {batch_count} failed: {e}")
                continue
    
    if batch_count == 0:
        raise RuntimeError("No batches processed during calibration")
    
    # Convert to INT8
    print("Converting to INT8...")
    model_int8 = quantization.convert(model_prepared, inplace=False)
    
    print(f"✓ Quantization complete ({batch_count} batches)")
    return model_int8
```

## Key Takeaways

1. **Always use `torch.ao.quantization`** - the old `torch.quantization` is deprecated
2. **`fuse_modules` requires pattern lists** - not the whole model
3. **Use `quantization.prepare()` and `quantization.convert()`** - not `torch.quantization()`
4. **Check function signatures** - ensure all parameters have correct types and commas
5. **Your existing `ml/training/quantization.py` is already correct** - use that as reference!

