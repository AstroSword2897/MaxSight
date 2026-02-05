# Export MaxSight Model for Xcode

**Complete guide to export the trained model for iOS/Xcode integration.**

---

## Quick Export

**Export iOS bundle (recommended - includes everything):**

```python
from ml.models.maxsight_cnn import create_model
from ml.training.export import export_ios_bundle
import torch

# Load trained model
model = create_model()
checkpoint = torch.load("checkpoints/final_model.pt", map_location="cpu", weights_only=True)
state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
model.load_state_dict(state, strict=False)
model.eval()

# Export iOS bundle
bundle_path = export_ios_bundle(
    model=model,
    output_dir="maxsight_ios_bundle",
    input_size=(1, 3, 224, 224)
)

print(f"✅ iOS bundle exported to: {bundle_path}")
```

**Output files:**
- `maxsight.pte` - ExecuTorch model (or `maxsight_traced.pt` if ExecuTorch unavailable)
- `model_config.json` - Model parameters and thresholds
- `runtime_config.json` - Runtime settings
- `processing_reference.py` - Reference implementation (port to Swift)
- `README_XCODE.md` - Complete iOS integration guide

---

## Individual Format Exports

**CoreML (iOS native):**
```python
from ml.training.export import export_to_coreml

coreml_path = export_to_coreml(
    model=model,
    save_path="maxsight.mlpackage",
    input_size=(1, 3, 224, 224),
    validate=True
)
```

**ExecuTorch (.pte):**
```python
from ml.training.export import export_to_executorch

pte_path = export_to_executorch(
    model=model,
    save_path="maxsight.pte",
    input_size=(1, 3, 224, 224),
    validate=True
)
```

**JIT (PyTorch mobile):**
```python
from ml.training.export import export_to_jit

jit_path = export_to_jit(
    model=model,
    save_path="maxsight_traced.pt",
    input_size=(1, 3, 224, 224),
    validate=True
)
```

---

## Preprocessing (Critical for Xcode)

**The model expects ImageNet preprocessing:**

1. **Resize**: 224x224 pixels
2. **Normalize**: 
   - Mean: `[0.485, 0.456, 0.406]` (RGB channels)
   - Std: `[0.229, 0.224, 0.225]` (RGB channels)

**Swift preprocessing example:**
```swift
func preprocessImage(_ image: UIImage) -> Tensor {
    // 1. Resize to 224x224
    let resized = image.resized(to: CGSize(width: 224, height: 224))
    
    // 2. Convert to tensor [0, 1]
    let tensor = Tensor.fromImage(resized)
    
    // 3. Normalize with ImageNet values
    let mean = Tensor([0.485, 0.456, 0.406])
    let std = Tensor([0.229, 0.224, 0.225])
    let normalized = (tensor - mean) / std
    
    // 4. Add batch dimension
    return normalized.unsqueeze(0)  // [1, 3, 224, 224]
}
```

**See `processing_reference.py` in the exported bundle for complete reference.**

---

## Which Checkpoint to Use

**Recommended:**
- `checkpoints/final_model.pt` - Final trained model (985MB)
- `checkpoints/best_model.pt` - Best validation performance
- `checkpoints/last_checkpoint.pt` - Latest checkpoint (609MB)

**All checkpoints contain:**
- `model_state_dict` - Model weights
- `epoch` - Training epoch
- `best_val_loss`, `best_val_map` - Performance metrics

---

## Export Validation

**After export, verify the bundle:**

```python
import json
from pathlib import Path

bundle_path = Path("maxsight_ios_bundle")

# Check all files exist
assert (bundle_path / "maxsight.pte").exists() or (bundle_path / "maxsight_traced.pt").exists()
assert (bundle_path / "model_config.json").exists()
assert (bundle_path / "runtime_config.json").exists()
assert (bundle_path / "processing_reference.py").exists()
assert (bundle_path / "README_XCODE.md").exists()

# Check config is valid JSON
with open(bundle_path / "model_config.json") as f:
    config = json.load(f)
    print(f"Input size: {config['input_size']}")
    print(f"Model params: {config['model_params']:,}")
    print(f"Model size: {config['model_size_mb']:.1f} MB")

print("✅ Bundle is valid")
```

---

## What Gets Exported

**Everything needed for Xcode (except web simulator):**

✅ **Model**: Trained weights in `.pte` or `.pt` format  
✅ **Config**: Model parameters, thresholds, output shapes  
✅ **Runtime config**: Settings for iOS app  
✅ **Processing reference**: Preprocessing/postprocessing code to port to Swift  
✅ **Documentation**: Complete iOS integration guide  

**Not included (stays in Python/web simulator):**
- Web simulator code (`tools/simulation/`)
- Training scripts
- Data pipeline code
- Colab-specific scripts

---

## Next Steps After Export

1. **Copy bundle to Xcode project:**
   - Drag `maxsight.pte` into Xcode project
   - Add to "Copy Bundle Resources"

2. **Install ExecuTorch framework:**
   - Add via Swift Package Manager
   - See `README_XCODE.md` for details

3. **Port preprocessing:**
   - Use `processing_reference.py` as reference
   - Implement in Swift (see README_XCODE.md)

4. **Load and run model:**
   - Follow examples in `README_XCODE.md`
   - Test with sample images

---

## Troubleshooting

### ExecuTorch Not Installed

**Install:**
```bash
pip install executorch
```

**Or use JIT fallback:**
- `export_ios_bundle` will create `maxsight_traced.pt` if ExecuTorch unavailable
- Can use PyTorch Mobile instead

### CoreML Export Fails

**Install CoreML tools:**
```bash
pip install coremltools
```

**Or use ExecuTorch/JIT instead** - CoreML is optional

### Model Output Shape Mismatch

**Check model config:**
- Verify `output_shapes` in `model_config.json`
- Ensure Swift code matches these shapes
- See `README_XCODE.md` for output parsing

### Preprocessing Differences

**Ensure exact match:**
- Image size: 224x224
- Normalization: ImageNet mean/std (see above)
- Color space: RGB
- Batch dimension: [1, 3, 224, 224]

---

## Complete Export Script

**Save as `export_for_xcode.py`:**

```python
#!/usr/bin/env python3
"""Export MaxSight model for Xcode integration."""
import sys
from pathlib import Path
import torch
from ml.models.maxsight_cnn import create_model
from ml.training.export import export_ios_bundle

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_model.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "maxsight_ios_bundle"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()
    
    print(f"Exporting to: {output_dir}")
    bundle_path = export_ios_bundle(
        model=model,
        output_dir=output_dir,
        input_size=(1, 3, 224, 224)
    )
    
    print(f"✅ Export complete: {bundle_path}")
    print(f"\nNext steps:")
    print(f"1. Copy {bundle_path} to your Xcode project")
    print(f"2. Follow instructions in {bundle_path}/README_XCODE.md")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python export_for_xcode.py checkpoints/final_model.pt maxsight_ios_bundle
```
