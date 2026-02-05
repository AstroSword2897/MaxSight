# Supported Vision Conditions in MaxSight

MaxSight supports **14 vision condition modes** with specific adaptations for each.

---

## 🎯 Primary Conditions (Most Common)

### 1. **Glaucoma**
- **What**: Peripheral vision loss (tunnel vision)
- **Adaptation**: Emphasizes peripheral regions, wider field detection
- **Prevalence**: ~3% of adults over 40
- **Usage**: `--condition-mode glaucoma`

### 2. **AMD (Age-related Macular Degeneration)**
- **What**: Central vision loss
- **Adaptation**: Emphasizes peripheral context, larger targets
- **Prevalence**: ~8% of adults over 55
- **Usage**: `--condition-mode amd`

### 3. **Cataracts**
- **What**: Blurred/cloudy vision, glare sensitivity
- **Adaptation**: Contrast enhancement, glare reduction
- **Prevalence**: Most common (50%+ over 80)
- **Usage**: `--condition-mode cataracts`

### 4. **Color Blindness**
- **What**: Difficulty distinguishing colors (red-green, blue-yellow)
- **Adaptation**: Color detection and announcement, pattern-based identification
- **Prevalence**: ~8% of men, ~0.5% of women
- **Usage**: `--condition-mode color_blindness`

---

## 🔬 Progressive/Degenerative Conditions

### 5. **Diabetic Retinopathy**
- **What**: Blood vessel damage causing blurred vision, dark spots
- **Adaptation**: Handles inconsistent visual field, spotty vision patterns
- **Prevalence**: ~30% of people with diabetes
- **Usage**: `--condition-mode diabetic_retinopathy`

### 6. **Retinitis Pigmentosa (RP)**
- **What**: Progressive peripheral vision loss, night blindness
- **Adaptation**: Brightness enhancement, central focus with context
- **Prevalence**: ~1 in 4,000
- **Usage**: `--condition-mode retinitis_pigmentosa`

---

## 🧠 Neurological/Developmental Conditions

### 7. **CVI (Cortical/Cerebral Visual Impairment)**
- **What**: Brain-based vision processing issues (not eye-based)
- **Adaptation**: Handles inconsistent visual interpretation, simplified scenes
- **Prevalence**: Most common cause of childhood vision impairment
- **Usage**: `--condition-mode cvi`

### 8. **Amblyopia (Lazy Eye)**
- **What**: Reduced vision in one eye, poor depth perception
- **Adaptation**: Single-eye compensation, depth cues from motion/context
- **Prevalence**: ~2-3% of children
- **Usage**: `--condition-mode amblyopia`

### 9. **Strabismus (Crossed/Misaligned Eyes)**
- **What**: Eye misalignment causing double vision or suppression
- **Adaptation**: Handles inconsistent binocular input, depth from motion
- **Prevalence**: ~4% of children
- **Usage**: `--condition-mode strabismus`

---

## 👓 Refractive Errors (Extreme Cases)

### 10. **Refractive Errors (General)**
- **What**: Severe uncorrected blur (any refractive error)
- **Adaptation**: Edge enhancement, contrast boost, motion detection
- **Usage**: `--condition-mode refractive_errors`

### 11. **Myopia (Nearsightedness)**
- **What**: Severe distance blur
- **Adaptation**: Enhanced distant object detection, depth estimation
- **Prevalence**: ~30% globally (increasing)
- **Usage**: `--condition-mode myopia`

### 12. **Hyperopia (Farsightedness)**
- **What**: Severe near blur, reading difficulty
- **Adaptation**: Enhanced near-field detection, text recognition
- **Prevalence**: ~25% of adults
- **Usage**: `--condition-mode hyperopia`

### 13. **Astigmatism**
- **What**: Distorted/blurred vision at all distances
- **Adaptation**: Multi-scale edge detection, distortion correction
- **Prevalence**: ~33% of people
- **Usage**: `--condition-mode astigmatism`

### 14. **Presbyopia**
- **What**: Age-related near vision loss (reading difficulty)
- **Adaptation**: Enhanced text/label detection, OCR support
- **Prevalence**: Nearly universal over 50
- **Usage**: `--condition-mode presbyopia`

---

## 🔄 Adaptation Mechanisms

### How the Model Adapts

| Condition | Visual Emphasis | Processing | Output |
|-----------|----------------|------------|--------|
| **Glaucoma** | Peripheral regions | Wider detection cone | Full-field awareness |
| **AMD** | Center context | Peripheral cues | Scene understanding |
| **Cataracts** | Contrast/edges | Dehazing, sharpening | Clearer boundaries |
| **Color Blindness** | Patterns/textures | Color naming | Non-color identification |
| **Diabetic Ret.** | Consistent areas | Spotty-field handling | Filled-in percept |
| **RP** | Central + brightness | Night enhancement | Navigable view |
| **CVI** | Simplified scenes | Reduced complexity | Interpretable output |
| **Amblyopia** | Dominant eye | Single-eye depth | 3D from motion |
| **Strabismus** | Stable view | Suppression handling | Unified percept |
| **Refractive** | Edges/motion | Blur-invariant features | Recognizable objects |

---

## 📊 Usage Statistics

**Most requested conditions** (order of implementation priority):
1. Cataracts (50%+ over 80) - blur, glare
2. AMD (8% over 55) - central loss
3. Glaucoma (3% over 40) - peripheral loss
4. Color blindness (8% men) - color confusion
5. Diabetic retinopathy (30% with diabetes) - spotty vision

**Fastest growing**:
- Myopia (pandemic + screen time)
- Diabetic retinopathy (diabetes epidemic)
- CVI (improved neonatal survival)

---

## 🧪 Training with Condition Modes

### Basic Usage
```bash
# Train for single condition
python scripts/train_maxsight.py \
    --data-dir /path/to/data \
    --condition-mode glaucoma \
    --epochs 20
```

### Multi-Condition Training
```bash
# Train separate models for each condition
for condition in glaucoma amd cataracts color_blindness; do
    python scripts/train_maxsight.py \
        --data-dir /path/to/data \
        --condition-mode $condition \
        --checkpoint-dir checkpoints_${condition} \
        --epochs 20
done
```

### Condition-Agnostic (Default)
```bash
# No condition mode = general model (works for all)
python scripts/train_maxsight.py \
    --data-dir /path/to/data \
    --epochs 20
```

---

## 🔍 Testing Condition Adaptations

### Verify a Condition Mode
```python
from ml.models.maxsight_cnn import create_model
import torch

# Create condition-specific model
model = create_model(condition_mode='glaucoma')

# Test with dummy input
x = torch.randn(1, 3, 224, 224)
outputs = model(x)

print(f"Model adapted for: {model.condition_mode}")
print(f"Outputs: {list(outputs.keys())}")
```

### Compare Condition Outputs
```python
conditions = ['glaucoma', 'amd', 'cataracts', 'color_blindness']
for cond in conditions:
    model = create_model(condition_mode=cond)
    outputs = model(torch.randn(1, 3, 224, 224))
    print(f"{cond}: {outputs['boxes'].shape}, urgency: {outputs['urgency_scores'].mean():.3f}")
```

---

## 📚 References

### Clinical Definitions
- **Glaucoma**: Optic nerve damage, often from high eye pressure
- **AMD**: Deterioration of macula (central retina)
- **Cataracts**: Clouding of eye's natural lens
- **Diabetic Retinopathy**: Diabetes-related retinal blood vessel damage
- **RP**: Genetic progressive retinal degeneration
- **CVI**: Brain's inability to interpret visual information
- **Amblyopia**: Brain favors one eye, suppresses the other
- **Strabismus**: Eyes don't align properly

### Adaptation Strategy
Each condition mode:
1. **Preprocessing**: Simulates the condition's visual effects for training data
2. **Attention**: Adjusts spatial attention weights (peripheral vs. central)
3. **Feature extraction**: Emphasizes condition-relevant features
4. **Output formatting**: Customizes alerts/descriptions for the condition

---

## 🆕 Adding New Conditions

To add a new vision condition:

1. **Define preprocessing** in `ml/utils/preprocessing.py`:
   ```python
   def apply_<condition>_transform(image):
       # Simulate visual effects
       return transformed_image
   ```

2. **Add condition logic** in `maxsight_cnn.py` `__init__`:
   ```python
   if condition_mode == '<new_condition>':
       self.<condition>_adapter = ConditionAdapter()
   ```

3. **Add to training script** `--condition-mode` choices

4. **Document** in this file with prevalence and adaptation strategy

---

## 💡 Condition Mode Best Practices

### When to Use Condition Modes
- ✅ **Training**: Train separate models for high-prevalence conditions (cataracts, AMD, glaucoma)
- ✅ **User profiles**: Load condition-specific model based on user's diagnosis
- ✅ **A/B testing**: Compare condition-adapted vs. general model performance

### When NOT to Use
- ❌ Don't train separate models for every condition (14 models = expensive)
- ❌ Don't use condition mode if diagnosis is uncertain
- ❌ Don't use for mild/corrected conditions (glasses-corrected myopia)

### Recommended Strategy
1. **General model** (no condition mode) - works for most users
2. **"Big 4" models** - glaucoma, AMD, cataracts, color_blindness
3. **On-demand** - train other conditions as needed for specific user groups

---

**All 14 conditions fully supported in the model!** 🎯
