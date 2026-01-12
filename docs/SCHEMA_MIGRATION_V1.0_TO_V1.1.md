# Schema Migration Guide: v1.0 → v1.1

**Critical Schema Updates for Production Safety**

This guide documents the migration from schema v1.0 to v1.1, including breaking changes, semantic clarifications, and safety improvements.

---

## Overview

Schema v1.1 addresses critical issues identified in stress testing:

1. **Semantic De-duplication** - Clear separation of scene vs user-specific fields
2. **Provenance Tracking** - Confidence sources for graceful degradation
3. **Payload Discipline** - References instead of embedded images
4. **Safety Gating** - Required output_validity for all action-oriented outputs

---

## Breaking Changes

### 1. Field Name Changes (Semantic Clarity)

**Scene Analysis Fields:**
- `contrast_sensitivity` → `scene_contrast_demand`
- `glare_risk_level` → `scene_glare_risk`
- `motion_perception_difficulty` → `scene_motion_difficulty`

**Functional Vision Fields:**
- `contrast_sensitivity` → `user_contrast_capacity`
- `glare_risk_level` → `user_glare_sensitivity`
- `motion_perception_difficulty` → `user_motion_capacity`

**Rationale:** Clear distinction between environment-driven (scene) and user-specific (functional) metrics.

### 2. Embedded Images → References

**Spatial Mapping:**
- `semantic_segmentation` (base64) → `semantic_segmentation_ref` (string ID)

**Scene Analysis:**
- `hazard_density_heatmap` (base64) → `hazard_density_heatmap_ref` (string ID)

**Rationale:** Reduce payload size, enable lazy loading, improve performance.

### 3. Required Output Validity

**New Requirement:**
- `output_recommendations.output_validity` is now **REQUIRED** when `output_recommendations` is present

**Rationale:** Safety gating - prevents actions when confidence is low or uncertainty is high.

---

## New Fields

### 1. Confidence Sources (Detections)

```json
{
  "detections": [{
    "confidence": 0.85,
    "confidence_sources": {
      "detection": 0.92,
      "distance": 0.63,
      "accessibility": 0.48,
      "urgency": 0.75
    }
  }]
}
```

**Purpose:** Enable graceful degradation when specific heads fail.

### 2. Output Validity (Output Recommendations)

```json
{
  "output_recommendations": {
    "output_validity": {
      "confidence": 0.72,
      "safe_to_act": true,
      "uncertainty": 0.35,
      "degraded_modes": []
    }
  }
}
```

**Purpose:** Safety gating - prevents unsafe actions.

---

## Migration Steps

### Step 1: Update Field Names

**Before (v1.0):**
```json
{
  "scene_analysis": {
    "contrast_sensitivity": 0.7,
    "glare_risk_level": 2
  },
  "functional_vision": {
    "contrast_sensitivity": 0.5
  }
}
```

**After (v1.1):**
```json
{
  "scene_analysis": {
    "scene_contrast_demand": 0.7,
    "scene_glare_risk": 2
  },
  "functional_vision": {
    "user_contrast_capacity": 0.5
  }
}
```

### Step 2: Replace Embedded Images with References

**Before (v1.0):**
```json
{
  "scene_analysis": {
    "hazard_density_heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "spatial_mapping": {
    "semantic_segmentation": "iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

**After (v1.1):**
```json
{
  "scene_analysis": {
    "hazard_density_heatmap_ref": "frame_123_heatmap"
  },
  "spatial_mapping": {
    "semantic_segmentation_ref": "frame_123_segmentation"
  }
}
```

**Implementation:** Store images separately and return reference IDs. Client requests images via separate endpoint.

### Step 3: Add Confidence Sources

**Before (v1.0):**
```json
{
  "detections": [{
    "confidence": 0.85
  }]
}
```

**After (v1.1):**
```json
{
  "detections": [{
    "confidence": 0.85,
    "confidence_sources": {
      "detection": 0.92,
      "distance": 0.63,
      "accessibility": 0.48,
      "urgency": 0.75
    }
  }]
}
```

**Implementation:** Track per-head confidence scores during inference.

### Step 4: Add Output Validity

**Before (v1.0):**
```json
{
  "output_recommendations": {
    "audio": {
      "spatial_beacons": [...]
    }
  }
}
```

**After (v1.1):**
```json
{
  "output_recommendations": {
    "output_validity": {
      "confidence": 0.72,
      "safe_to_act": true,
      "uncertainty": 0.35,
      "degraded_modes": []
    },
    "audio": {
      "spatial_beacons": [...]
    }
  }
}
```

**Implementation:** Compute aggregated confidence and uncertainty. Set `safe_to_act` based on safety rules.

---

## Safety Rules (Enforced in v1.1)

### Rule 1: Confidence Threshold

**If `confidence < 0.5`, then `safe_to_act = false`**

```python
if output_validity['confidence'] < 0.5:
    output_validity['safe_to_act'] = False
```

### Rule 2: Uncertainty Threshold

**If `uncertainty > 0.7`, then `safe_to_act = false`**

```python
if output_validity['uncertainty'] > 0.7:
    output_validity['safe_to_act'] = False
```

### Rule 3: High Urgency + Low Confidence

**If `urgency >= 2` and `confidence < 0.5`, then `safe_to_act = false`**

```python
for det in detections:
    if det['urgency'] >= 2 and det['confidence'] < 0.5:
        output_validity['safe_to_act'] = False
        break
```

---

## Backward Compatibility

### Option 1: Strict Mode (Recommended)

**Reject v1.0 fields, require v1.1 fields.**

```python
validator = SchemaValidator(strict=True)
is_valid, errors = validator.validate(outputs)
```

**Use when:** Production deployment, safety-critical applications.

### Option 2: Permissive Mode (Migration Period)

**Accept v1.0 fields, warn about deprecated fields.**

```python
validator = SchemaValidator(strict=False)
is_valid, errors = validator.validate(outputs)
```

**Use when:** Migration period, gradual rollout.

### Option 3: Auto-Migration

**Automatically convert v1.0 → v1.1.**

```python
from ml.utils.schema_migration import migrate_v1_0_to_v1_1

outputs_v1_1 = migrate_v1_0_to_v1_1(outputs_v1_0)
```

**Use when:** Legacy system integration.

---

## Validation and Testing

### Run Schema Stress Tests

```python
from ml.utils.schema_validator import SchemaStressTester

tester = SchemaStressTester()
results = tester.run_all_tests(outputs, outputs_sequence)

report = tester.generate_report(results)
```

### Validate Outputs

```python
from ml.utils.schema_validator import validate_and_downgrade

outputs, is_valid, errors = validate_and_downgrade(
    outputs,
    strict=True,
    auto_downgrade=True
)
```

---

## Checklist

- [ ] Update field names (scene_* vs user_*)
- [ ] Replace embedded images with references
- [ ] Add confidence_sources to detections
- [ ] Add output_validity to output_recommendations
- [ ] Implement safety rules
- [ ] Run schema stress tests
- [ ] Update client code to handle references
- [ ] Update documentation
- [ ] Deploy with permissive mode first
- [ ] Switch to strict mode after validation

---

## Timeline

**Week 1:** Update model outputs to v1.1 format  
**Week 2:** Deploy with permissive validation  
**Week 3:** Run stress tests, fix issues  
**Week 4:** Switch to strict validation  
**Week 5:** Full production deployment  

---

## Support

For questions or issues during migration:
- Review `docs/SCHEMA_STRESS_TESTS.md`
- Check `ml/utils/schema_validator.py` for validation logic
- Run stress tests: Use `SchemaStressTester` from `ml.utils.schema_validator`

---

**Last Updated:** 2024  
**Status:** Active Migration  
**Next Review:** After v1.1 deployment

