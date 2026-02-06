# File Details Summary

## Overview

This document provides details about the requested files and directories. Some paths don't exist exactly as requested, but equivalent files are documented.

---

## 1. ml/authorization/ → ml/auth/token.py

**Status:** Path `ml/authorization/` doesn't exist, but `ml/auth/token.py` provides authorization functionality.

### File: `ml/auth/token.py`

**Purpose:** HMAC-signed session token generation and verification for secure session management.

**Key Features:**
- Stateless token system (no server-side storage needed)
- HMAC-SHA256 signature for tamper detection
- Built-in expiration time (default: 1 hour, configurable via `MAXSIGHT_SESSION_TIMEOUT`)
- URL-safe base64 encoding

**Functions:**

1. **`make_token(payload: Dict) -> str`**
   - Generates HMAC-signed token with expiration
   - Adds `exp` field (expiration timestamp)
   - Signs payload with secret key
   - Returns token in format: `payload.signature`

2. **`verify_token(token: str) -> Dict`**
   - Verifies HMAC signature
   - Checks expiration
   - Returns decoded payload if valid
   - Raises `ValueError` if invalid/expired/tampered

**Configuration:**
- `MAXSIGHT_SECRET_KEY` environment variable (default: "change_me_in_production")
- `MAXSIGHT_SESSION_TIMEOUT` environment variable (default: 3600 seconds)

**Security:**
- Uses `hmac.compare_digest()` for constant-time signature comparison (prevents timing attacks)
- Token tampering detection via signature verification
- Automatic expiration checking

---

## 2. ml/data/analysis/training_architecture.md

**Status:** File does not exist.

**Note:** No training architecture documentation found in `ml/data/analysis/`. Consider checking:
- `ml/training/configs/` for training configuration files
- `docs/` for architecture documentation
- `README.md` for high-level architecture overview

---

## 3. ml/data/evaluation/ → ml/evaluation/

**Status:** Path `ml/data/evaluation/` doesn't exist, but `ml/evaluation/` provides evaluation functionality.

### File: `ml/evaluation/metrics.py`

**Purpose:** Comprehensive evaluation metrics for Phase 9, including multi-modal, accessibility-specific, and robustness metrics.

**Key Classes:**

1. **`MultiModalMetrics` (dataclass)**
   - `vision_accuracy`: Vision modality accuracy
   - `audio_accuracy`: Audio modality accuracy
   - `haptic_accuracy`: Haptic modality accuracy
   - `fusion_improvement`: Improvement from fusion vs single modality
   - `cross_modal_alignment`: Alignment between modalities

2. **`AccessibilityMetrics` (dataclass)**
   - `detection_rate`: % of critical objects detected
   - `false_positive_rate`: False positive rate
   - `response_time_ms`: Response time in milliseconds
   - `navigation_success_rate`: Navigation success rate
   - `text_readability_score`: Text readability score
   - `scene_description_quality`: Scene description quality score

3. **`RobustnessMetrics` (dataclass)**
   - `lighting_robustness`: Performance across lighting conditions
   - `occlusion_robustness`: Performance with occlusions
   - `motion_robustness`: Performance with motion blur
   - `noise_robustness`: Performance with noise
   - `adversarial_robustness`: Performance against adversarial examples

4. **`EvaluationMetrics` (main class)**
   - `reset()`: Reset all metrics
   - `compute_multi_modal_metrics()`: Compute multi-modal metrics
   - `compute_accessibility_metrics()`: Compute accessibility-specific metrics
   - `compute_robustness_metrics()`: Compute robustness metrics
   - `generate_report()`: Generate comprehensive evaluation report

**Helper Methods:**
- `_iou_overlap()`: Compute IoU between two bounding boxes

### File: `ml/evaluation/__init__.py`

**Exports:**
- `EvaluationMetrics`
- `MultiModalMetrics`
- `AccessibilityMetrics`
- `RobustnessMetrics`

### File: `ml/training/evaluation.py`

**Purpose:** Evaluation report generator with lighting-aware metrics analysis.

**Key Functions:**

1. **`generate_evaluation_report(metrics: Dict, save_path: Optional[Path]) -> str`**
   - Generates text report with overall and lighting-stratified metrics
   - Includes performance degradation analysis
   - Checks against targets (≥85% recall, ≥75% dark recall, <10% degradation)

2. **`plot_lighting_metrics(metrics: Dict, save_path: Optional[Path])`**
   - Plots precision/recall comparison across lighting conditions
   - Creates bar charts for precision, recall, and F1 scores
   - Requires matplotlib

3. **`analyze_lighting_degradation(metrics: Dict, lighting_conditions: Optional[List]) -> Dict`**
   - Calculates degradation % for each lighting condition vs normal
   - Returns positive values for worse performance

4. **`export_metrics_json(metrics: Dict, save_path: Path)`**
   - Exports metrics to JSON format
   - Handles torch tensors (converts to float)

---

## 4. ml/data/inference_datasets.py

**Status:** File exists (737 lines)

**Purpose:** Inference dataset loaders for MaxSight, supporting multiple datasets for evaluation.

### Key Components:

#### **Standard Metadata Schema:**
```python
STANDARD_METADATA_KEYS = {
    'weather': None,
    'scene': None,
    'labels': None,
    'annotation_path': None,
    'label': None,
    'confidence': None
}
```

#### **Dataset Classes:**

1. **`OpenImagesV6Dataset`**
   - **Purpose:** Open Images V6 dataset loader
   - **Coverage:** 9M images with 600 object classes
   - **Features:**
     - Aggregates all labels per image (not just first)
     - Handles corrupted images (skip or dummy fallback)
     - Supports max_samples limit
     - Standard metadata schema
   - **Structure:** Images in subdirectories (first 2 chars of image_id)

2. **`BDD100KDataset`**
   - **Purpose:** BDD100K dataset loader (driving scenes)
   - **Features:**
     - Weather and scene attributes
     - JSON label files
     - Corrupted image handling
   - **Structure:** `images/100k/{split}/` and `labels/bdd100k_labels_images_{split}.json`

3. **`ADE20KDataset`**
   - **Purpose:** ADE20K dataset loader (scene parsing)
   - **Features:**
     - Annotation support (PNG masks)
     - Corrupted image handling
   - **Structure:** `images/{split}/` and `annotations/{split}/`

#### **Utility Functions:**

1. **`create_imagenet_transform() -> transforms.Compose`**
   - Creates ImageNet normalization transform
   - Resize to 224x224, normalize with ImageNet stats
   - Configurable for different backbones

2. **`DetectionPostProcessor`**
   - **Purpose:** Post-processor interface for model outputs
   - **Features:**
     - Handles different output formats
     - Batch processing support
     - Falls back to manual processing if `model.get_detections()` unavailable
   - **Parameters:**
     - `confidence_threshold`: Default 0.3
     - `max_detections`: Default 10
     - `nms_threshold`: Default 0.5

3. **`create_inference_dataloader()`**
   - Creates DataLoader for inference datasets
   - Supports: `open_images_v6`, `bdd100k`, `ade20k`
   - Auto-configures `pin_memory` based on CUDA availability

4. **`run_inference_on_dataset()`**
   - Runs MaxSight inference on inference dataset
   - **Features:**
     - Handles corrupted images
     - Uses postprocessor interface
     - Tracks statistics (total images, detections, corrupted images)
     - Global image index counter (handles variable batch sizes)
     - Non-blocking transfers if pin_memory enabled

**Key Fixes Applied:**
- Proper error handling for corrupted images
- Standard metadata schema across all datasets
- Label aggregation (not just first label)
- Global counter for image indexing
- Proper pin_memory handling

---

## 5. ml/data/security/ → ml/security/

**Status:** Path `ml/data/security/` doesn't exist, but `ml/security/` provides security functionality.

### File: `ml/security/validation.py`

**Purpose:** Input validation utilities for MaxSight.

**Functions:**

1. **`is_valid_b64(s: str) -> bool`**
   - Validates Base64 string format
   - Uses `base64.b64decode(validate=True)` for validation
   - Returns `True` if valid, `False` otherwise

2. **`decode_and_validate_image(base64_str: str, max_size_mb: int = 10, allowed_types: tuple = ...) -> tuple[bool, Optional[bytes], Optional[str]]`**
   - Decodes Base64 image and validates format and size
   - **Parameters:**
     - `base64_str`: Base64-encoded image string
     - `max_size_mb`: Maximum file size in MB (default: 10)
     - `allowed_types`: Tuple of allowed formats (default: jpg, png, gif, bmp, webp, tiff)
   - **Returns:** Tuple of (success, decoded_bytes, error_message)
   - **Validation:**
     - Base64 format check
     - File size check
     - Magic number validation (prevents malicious uploads)

### File: `ml/security/magic.py`

**Purpose:** File magic number detection for input validation.

**Magic Numbers Supported:**
- JPEG: `\xFF\xD8\xFF`
- PNG: `\x89PNG\r\n\x1A\n`
- GIF87a: `GIF87a`
- GIF89a: `GIF89a`
- BMP: `BM`
- WEBP: `RIFF` + `WEBP` at offset 8
- TIFF (little-endian): `II*\x00`
- TIFF (big-endian): `MM\x00*`

**Functions:**

1. **`detect_magic(file_bytes: bytes) -> Optional[str]`**
   - Detects file type from magic number (first few bytes)
   - Returns file type string (e.g., 'jpg', 'png') or `None`
   - Special handling for WEBP (requires both RIFF header and WEBP identifier)

2. **`validate_image_magic(file_bytes: bytes, allowed_types: tuple = ...) -> bool`**
   - Validates that file bytes match an allowed image type
   - Uses `detect_magic()` internally
   - Returns `True` if detected type is in allowed_types

**Security Features:**
- Prevents malicious file uploads by checking actual file signatures
- Not just file extension validation (can be spoofed)
- Validates first few bytes (magic numbers) which are harder to fake

---

## Summary

| Requested Path | Actual Path | Status |
|----------------|-------------|--------|
| `ml/authorization/` | `ml/auth/token.py` | ✅ Equivalent exists |
| `ml/data/analysis/training_architecture.md` | N/A | ❌ Does not exist |
| `ml/data/evaluation/` | `ml/evaluation/` | ✅ Equivalent exists |
| `ml/data/inference_datasets.py` | `ml/data/inference_datasets.py` | ✅ Exists |
| `ml/data/security/` | `ml/security/` | ✅ Equivalent exists |

---

## Key Takeaways

1. **Authorization:** Token-based authentication system using HMAC signatures
2. **Evaluation:** Comprehensive metrics for multi-modal, accessibility, and robustness evaluation
3. **Inference Datasets:** Support for OpenImagesV6, BDD100K, and ADE20K with standardized metadata
4. **Security:** Input validation with Base64 decoding and magic number detection to prevent malicious uploads
