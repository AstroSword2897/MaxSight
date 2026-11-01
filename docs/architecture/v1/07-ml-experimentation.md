<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D7: ML Experimentation Plan

### D7.0 Model-development dependency chain (each arrow is a gate)

```
Taxonomy (DR-202) ─▶ Annotation spec (guide v1, κ target) ─▶ Collection (D7.4) ─▶ Dataset audit (gate_pretrain)
   ─▶ Training (Stage 1 → Stage 2) ─▶ Evaluation (pipeline replay, D7.9) ─▶ Mobile export (Core ML, parity)
   ─▶ Device benchmarking (L2/L5/L7, thermal) ─▶ Selection (DR-041) ─▶ Release (D8)
```

| Gate | Pass condition | If it fails |
|---|---|---|
| Taxonomy | ADR for DR-202 signed | stay on default classes; no collection beyond pilot clips |
| Annotation spec | guide piloted on 20 clips with κ ≥ 0.7 | revise definitions before scaling |
| Dataset audit | no leakage, no class > 15%, ≥ 30 relevant events per class in val+test | collect more for the failing classes |
| Training | val event recall improves over the COCO-only baseline | adjust augmentation or replay; check labels |
| Evaluation | report complete with CIs; failure cases reviewed | — |
| Export | Δbox ≤ 1e-3 rel., hazard-state agreement ≥ 99% vs PyTorch | move NMS/ops to Swift; change export path |
| Device | L5 within PR-001 frame interval; 30-min thermal without THERMAL_REDUCED on the reference device | lower input size or rate, or INT8 (DR-116), or the next candidate |

**Model identifier reconciliation:**

| Name | Meaning |
|---|---|
| "SSD-MobileNet" (DR-011) | the decision family |
| **M-SSD** | its V1 implementation: SSDLite320 head on a MobileNetV2 backbone (the SSD-MobileNetV2 family's mobile variant) |
| M-SSD-REF | Google's TF `ssd_mobilenet_v2_320x320` model, used **only** as a latency reference |
| `MaxSightDetector-SSDMBv2-<semver>.mlpackage` | the shipped artifact, with its manifest `model_id = "m-ssd"` |

The same IDs are used in D1, D3, D7, D9 and in manifests.

**Coverage warning:**
- COCO/Open Images pretraining gives **no** reliable curb or drop-off detection, and only limited stair coverage.
- Environmental-category recall is reported **separately** and comes only from MaxSight Hazard v1.
- If box detection proves unsuitable for curbs and drop-offs (thin, elongated edges), a segmentation head is a **new decision**, not a silent change.

### D7.1 Candidates (exact variants)

| ID | Variant | Source / weights | Input | Role |
|---|---|---|---|---|
| M-SSD | **SSDLite320-MobileNetV2** (torchvision `SSD` + `SSDLiteHead`, MobileNetV2 `IMAGENET1K_V2` backbone, C4 tap at block-13 expansion 576 ch, extras 512/256/256/128) | built in-repo; trained by us | 320×320 | **Initial benchmark (DR-011)** |
| M-SSD-REF | TF Model Zoo `ssd_mobilenet_v2_320x320_coco17_tpu-8` (.tflite) | Google | 320 | latency reference only (Android, D8) |
| M-R50 | **RetinaNet-ResNet50-FPN-v2** (torchvision, COCO weights) | torchvision | short side 640 | ResNet50+FPN comparison (DR-012) |
| M-R50-MS | MaxSightCNN Stage A wrapper (ResNet50+FPN + tier-1 heads, no condition input) | `[E] ml/models/maxsight_cnn.py` | 224 | research comparison |
| M-EDL0 | **EfficientDet-Lite0** (`effdet` `tf_efficientdet_lite0`, COCO weights) | effdet | 320 | comparison (DR-012) |
| M-SSD-FPN | M-SSD + FPNLite (torchvision `FeaturePyramidNetwork`, 128 ch) | built in-repo | 320 | test whether multi-scale fusion earns its cost |

### D7.2 Taxonomy (DR-202 Open; default 17 detector classes)

`person, bicycle, motorcycle, vehicle, animal, chair, couch, table, bench, potted_plant, bag_box, trash_bin, pole_bollard, fire_hydrant, traffic_cone, door, stairs` (+ `curb` once labelled).

Hazard categories come from D3.4.

### D7.3 Data sources and handling

| Source | Use | Classes | Licence | Rule |
|---|---|---|---|---|
| COCO 2017 | pretrain | person, bicycle, motorcycle, car/bus/truck→vehicle, dog/cat→animal, chair, couch, dining table, bench, potted plant, suitcase/backpack/handbag→bag_box, fire hydrant | CC BY 4.0 ann. | pretrain only |
| Open Images V7 | pretrain | Door, Stairs, Waste container, Traffic cone, Box | CC BY 4.0 / CC BY 2.0 | ≤4,000 images per class (`[E] scripts/download_open_images_fiftyone.py`) |
| Mapillary Vistas v2 | pretrain + curb (later) | pole, bollard → pole_bollard; curb | **CC BY-NC-SA 4.0** | dev and research only; replace before commercial release |
| BDD100K | eval-only robustness (night/rain) | person, rider, vehicle | research licence | never training |
| **MaxSight Hazard v1** | fine-tune + **all hazard evaluation** | all + hazard attributes | own | faces blurred before storage |
| `[E] datasets/{train,val,test}` (synthetic) | **retired** to `datasets/synthetic_legacy/` | — | — | smoke tests only (DR-043) |

**Cleaning rules:**
- drop boxes < 8 px on the short side
- drop images without mapped classes (keep 5% as negatives)
- dedupe by perceptual hash (pHash distance ≤ 4)
- mark unmapped probable obstacles as `ignore` regions
- cap at 8,000 instances per class
- no class > 15% (DR-113)
- per-source class histograms go to `class_report.md`

**Annotation (MaxSight Hazard v1, Q88–90):**
- CVAT video tracks at 5 fps (keyframes every 1 s, interpolated).
- Attributes: visibility, hazard_relevance (none/possible/relevant), category, `distance_gt_m` at tape keyframes.
- A guide with 25 example frames.
- 10% double annotation, Cohen's κ ≥ 0.7 *(exp.)*, disagreements resolved by a reviewer.

### D7.4 Capture plan (data volume; hardware per D9.4)

| Item | Value |
|---|---|
| Environments | 5 (home hallway, doorway room, office corridor, lobby, sidewalk); lobby held out |
| Lighting | bright and dim |
| Per environment per lighting | 30 static approaches (10 classes × L/C/R), 6 dynamic (supervised), 4 environmental (stairs/curbs), 10 context (pass-by, doors), 6 empty |
| Totals | ≈ 560 clips, ≈ 400 relevant events, ≈ 42k labelled frames |
| Distance ground truth | tape-measured floor marks at 1/2/3 m annotated at keyframes (no additional sensors, DR-017) |
| Mount | chest, height and pitch measured per session |

**Statistics:** ≥598 independent events per class are needed to claim ≥99.5% recall at 95% confidence with 0 misses. This dataset is a **development set**; certification-scale capture continues after V1 model selection.

### D7.5 Splits and leakage prevention (DR-042)

- Split by **clip → session → environment**, 70/15/15, plus a held-out environment.
- `splits.json` carries a content hash.
- `gate_pretrain.py` asserts:
  - no clip, session or environment overlap
  - no near-duplicate frames across splits (pHash)
  - class balance
  - κ threshold
- Pilot data (later) is split by participant.

### D7.6 Training configuration

| | Stage 1 (pretrain) | Stage 2 (fine-tune) |
|---|---|---|
| Data | harmonised mix (COCO + OI + Vistas, balanced) | Hazard v1 train + 20% mix replay |
| M-SSD | SGD m=0.9, lr 0.08 @ bs 64, warm-up 5, cosine, wd 4e-5, 120 epochs | AdamW head 1e-3 / backbone 1e-4, wd 1e-4, 40 epochs |
| M-R50 / M-EDL0 | from COCO weights; 30 epochs mix | same as M-SSD Stage 2 |
| Augment | PhotometricDistort, ZoomOut, IoUCrop, HFlip, SanitizeBoxes | + gamma 0.4–1.0, motion blur k 3–9 |
| Compute | SageMaker `ml.g5.xlarge` (`[E] scripts/ops/sagemaker_train.py`) | Mac MPS or g5 |
| Selection | COCO-val subset mAP@0.5 | **val relevant-event recall**, tie-break mAP |
| Reproducibility | seed, git SHA, lockfile, dataset and split hashes, hardware in `manifest.json` (`[E] ml/training/reproducibility.py`) | |

**Calibration:** temperature scaling on val → `T` stored in the model package; ECE + reliability diagram reported.

### D7.7 Export, quantisation and runtime compatibility

- **Export:** `torch.jit.trace(backbone+head)` → `coremltools.convert(..., ImageType(320×320), minimum_deployment_target=iOS17)`. Anchor decode and NMS run in Swift (MaxSightCore), golden-tested.
- **Variants:** FP32, FP16 (`compute_precision=FLOAT16`), INT8 (`coremltools.optimize` linear quantisation with 500 Hazard v1 calibration frames).
- **Gate:** adopt the lowest precision whose per-class relevant recall drops ≤ 1 pt vs FP32, with hazard-state agreement ≥ 99% on golden clips (DR-116).
- **Android (later):** `ai-edge-torch` → `.tflite` FP16/INT8, plus the M-SSD-REF latency comparison.

### D7.8 Spatial-estimation experiment (DR-203)

- **Methods (camera-only):** ground plane with the validity monitor (D3.6.2) vs Depth-Anything-V2-Small (vendored, Core ML) scale-aligned to the ground plane. Both are run on walking clips (not only static) to quantify motion effects.
- **Data:** tape keyframes (≥720).
- **Metrics:** MAE, median AE and bias per 0.5–1/1–2/2–4 m bucket × lighting; valid-estimate rate; σ coverage (68%); added latency on device.
- **Decision rule:** choose the simplest method meeting SG-04 thresholds once approved; otherwise distances stay zones or unknown.

### D7.9 Evaluation protocol and selection procedure

Every candidate gets an identical input contract and test split, and is evaluated as the **full Stage A pipeline replay** (not the detector alone) for hazard metrics.

| Group | Metrics |
|---|---|
| Model | precision, recall, mAP@0.5, per-class AP, localisation IoU distribution |
| Module | IDF1 + ID switches (`motmetrics`), spatial MAE, ECE |
| Hazard | event recall per category, false-negative rate, false alerts/min, FPR |
| Runtime | p50/p95/p99 inference and camera-to-alert on the reference iPhone, memory, energy (Xcode Energy Log, mAh/10 min), thermal state over 30 min |
| Robustness | per-stratum results (lighting, clutter, motion, occlusion) + BDD100K night subset |

**Statistics:** 95% CIs by 2,000-resample cluster bootstrap over clips; `insufficient_evidence` when n is below `sample_size.py`.

**Selection** (`ml/evaluation/decision_matrix.py`):
1. **Hard filter.** Reject any candidate whose lower CI of relevant-event recall is below the floor **in any hazard category**, or whose p95 camera-to-alert exceeds budget, or that fails export parity.
2. **Weighted score** for survivors: recall 30, latency 20, power/thermal 15, small-object 10, robustness 10, memory 5, runtime compatibility 5, maintainability 5.
3. **Review:** you approve; the result is recorded in ADR-007.

**Regression:** every new model runs the fixed benchmark plus golden clips; it is promoted only if no category regresses beyond its CI and every gate stays passed (Q100).
