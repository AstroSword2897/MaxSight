# MaxSight Architecture Reference

**Status:** authoritative reference as of 2026-11-07 (stack `ds/12`…`ds/17`).
**Audience:** engineers working on MaxSight, the AWS preparation team, and reviewers.
**Scope:** every input the system accepts, how each component uses it, and the integrity and security
controls around it. For the per-component registry see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md);
for design rationale see the [v1 specification series](v1/00-overview.md).

Every statement here was checked against the code at the commit above. Where the code and an older
document disagree, the code wins and the older document has been corrected or flagged in
[§13](#13-known-gaps-and-open-decisions).

---

## Contents

1. [What MaxSight is](#1-what-maxsight-is)
2. [System map](#2-system-map)
3. [Inputs: runtime](#3-inputs-runtime)
4. [Inputs: model artifacts and keys](#4-inputs-model-artifacts-and-keys)
5. [Inputs: training data](#5-inputs-training-data)
6. [Inputs: configuration files](#6-inputs-configuration-files)
7. [Inputs: environment variables and secrets](#7-inputs-environment-variables-and-secrets)
8. [Inputs: command-line entry points](#8-inputs-command-line-entry-points)
9. [Model architecture](#9-model-architecture)
10. [Data pipeline](#10-data-pipeline)
11. [Training, checkpoints and export](#11-training-checkpoints-and-export)
12. [Integrity and security controls](#12-integrity-and-security-controls)
13. [Known gaps and open decisions](#13-known-gaps-and-open-decisions)
14. [CI/CD and quality gates](#14-cicd-and-quality-gates)
15. [AWS preparation checklist](#15-aws-preparation-checklist)
16. [Change log for this revision](#16-change-log-for-this-revision)
17. [Related documentation](#17-related-documentation)

---

## 1. What MaxSight is

MaxSight is an assistive, multi-task vision system for people with low vision. A camera frame goes
through **Stage A**, a fail-closed hazard path that must answer within its latency budget. Optional
**Stage B** enrichment adds scene description, retrieval, therapy and personalization. Output reaches the
wearer as haptic, voice or overlay cues.

**Design invariants** (enforced in code and tests):

- Hazard authority belongs to Stage A only. Retrieval and therapy are advisory and can never override
  or suppress a hazard.
- When the hazard path cannot be trusted (no verified model, failed smoke test), the device says so
  (`HAZARD_UNAVAILABLE_MESSAGE`) rather than guessing.
- Inference never runs on partly random weights: checkpoints load all-or-nothing.
- The same image never appears in two data splits.

MaxSight is **not** a clinical diagnostic tool, a cloud captioning API, or a system where advisory
components can override safety output.

## 2. System map

```
 Camera / video                Audio (MFCC)              User + condition mode
       |                            |                            |
       v                            v                            v
 +-----------+   CameraFrame   +------------------------------------------+
 | app/ (L1) |---------------->| Stage A  ml/runtime/stage_a  (on device)  |
 | client    |                 |  resolve_model_handle: ACTIVE->LKG->REFUSED|
 | haptics   |<--HazardResult--|  verify signature -> strict load -> infer |
 | voice     |                 +------------------------------------------+
 | overlays  |                                  | (optional, budget permitting)
 +-----------+                                  v
       |                         +------------------------------------------+
       |  RuntimeRequest         | Stage B  app/stage_b + MaxSightCNN Stage B |
       +------------------------>|  hybrid ViT, temporal, scene graph, heads  |
         /v1/runtime/process     |  retrieval (advisory) / therapy (advisory) |
                                 +------------------------------------------+

 Offline:  sources.yaml -> acquisition -> cleaning -> splits -> audit -> split manifest
           -> gold shards (sha256) -> training (run_config YAML) -> checkpoint (atomic)
           -> certification (safety gates) -> signing (Ed25519 .sig.json) -> OTA / SageMaker
```

| Layer | Code | Responsibility |
|---|---|---|
| L1 client | `app/` | connectivity, Stage B client, OTA model update, haptics/voice/overlays, personal mode |
| L2 contracts | `docs/contracts/`, `tools/simulation/` | OpenAPI, JSON Schemas, web simulator (dev) |
| L3 runtime | `ml/runtime/`, `ml/evaluation/` | Stage A types, runner, model resolution, signature verification, tiers, safety gates |
| L4 domain | `ml/models/`, `ml/therapy/`, `ml/retrieval/` | MaxSightCNN and heads, therapy engine, retrieval |
| L5–L6 data | `ml/datasets/`, `ml/data/`, `ml/training/` | acquisition, cleaning, splits, gold artifacts, training, checkpoints |
| L7 external | AWS SageMaker, S3, Redis (optional), Hugging Face | training/hosting, storage, cache, encoders |

## 3. Inputs: runtime

### 3.1 Camera frame (Stage A)

`ml/runtime/stage_a/types.py`

| Field | Type | Meaning |
|---|---|---|
| `image` | `np.ndarray` `[H, W, 3]` uint8 RGB (grey `[H, W]` and RGBA are accepted and converted) | the frame |
| `frame_id` | `str` | caller's frame identifier |
| `timestamp` | `float` | capture time, carried to `HazardResult.timestamp_source` |

Preprocessing (`ml/runtime/stage_a/preprocess.py`): nearest-neighbour resize to 224×224, divide by 255,
ImageNet mean/std normalisation, NCHW float32. The frame is assumed to be uint8. See
[§13](#13-known-gaps-and-open-decisions) for the missing dtype check and the bilinear-vs-nearest
train/serve difference.

**Output, `HazardResult`:** `event_type`, `urgency` (0–3), `direction`, `distance_zone`
(near/medium/far), `confidence`, `uncertainty`, `latency_ms`, `model_version`, `model_hash`,
`condition_mode`, `timestamp_source`, `timestamp_emit`, `distance_meters` (optional).

### 3.2 Model forward inputs (`MaxSightCNN.forward`)

| Argument | Shape / type | Notes |
|---|---|---|
| `images` | `[B, 3, 224, 224]` float, ImageNet-normalised; or `[B, T, 3, H, W]` for temporal mode | required |
| `audio_features` | MFCC `[B, 13, T]` (dataset emits `[1, 13, T]` per sample) | optional; needs `use_audio=True` |
| `condition_tensor` | one-hot `[B, 14]` | optional; see 3.3 |
| `user_id` | `LongTensor [B]`, index into `user_embeddings` (10 000 rows) | optional |
| `use_temporal`, `prev_temporal_state`, `frame_id` | flags / cache keys | optional |

Outputs (dict, tier-dependent): `boxes`, `classifications`, `objectness`, `urgency`, `urgency_scores`,
`distance_zones`, `direction`, `depth_map`, `uncertainty`, `uncertainty_score`, `global_confidence`,
`motion`, `temporal_consistency`, `text_regions`, `colors`, `flicker`, `scene_description`,
`description`, `description_logits`, `scene_graphs`, `spatial_relations`, `semantic_relations`,
`central_priority`, `peripheral_priority`, `object_findability`, `personalization`,
`user_object_similarity`, `sound_classifications`, `sound_probs`, `sound_direction`, `sound_urgency`,
`stage_a_completed`, `stage_a_latency_ms`, `stage_b_completed`, `skip_stage_b_reason`.
Contract: `docs/contracts/schemas/model_output.json`.

### 3.3 Condition modes

`ml/runtime_constants.py`, contract `docs/contracts/schemas/condition_tensor.json` (width 14).

| id | mode | id | mode |
|---|---|---|---|
| 0 | none | 7 | amblyopia |
| 1 | glaucoma | 8 | strabismus |
| 2 | amd | 9 | color_blindness |
| 3 | cataracts | 10 | myopia |
| 4 | diabetic_retinopathy | 11 | hyperopia |
| 5 | retinitis_pigmentosa | 12 | presbyopia |
| 6 | cvi | 13 | astigmatism |

### 3.4 Production runtime API

`docs/contracts/openapi.yaml`

| Endpoint | Input | Output |
|---|---|---|
| `POST /v1/runtime/process` | `RuntimeRequest`: `frame_id` (req), `perception` (req, object), `user_id`, `tier` (bronze/silver/gold), `enable_rag`, `enable_therapy` | `RuntimeResponse` (`runtime_response.json`): `frame_id`, `tier`, `degraded_mode`, `classifications`, `critical_events`, `secondary_events`, `therapy`, `latency_ms`; `503` when the critical path is unavailable |
| `GET /health` | – | health and degraded mode |

### 3.5 SageMaker endpoint

`ml/infra/inference_handler.py` (`model_fn`, `input_fn`, `predict_fn`, `output_fn`)

- **Model directory:** `best.pt` (required: the endpoint refuses to start without it, and the checkpoint
  must match the model exactly) and `model_meta.json` (`tier`, default `T2_DETECTOR`).
- **Content types:** `application/json` with one of `image_b64`, `image_bytes`, or `ping`; or
  `application/octet-stream` (raw image bytes). Anything else raises `UnsupportedContentTypeError`.
- **Preprocessing:** decode → RGB → 224×224 → ImageNet normalisation.
- **Output:** `InferenceOutput`. Downstream must check `output_policy_applied` and `suppressed` before
  using `therapy_feedback`.

### 3.6 Web simulator (development only)

`tools/simulation/web_simulator.py` (Flask): `/`, `/api/init`, `/api/process`, `/api/scenarios`,
`/api/conditions`, `/api/stats`, `/api/session/{start,stop,status,abort}`, `/api/mode`, `/api/health`,
`/api/sample`, `/api/metrics`, `/api/dev/sprint-self-tests`, `/api/dev/validate-manifest`.

### 3.7 Session tokens

`ml/auth/token.py`: HMAC-SHA256 signed tokens with expiry. The signing key comes from
`MAXSIGHT_SECRET_KEY`, read on every call, and must be at least 32 bytes. There is no default. The module
has no callers yet.

## 4. Inputs: model artifacts and keys

| Artifact | Format | Producer | Consumer | Rules |
|---|---|---|---|---|
| Training checkpoint | torch file, dict with `model_state_dict` (+ `epoch`, `config`, …) | `ml/training/checkpoint_io.write_atomic_torch` | resume, export, inference | atomic write; `read_checkpoint` raises on missing/corrupt/no-weights |
| Bare state dict | torch file, `{param: tensor}` | quantization, stress tests, fusion training | `load_model_weights` | accepted by the inference loader |
| Stage A artifact | `.pt` / `.pth` | build / OTA | `TorchStageARunner` | any other suffix is refused |
| Signature instance | `<artifact>.sig.json` | `ml/infra/artifact_signing.sign_artifact` | `ml/runtime/stage_a/signature_contract` | must satisfy `artifact_signature.json`; see §12.3 |
| Certification manifest | JSON, `model_certification_manifest.json` | `scripts/infra/run_safety_gate_ci.py` | signer | only `all_passed: true` with every cell `passed` is signed |
| Verification public keys | PEM `current.ed25519.pub`, `next.ed25519.pub` | `scripts/infra/generate_signing_keypair.py` | devices | public only; nothing secret in `ml/runtime/stage_a/keys/` |
| Signing private key | PKCS#8 PEM Ed25519 | key ceremony, secret store | build signer | never in the repo; passed by path in `MAXSIGHT_SIGNING_KEY_FILE` |
| LKG fixture | `fixtures/last_known_good/weights.stub` | repo | resolution | a stub, not weights: with strict loading it is refused, so the device reports REFUSED |
| Reproducibility manifest | JSON with `checkpoint_hash`, `checkpoint_hash_scheme`, `config_hash`, seed | train loop | audit | hash scheme `sha256-v2:key,dtype,shape,bytes` |

**Inference-side loading** goes through `ml/training/checkpoint_io.load_model_weights`. It accepts wrapped
or bare state dicts and checks every key and shape **before** copying anything. So a refused checkpoint
leaves the model untouched; plain `nn.Module.load_state_dict` copies the matching tensors before it
reports an error. The callers are the Stage A runner, SageMaker `model_fn`, `ml/training/export.py`,
`scripts/ops/{deploy_top7, export_for_xcode, run_checkpoint_inference, run_inference_on_inference_datasets}.py`
and `scripts/product/run.py`. Only two places load with `strict=False`, both on purpose: training resume
(it logs missing and unexpected keys) and transfer learning.

## 5. Inputs: training data

### 5.1 Sources

`configs/data/sources.yaml`: `coco2017`, `open_images_v7`, `mapillary_vistas_v2`, `bdd100k`, `sanpo`,
`diode`, `pie`, `ego4d`, each with licence, URL and approval status. Licence terms are tracked in
[licence-register.md](../data-governance/licence-register.md). MaxSight is research use
(NonCommercial data allowed), and per-image licences are kept so a commercial export stays possible.

### 5.2 Detector label space

`configs/data/label_map.yaml`, 17 classes: person, bicycle, motorcycle, vehicle, animal, chair, couch,
table, bench, potted_plant, bag_box, trash_bin, pole_bollard, fire_hydrant, traffic_cone, door, stairs.
Source names not listed are dropped; names under `ignore` become ignore regions. The hazard taxonomy
lives in `configs/hazard/taxonomy.yaml` and [hazard-taxonomy.md](../contracts/hazard-taxonomy.md).

### 5.3 Manifest record (`ml/datasets/manifest.py`, `ImageRecord`)

| Field | Meaning |
|---|---|
| `image_id`, `source`, `path`, `width`, `height` | identity and location |
| `group_id` | recording session / source video; splits assign whole groups |
| `boxes` (`BoxRecord`: `cls`, `bbox_xyxy`, `source_class`), `ignore_regions` | labels |
| `sha256`, `dhash` | exact and perceptual content hashes, set by cleaning, used for de-duplication and leakage checks |
| `split` | train / val / test / test_holdout |
| `is_synthetic`, `annotation_source` | provenance; ground-truth gates accept human-reviewed, non-synthetic records only |
| `license`, `licence_id`, `dataset_version`, `label_map_version`, `url`, `extra` (`environment`, …) | governance |

### 5.4 Gold artifacts (`ml/data/gold/`)

- **Meta** (`load_gold_meta`, `validate_meta`): `meta_schema_version`, `line_schema_version`,
  `label_space`, `num_classes`, `class_map_hash`, `num_samples`, `shards[{uri, num_lines, sha256}]`,
  `built_at`, plus provenance `dataset_id`, `version`, `split`.
- **Shard lines (JSONL):** `schema_version`, `image_path`, `boxes`, `labels`, `metadata`.
- **Integrity:** training verifies every shard against its meta `sha256` (§12.2) and refuses shards the
  meta does not list.

### 5.5 Legacy annotation formats (`ml/data/`)

`ml/data/coco_dataset_splitter.py` reads COCO `instances_*.json` and writes either COCO splits or the
MaxSight list format: `image_id`, `image_path`, `objects[{box: [cx, cy, w, h] normalised, category,
urgency, distance, confidence}]`, `urgency`, `lighting`, `description`, `audio_path`. It is still used by
`scripts/ops/gather_training_data.py` and `scripts/ops/medallion_build.py`. It requires `image_dir`
and groups byte-identical images so they share a split.

## 6. Inputs: configuration files

| File | Read by | Purpose |
|---|---|---|
| `configs/data/sources.yaml` | `ml/datasets/sources.py`, `scripts/acquire_data.py` | dataset sources, licences, approval |
| `configs/data/label_map.yaml` | `ml/datasets/label_map.py` | 17-class detector space and source mapping |
| `configs/hazard/taxonomy.yaml` | tests only (`tests/unit/test_hazard_taxonomy.py`, contract tests); draft pending ticket 301 approval | hazard classes and criticality |
| `configs/security/artifact_signing.yaml` | `ml/runtime/stage_a/signature_contract.load_policy` | signing policy (§12.3) |
| `ml/training/configs/t*.yaml` | `ml/training/run_config.py` → `runner.py` | training runs: `model`, `data`, `training`, `distributed`, `loss`, `validation`, `checkpoint`, `logging`, `target_metrics`, `dataset`, `provenance` |
| `ml/training/configs/registry/{datasets,label_spaces}.yaml` | `ml/data/dataset_registry.py` | dataset registry and label spaces |
| `ml/config/tiers/{bronze,silver,gold}.yaml` | `ml/runtime/tier_router.py` | compute tiers by device budget (never by user condition) |
| `ml/config/safety_gates.yaml` | `ml/evaluation/safety_gates.py` | release thresholds (hazard recall ≥ 0.95, false-safe ≤ 0.01, direction ≥ 0.90, distance-zone ≥ 0.85, time-to-alert p95 ≤ 80 ms, ≤ 12 alerts/min); platforms torch_ref, onnx, coreml; SG-01/02 need hazard ground truth |
| `ml/config/assistive_supervision.yaml` | `ml/data/assistive_supervision.py` | distance/urgency supervision rules |
| `ml/config/rag_slo.yaml` | `ml/retrieval/rag_reliability.py` | retrieval SLOs |
| `ml/config/therapy_constraints.yaml` | `ml/therapy/constraints_loader.py` | therapy safety constraints |
| `scripts/product/deprecations.yaml` | `scripts/infra/pre_sagemaker_gate.py`, `scripts/infra/validate_runtime_contracts.py` | deprecated entry points |
| `docs/contracts/openapi.yaml`, `docs/contracts/schemas/*.json` | contract tests | API and data contracts |
| `environment.yml`, `requirements*.txt`, `requirements-ci/*.txt`, `pyproject.toml` | installers, CI | dependencies |

Training YAMLs and `TierConfig.for_tier` presets must agree on feature flags. Training builds the model
from the YAML (`TierConfig.from_dict`), while SageMaker inference builds it from the preset named in
`model_meta.json`. If they disagree, the strict loader refuses the checkpoint.

## 7. Inputs: environment variables and secrets

Secrets are marked **secret**. Never commit them, and in AWS use Secrets Manager or parameter injection.

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `MAXSIGHT_SIGNING_KEY_FILE` | – (**secret path**) | `ml/infra/artifact_signing.py` (named in the signing policy) | path to the Ed25519 private key; without it nothing is signed |
| `MAXSIGHT_SIGNING_POLICY` | `configs/security/artifact_signing.yaml` | `signature_contract.load_policy` | alternate policy file |
| `MAXSIGHT_SIGNING_MODE` | `local` | `artifact_signing.py` | `kms` is refused (devices verify Ed25519 only) |
| `MAXSIGHT_SECRET_KEY` | – (**secret**, ≥ 32 bytes) | `ml/auth/token.py` | session token key |
| `MAXSIGHT_SESSION_TIMEOUT` | `3600` | `ml/auth/token.py` | token TTL, seconds |
| `HF_TOKEN` | – (**secret**) | `ml/retrieval/encoders/global_encoder.py` | Hugging Face downloads |
| `AWS_DEFAULT_REGION` | `us-east-1` | sagemaker utils, run_config | AWS region |
| `SAGEMAKER_ROLE_ARN` | – | sagemaker utils, run_config, deploy | execution role |
| `MAXSIGHT_SKIP_ROLE_ASSERT` | – | sagemaker utils | skip role check (dev only) |
| `MAXSIGHT_S3_BUCKET` / `MAXSIGHT_S3_PREFIX` | `maxsight-ml` / `maxsight` | sagemaker utils, deploy, processing | artifact storage |
| `MAXSIGHT_MODEL_PACKAGE_GROUP` | – | `ml/infra/model_registry.py` | SageMaker model registry group |
| `MAXSIGHT_ENV` | – | `scripts/ops/sagemaker_deploy.py` | deployment environment |
| `SM_TRAIN_INSTANCE` / `SM_INFER_INSTANCE` / `SM_MAX_HOURS` | `ml.g5.2xlarge` / `ml.g5.xlarge` / `48` | sagemaker utils | job sizing |
| `SM_VOLUME_KMS_KEY_ID` | – | sagemaker utils | volume encryption key |
| `SM_MODEL_DIR`, `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VAL`, `SM_OUTPUT_DATA_DIR` | set by SageMaker | `sagemaker_entry.py`, `sagemaker_config.py` | container paths |
| `CHECKPOINTS_BASE` | – | `deploy_top7.py`, `find_trained_checkpoints.py` | checkpoint root |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE` | `0`, `0`, `1` | `ml/training/distributed.py` | distributed training |
| `REDIS_URL` | `redis://localhost:6379/0` | `ml/cache/redis_cache.py` | optional cache |
| `MAXSIGHT_RUNTIME` | `simulator` | `ml/runtime/mode.py` | runtime mode |
| `MAXSIGHT_HAPTIC_BACKEND` | `auto` | `app/ui/haptic_backends.py` | haptics backend |
| `MAXSIGHT_ASSISTIVE_SPEC_PATH` | bundled | `ml/data/assistive_supervision.py` | alternate supervision spec |
| `MAXSIGHT_INFRA_STRICT_PLACEHOLDERS` | – | `scripts/infra/validate_infra_stubs.py` | fail on infra placeholders |
| `DEBUG`, `DEBUG_LEVEL` | `0`, `off` | `ml/middleware/error_sanitizer.py` | error detail in responses (keep off in prod) |
| `HTTPS` | – | `ml/middleware/security_headers.py` | HSTS headers |
| `IMAGE_DIR` | required | `scripts/ops/find_annotation_images.py` | image root |
| Simulator: `FLASK_ENV`, `MAXSIGHT_PORT`, `MAXSIGHT_STRICT_PORT`, `MAXSIGHT_CHECKPOINT_PATH`, `MAXSIGHT_CORS_ORIGINS`, `MAXSIGHT_BEHIND_PROXY`, `MAXSIGHT_ENABLE_HAPTICS_STUB`, `MAXSIGHT_ALLOW_LEGACY_PICKLE` | various | `tools/simulation/*` | development harness only |

## 8. Inputs: command-line entry points

**Product CLI**, `scripts/product/run.py`: `validate`, `train`, `transfer`, `smoke`, `gate`, `certify`
(`--sign --artifact <model.pt>` signs an all-pass model; needs `MAXSIGHT_SIGNING_KEY_FILE`), `export`,
`package`.

**Data CLIs:**
- `scripts/acquire_data.py coco|…`: class-capped acquisition with a byte budget.
- `scripts/validate_data.py`: the pre-training quality gate, wrapping clean, split and audit. Options:
  `--manifest`, `--root`, `--clean`, `--split`, `--holdout`, `--max-class-share`, `--out`, `--report`,
  `--dataset-id`.
- `scripts/split_manifest.py create|check`: immutable, versioned split manifests.
- `ml/data/coco_dataset_splitter.py` (legacy; `--image_dir` required).

**Security:** `scripts/infra/generate_signing_keypair.py --slot current|next --private-out <outside repo>`;
`scripts/infra/run_safety_gate_ci.py` (certification).

**Operations (`scripts/ops/`):** dataset building (`build_gold_manifest`, `build_video_manifest`,
`build_pseudo_panoptic_manifest`, `medallion_build`, `ingest_datasets`, `gather_training_data`,
`clean_and_preprocess`, `download_*`, `sync_medallion_s3`, `patch_missing_images`), training
(`train_maxsight`, `train_from_gold_index`, `train_medallion_models`, `smoke_train`, `sagemaker_train`,
`sagemaker_processing_submit`), evaluation (`run_checkpoint_inference`,
`run_inference_on_inference_datasets`, `compare_condition_models`), deployment (`deploy_top7`,
`export_for_xcode`, `sagemaker_deploy`, `create_model_package_group`, `check_export_status`,
`cleanup_cloud_checkpoints`).

## 9. Model architecture

`ml/models/maxsight_cnn.py`. `create_model(num_classes, condition_mode, use_audio, fpn_channels=256,
tier_config)` defaults to T5_TEMPORAL.

### 9.1 Capability tiers (effective flags)

| tier | SE | CBAM | hybrid ViT | dynamic conv | cross-task | cross-modal | temporal | retrieval |
|---|---|---|---|---|---|---|---|---|
| T0_BASELINE_CNN | - | - | - | - | - | - | - | - |
| T1_LIGHTWEIGHT | on | on | - | - | - | - | - | - |
| T2_DETECTOR (SageMaker default) | on | on | - | - | - | - | - | - |
| T2_HYBRID_VIT | on | on | on | - | - | - | - | - |
| T3_MULTI_TASK | on | on | on | - | on | - | - | - |
| T4_ADVANCED | on | on | on | - | on | on | - | - |
| T5_TEMPORAL (production target) | on | on | on | - | on | on | on | on |

### 9.2 Forward path

1. **Stage A:** ResNet-50 backbone (ImageNet weights) → FPN (P2–P5, `fpn_channels`) → optional SE/CBAM attention → fused
   features and scene context → hazard heads (urgency, distance, direction, uncertainty).
2. **Stage B** (`_forward_stage_b_backbone`), when enabled:
   - **Hybrid CNN-ViT** (`ml/models/backbone/hybrid_backbone.py`). It runs on the raw image, and its
     middle FPN level is added to the Stage A features (×0.3). The cross-layer projections take the FPN
     features (`cnn_out_channels`). A failure raises; there is no silent fallback.
   - **Dynamic convolution** (`ml/models/backbone/dynamic_conv.py`). A residual
     `stage_b + gate × DynamicConv2d(stage_b)` with the gate initialised to 0. It is registered at
     construction and **off in every tier**. To enable it, switch it on in the preset and the training
     YAML together, then retrain.
   - **Temporal encoder** (ConvLSTM / TimeSformer) for `[B, T, …]` input.
3. **Heads** (`ml/models/heads/`): detection, depth, motion, OCR/text regions, colour, flicker,
   priority, findability, contrast, fatigue, ROI priority, therapy state, personalization, sound.
4. **Cross-task / cross-modal attention** (`ml/models/attention/attention.py`, exported from
   `ml.models.attention`).
5. **Scene graph** (`ml/models/scene_graph/scene_graph_encoder.py`). Spatial predicates (left, right,
   above, below, near, far) plus semantic relations. `semantic_k` uses relation row `k`; rule predicates
   use a SHA-256-derived row, so rows are identical in every process.
6. **Retrieval** (`ml/retrieval/`): advisory only, asynchronous, and its failures never change outputs.

### 9.3 Personalization

`app/personal_mode.py` gives each user their own row in the fusion user embedding, assigned on first
use. When the table is full, a new user gets no user embedding rather than another user's row. Today
the rows are not trained per user.

## 10. Data pipeline

**Authoritative stack: `ml/datasets/`** (decision 2026-11-03, see
[DATA_REPOSITORY_AUDIT.md](../data-governance/DATA_REPOSITORY_AUDIT.md)).

```
acquire (coco, openimages, …)  ->  manifest (ImageRecord)
   -> clean (decode check, small boxes, exact sha256 + perceptual dhash de-duplication)
   -> splits.assign_splits (whole groups; holdout environments -> test_holdout)
   -> audit (class share, boxes in bounds, group leakage, same image across splits)
   -> split_manifest.build (refuses images shared across groups, records without hashes)
   -> gold artifacts (shards + meta sha256)  ->  training (verifies shard sha256)
```

**Leakage rules** (enforced; `ml/datasets/split_manifest.check_leakage` and `ml/datasets/audit.audit`):

- No group (session or video) in more than one split.
- A held-out environment appears only in `test_holdout`.
- No image in two splits: equal SHA-256, or dHash within the near-duplicate distance
  (`audit.content_duplicates`).
- Every record must carry `sha256` or `dhash`. Building refuses records without them; the audit warns.

**Legacy stack: `ml/data/`.** It is still used by two ops scripts (an open architecture decision). The
legacy COCO splitter now groups byte-identical images and refuses unreadable images. It has no
near-duplicate check.

## 11. Training, checkpoints and export

- **Configuration:** `ml/training/run_config.py` resolves a YAML into `ResolvedTrainingConfig`, with
  strict schema checks, dataset registry checks, and optional on-disk hash checks.
- **Loop:** `ml/training/train_loop.py`. Checkpoints are written atomically
  (`checkpoint_io.write_atomic_torch`). Resume raises on a missing or corrupt checkpoint. The best model
  gets a reproducibility manifest with a content hash under scheme `sha256-v2:key,dtype,shape,bytes`.
- **Gold loaders:** `ml/data/data_pipeline._create_gold_data_loaders_for_resolved` verifies shards
  whenever a meta is present.
- **Augmentation:** `ml/data/multi_modal_augment.py`. Images are uint8 or float in [0, 1], before
  mean/std normalisation (normalised floats are refused). Audio is a spectrogram or MFCC with
  `(freq, time)` dimensions; raw waveforms are refused.
- **Export:** `ml/training/export.py` (JIT, CoreML, ONNX, ExecuTorch, iOS bundle). `--checkpoint` must
  exist and match exactly; omitting it exports an explicitly logged untrained model.
- **Certification and signing:** `run.py certify` runs the safety gates. `--sign` signs only all-pass
  artifacts.

## 12. Integrity and security controls

### 12.1 Model loading and runtime

| Control | Where | Guarantee | Tests |
|---|---|---|---|
| All-or-nothing weight loading | `checkpoint_io.load_model_weights` | missing, unexpected or reshaped keys raise before any copy | `tests/test_strict_weight_loading.py` |
| Stage A refuses unloadable artifacts | `torch_runner.TorchStageARunner` | corrupt, stub, non-`.pt` or missing file → `CheckpointError`, so resolution → REFUSED | same, `test_torch_stage_a_infer.py` |
| SageMaker refuses missing or mismatched `best.pt` | `inference_handler.model_fn` | the endpoint never serves random weights | `test_strict_weight_loading.py` |
| Hybrid branch fails loudly | `maxsight_cnn._forward_stage_b_backbone` | no silent fallback, no unbounded state | `test_hybrid_backbone_integration.py`, `test_phase0_backbone.py` |
| Process-stable relation rows | `scene_graph_encoder._semantic_index` | no `PYTHONHASHSEED` dependence | `test_scene_graph_relation_index.py` |
| Per-user personalization rows | `personal_mode._user_row` | no row sharing between users | `test_personal_mode_user_rows.py` |

### 12.2 Data integrity

| Control | Where | Guarantee | Tests |
|---|---|---|---|
| Same image never in two splits | `audit.content_duplicates`, `split_manifest` | exact and near duplicates across splits/groups refused | `tests/unit/test_split_content_leakage.py` |
| Legacy splitter content grouping | `coco_dataset_splitter._content_groups` | byte-identical images share a split | `tests/unit/test_coco_splitter_content.py` |
| Gold shard verification in training | `data_pipeline`, `gold/dataset._verify_shard_integrity` | tampered or unlisted shards refused | `tests/test_gold_shard_verification.py` |
| Checkpoint content hash | `reproducibility.checkpoint_content_hash` | dtype, shape and key framing hashed; bf16 supported | `tests/test_checkpoint_content_hash.py` |

### 12.3 Model artifact signing (instance contract + YAML policy)

- **Contract:** `docs/contracts/schemas/artifact_signature.json`. Each artifact `model.pt` ships
  `model.pt.sig.json` containing `schema_version`, `artifact_name`, `artifact_sha256`,
  `algorithm: ed25519`, `key_id` (16 hex of SHA-256 of the raw public key), `signed_at`,
  `certification` (the all-pass manifest) and `signature` (base64, 64 bytes). The signature covers the
  canonical JSON of all other fields, so editing any of them invalidates it.
- **Policy:** `configs/security/artifact_signing.yaml`, strictly validated: unknown keys or a non-Ed25519
  algorithm are refused.
  - `build.private_key_env` names the variable that holds the private-key **path**.
  - `device.public_keys_dir` and `device.trust_window` (`current.ed25519.pub`, `next.ed25519.pub`) are
    the keys devices trust.
  - `device.require_signature: true` refuses unsigned artifacts.
  - `device.unsigned_allowed: []` is the explicit dev-only exception list.
- **Device check** (`signature_contract.check_artifact_signature`), in order:
  1. The instance file exists, unless the policy allows the artifact unsigned.
  2. The instance satisfies the contract.
  3. `artifact_name` matches the file.
  4. The SHA-256 matches the file.
  5. The key is in the trust window.
  6. The Ed25519 signature verifies.

  With no public keys provisioned, every artifact is refused.
- **Build:** `sign_artifact` refuses non-all-pass manifests, a missing key (there is no random-key
  fallback), and KMS mode.
- **Provisioning and rotation:** see [keys/README.md](../../ml/runtime/stage_a/keys/README.md).
- **Tests:** `tests/test_artifact_signature_contract.py` (schema and runtime-validator agreement,
  tampering, untrusted key, moved instance, malformed JSON, no keys, policy validation, key script),
  `test_signature_in_resolution.py`, `test_artifact_signing.py`, `test_stage_a_key_bundle.py` (guards
  against committed secret material).

### 12.4 Other security controls

- **Session tokens:** no default secret; ≥ 32-byte key; constant-time comparison
  (`tests/test_auth_token_secret.py`).
- **Stage A isolation:** `scripts/infra/validate_stage_a_isolation.py` forbids networking, AWS and
  app-layer imports under `ml/runtime/stage_a/`.
- **Error sanitisation and security headers:** `ml/middleware/`.

## 13. Known gaps and open decisions

| Item | State | Impact | Owner action |
|---|---|---|---|
| Former HMAC keys still in git history | removed from the tree and no longer trusted; **the repository is public** | treat them as compromised | decide whether to rewrite history |
| Signing key custody | **CONFLICTED:** roadmap P5-06 / ADR-016 planned KMS-held ECDSA keys; this revision signs with an Ed25519 file key from the secret store | KMS custody is stronger (the private key never leaves KMS) | decide; the contract's `algorithm` field and the policy can add a KMS-backed algorithm later |
| No Ed25519 keys provisioned yet | devices refuse every signed-or-unsigned model until keys exist | expected fail-closed state | run the key ceremony (keys/README.md) and set `MAXSIGHT_SIGNING_KEY_FILE` in the build |
| On-device (Swift/iOS) verification | not in this repository (UNINSPECTED) | the iOS app must implement the same contract | port `check_artifact_signature` |
| No production caller of `resolve_model_handle` in this repository | library API only | integration lives outside the repo | confirm in the app |
| Hybrid tiers (T2_HYBRID_VIT, T3, T4, **T5**) | hybrid branch now active; kept as is by owner decision | checkpoints trained before the fix carry untrained hybrid weights | retrain or fine-tune when convenient |
| Legacy `ml/data` stack used by two ops scripts | content-checked, but no near-duplicate check | possible near-copy leakage on that path | migrate the scripts to `ml/datasets` |
| Stage A preprocess | no dtype check; nearest-neighbour vs bilinear in training | a float frame would be scaled twice; small train/serve skew | add a dtype check; align the resize |
| The test suite writes `tools/simulation/baseline_output.json` | a tracked file is modified by tests | dirty working trees after test runs | write to a temp path instead |
| Full suite on this revision | targeted suites pass; the full run was invalidated locally by a full disk | final verification pending | run the full suite in AWS (§15) |
| Documents linked but missing from the repository | README links `docs/DOWNLOAD_AND_START.md` (the setup guide), `docs/SYSTEMS.md`, `docs/medallion_data.md`, `docs/algorithmic_efficiency.md`, `docs/video_and_navigation_datasets.md`, `docs/video_panoptic_manifest.md`, `tools/simulation/README.md`, `TRAINING_RUNBOOK.md`, `PRE_TRAIN_CHECKLIST.md`, `TRAINING_SETUP_SUMMARY.md`, `WHAT_HAS_BEEN_DONE.md` | readers hit dead links, including setup instructions | restore the files or remove the links |
| `test_phase4_knowledge.py:125`, `test_export_temporal_smoke.py:62` | still skip on exceptions | may hide defects | convert to assertions |

## 14. CI/CD and quality gates

- **`ci.yml`:** `detect-changes`, `lint`, `stage-a-contracts` (Stage A, resolution, signing contract,
  isolation validator), `safety-gates`, `regression-guards`, `phone-app-layer`, `infra-iam`,
  `rag-reliability`, `drift-checks`, `torch-model-smoke`, `torch-condition-tensor`, `torch-timing`,
  `unit-tests-cpu`, `sagemaker-ops`, `ci-summary`.
- **`quality.yml`:** `quality-drift`, `ruff-tier1`, `mypy-tier1`, `pyright-tier1`, `therapy-coverage`,
  `pre-sagemaker-gate`.
- **`model-evaluation.yml`, `release-gates.yml`:** placeholders. They fail rather than pass green.
- **Reusable runner:** `reusable-python.yml` with profiles (e.g. `torch-cpu` →
  `requirements-ci/torch-cpu.txt`).

## 15. AWS preparation checklist

1. **Environment:** Python 3.11 with `requirements.txt`. It includes `faiss-cpu` and `cryptography`;
   on macOS hosts `tests/conftest.py` applies the OpenMP guard.
2. **Run the full suite** on an instance with enough disk (≥ 10 GB free): `pytest -q`. The last clean
   local baseline before this revision was 1301 passed, 5 skipped, 12 xfailed, 0 failed.
3. **Secrets** (Secrets Manager → environment): `MAXSIGHT_SIGNING_KEY_FILE` (a mounted file path),
   `MAXSIGHT_SECRET_KEY` if tokens are used, and `HF_TOKEN` if encoders download.
4. **Signing keys:** run `scripts/infra/generate_signing_keypair.py --slot current` in a trusted
   environment, store the private PEM in the secret store, and commit only `current.ed25519.pub`.
5. **SageMaker:** `SAGEMAKER_ROLE_ARN`, `MAXSIGHT_S3_BUCKET`, `MAXSIGHT_S3_PREFIX`, `AWS_DEFAULT_REGION`,
   `SM_VOLUME_KMS_KEY_ID`; the model package includes `best.pt` and `model_meta.json` with the `tier`.
6. **Data:** gold artifacts must ship their meta with shard `sha256`. Split manifests require `sha256`
   or `dhash` on every record.
7. **Checkpoints:** T2_HYBRID_VIT, T3, T4 and T5 checkpoints from before this revision load but carry
   untrained hybrid weights.

## 16. Change log for this revision

| PR / branch | Change |
|---|---|
| #35 `ds/12` | all-or-nothing inference loading (Stage A, SageMaker, export, ops scripts); checkpoint hash v2; Ed25519 signature contract + YAML policy; public HMAC keys removed |
| #38 `ds/15` | same-image-across-splits refusal (split manifest, audit); gold shard verification in training; legacy COCO splitter content grouping |
| #39 `ds/16` | hybrid backbone fix (FPN features, no swallow, no lazy adapter); DynamicConv2d conditions; scene-graph stable rows; per-user personalization rows; token secret; multi-modal augmentation fix; `use_dynamic_conv` builds a real layer (off everywhere); Phase 0/5 tests assert instead of skipping |
| #40 `ds/17` | this document and the documentation updates |

## 17. Related documentation

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md): component registry, five-questions method, topology.
- [v1 series](v1/00-overview.md): decision register, Stage A/B specifications, safety validation, roadmap.
- [docs/architecture.md](../architecture.md): short model overview.
- [Stage A keys and rotation](../../ml/runtime/stage_a/keys/README.md).
- [Data governance](../data-governance/): acquisition plan, dataset status, licence register, repository audit.
- [Ops](../ops/): AWS runbook, production remediation, repro gate, pre-integration checklist.
- [Productization](../productization/): scope and claims, release gates, runtime boundary, pilot protocol.
- [Contracts](../contracts/): OpenAPI, JSON Schemas, hazard taxonomy.
