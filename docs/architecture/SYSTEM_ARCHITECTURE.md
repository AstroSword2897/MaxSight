# MaxSight System Architecture

MaxSight is an assistive multi-task vision runtime for on-device hazard awareness and adaptive assistance across clinical vision-condition modes. It combines a shared backbone and feature pyramid with task heads, a fail-closed Stage A hazard path, optional Stage B / retrieval / therapy enrichment, and CI-enforced safety certification.

**MaxSight is not:** a cloud-only chat or captioning API; a system in which retrieval or therapy may override hazard authority; a municipal work-order or payments product; or a substitute for clinical diagnosis.

---

## Table of contents

1. [Current Implementation Mapping](#3-current-implementation-mapping)
2. [Component Registry by layer](#4-component-registry-by-layer)
3. [Five Questions Method](#5-five-questions-method)
4. [Display / UX](#6-display--ux)
5. [Platform overview](#7-platform-overview)
6. [Tech stack](#8-tech-stack)
7. [Runtime topology and graceful degradation](#9-runtime-topology--graceful-degradation)
8. [Background jobs](#10-background-jobs)
9. [Key data flows](#11-key-data-flows)
10. [Auth, roles and access control](#12-auth-roles--access-control)
11. [Architectural principles](#13-architectural-principles)
12. [Related documentation](#14-related-documentation)
13. [Testing](#15-testing-architecture-relevant)
14. [Schema / migration checklist](#16-schema--migration-checklist)

---

## 3. Current Implementation Mapping

| Architecture component | Current implementation | Status |
|---|---|---|
| Stage A types + runner protocol | `ml/runtime/stage_a/types.py` (`CameraFrame`, `HazardResult`, `StageARunner`) | shipped |
| Model handle resolution | `ml/runtime/stage_a/model_handle.py` (`ACTIVE` → `LAST_KNOWN_GOOD` → `REFUSED`) | shipped |
| Torch Stage A runner | `ml/runtime/stage_a/torch_runner.py` | shipped |
| Artifact signing / hash verify | `ml/infra/artifact_signing.py`, `ml/runtime/stage_a/verify.py` | shipped |
| Runtime contracts / tiers / degraded modes | `ml/runtime/contracts.py`, `ml/runtime/tier_router.py` | shipped |
| Condition tensor contract | `ml/runtime_constants.py` (`CONDITION_MODE_IDS`, width 14), `docs/contracts/schemas/condition_tensor.json` | shipped |
| MaxSightCNN + heads | `ml/models/maxsight_cnn.py`, `ml/models/heads/*`, `ml/models/backbone/` | shipped |
| Therapy engine + safety | `ml/therapy/therapy_engine.py`, `ml/therapy/therapy_safety.py` | shipped |
| Retrieval / RAG reliability | `ml/retrieval/*`, `ml/retrieval/rag_reliability.py`, `ml/config/rag_slo.yaml` | shipped (advisory) |
| Safety gates + cert manifest | `ml/evaluation/safety_gates.py`, `ml/config/safety_gates.yaml`, `scripts/infra/run_safety_gate_ci.py` | shipped (fail-closed without hazard GT) |
| Phone connectivity | `app/connectivity/monitor.py` | shipped |
| Stage B client | `app/stage_b/client.py`, `app/stage_b/messages.py` | shipped |
| OTA model update | `app/model_update/{downloader,staging,activation,storage}.py` | shipped |
| Haptics / voice / overlays | `app/ui/*`, `app/overlays/overlay_engine.py` | shipped |
| Personal mode | `app/personal_mode.py` | shipped |
| Training loop + export | `ml/training/train_loop.py`, `ml/training/export.py` | shipped (export targets optional) |
| SageMaker / IAM stubs | `scripts/ops/sagemaker_*.py`, `infra/iam/*` | partial (dry-run / stubs) |
| Web simulator | `tools/simulation/` | shipped (dev harness) |
| Contract schemas | `docs/contracts/schemas/*.json`, `docs/contracts/openapi.yaml` | shipped |
| iOS native shell | `ios/` | planned / thin |
| Quality baseline docs tree | `docs/quality/` | partial |
| Hazard-labeled GT for SG-01/02 | referenced by `safety_gates.yaml` | planned (cells blocked until present) |

---

## 4. Component Registry by layer

Every component uses the contract: **Purpose · Inputs · Outputs · Owns · Does not own · Dependencies · Files**.

### L1 — Client (`app/`)

#### Connectivity monitor

| | |
|---|---|
| **Purpose** | Track network reachability so Stage B and OTA can fail predictably while Stage A continues offline. |
| **What it does** | Runs a connectivity state machine from OS/network probes and timers. |
| **Why it exists** | Hazard inference must not depend on the network; enrichment and downloads must. |
| **What it is used for** | Gating Stage B RPC attempts and OTA download starts. |
| **Inputs** | Probe results, retry/backoff timers. |
| **Outputs** | Connectivity state transitions consumed by Stage B client and model-update downloader. |
| **Owns** | Connectivity state and transitions. |
| **Does not own** | Hazard inference, model weights, certification. |
| **Dependencies** | Platform network APIs (device). |
| **Files** | `app/connectivity/monitor.py`, `app/connectivity/__init__.py` |

#### Stage B client

| | |
|---|---|
| **Purpose** | Best-effort enrichment channel to secondary compute without blocking Stage A. |
| **What it does** | Sends local frame metadata under a timeout budget when connectivity allows; accepts preemptible secondary payloads. |
| **Why it exists** | Scene/OCR/therapy context is useful but must never become CriticalEvent authority. |
| **What it is used for** | Optional secondary events on connected devices. |
| **Inputs** | Frame metadata, connectivity state, timeout budget; message types in `app/stage_b/messages.py`. |
| **Outputs** | Preemptible secondary payloads (`SecondaryEvent`-aligned content). |
| **Owns** | Stage B request/response client behavior and timeouts. |
| **Does not own** | `HazardResult` / Stage A. |
| **Dependencies** | Connectivity monitor; optional remote Stage B endpoint. |
| **Files** | `app/stage_b/client.py`, `app/stage_b/messages.py` |

#### OTA / model update

| | |
|---|---|
| **Purpose** | Download → stage → activate / rollback signed model artifacts on device. |
| **What it does** | Fetches artifact, stages it, activates after verification, or rolls back to last-known-good. |
| **Why it exists** | Ship improved weights without bricking the hazard path. |
| **What it is used for** | Pilot/release model rollout on phone. |
| **Inputs** | Signed artifact URI, expected hash, staging directory. |
| **Outputs** | Active model path for handle resolution, or rollback to last-known-good. |
| **Owns** | Download, staging, activation, local storage layout for updates. |
| **Does not own** | Training, safety-gate math, Stage A inference. |
| **Dependencies** | Connectivity; artifact signing/hash verify; model handle resolution. |
| **Files** | `app/model_update/downloader.py`, `staging.py`, `activation.py`, `storage.py` |

#### Haptics / voice / overlays

| | |
|---|---|
| **Purpose** | Render urgency, direction, and distance to non-visual (and visual) channels. |
| **What it does** | Maps hazard fields to haptic patterns, TTS, and overlay draw calls. |
| **Why it exists** | Wearers need hazard cues without relying on fine visual detail. |
| **What it is used for** | Every Stage A emit and scheduled secondary cues. |
| **Inputs** | `HazardResult` / `CriticalEvent`: `urgency: int`, `direction: str`, `distance_zone: str`, confidence/uncertainty. |
| **Outputs** | Device haptic patterns, TTS utterances, overlay draw calls. |
| **Owns** | Channel rendering and pattern selection. |
| **Does not own** | Model inference or gate thresholds. |
| **Dependencies** | Stage A (or orchestrator) event stream; haptic backends. |
| **Files** | `app/ui/haptic_feedback.py`, `haptic_backends.py`, `hazard_haptics.py`, `voice_feedback.py`, `app/overlays/overlay_engine.py` |

#### Personal mode

| | |
|---|---|
| **Purpose** | Per-user fusion/output preference scaling without changing the Stage A wire contract. |
| **What it does** | Adjusts soft personalization weights from wearer preference state (torch-backed when present). |
| **Why it exists** | Cue intensity and fusion preference vary by wearer. |
| **What it is used for** | Soft personalization of secondary/fusion paths. |
| **Inputs** | User profile / preference state. |
| **Outputs** | Adjusted weights/scaling factors. |
| **Owns** | Personalization state application. |
| **Does not own** | Safety gate thresholds (`HAZARD_RECALL_MIN`, etc.) or Stage A types. |
| **Dependencies** | Optional torch; fusion helpers under `ml/retrieval/fusion/`. |
| **Files** | `app/personal_mode.py` |

---

### L2 — Contracts / edge

#### OpenAPI + JSON schemas

| | |
|---|---|
| **Purpose** | Freeze wire shapes shared by phone, simulator, and CI validators. |
| **What it does** | Declares schemas for runtime response, model output, condition tensor, certification manifest, RAG reliability. |
| **Why it exists** | Prevent silent contract drift across packages. |
| **What it is used for** | `scripts/infra/validate_runtime_contracts.py`, certification schema tests. |
| **Inputs** | Schema documents (see Files). |
| **Outputs** | Validation pass/fail; no runtime tensors. |
| **Owns** | Documented wire contracts. |
| **Does not own** | Inference implementation. |
| **Dependencies** | jsonschema / YAML tooling in CI. |
| **Files** | `docs/contracts/openapi.yaml`, `docs/contracts/schemas/runtime_response.json`, `model_output.json`, `condition_tensor.json`, `model_certification_manifest.json`, `rag_reliability.json` |

#### Web simulator

| | |
|---|---|
| **Purpose** | Local HTTP/UI harness for frame injection and regression baselines. |
| **What it does** | Accepts uploaded frames/session config; runs phone-adjacent pipelines; may snapshot baselines. |
| **Why it exists** | Exercise app-adjacent paths without a physical device. |
| **What it is used for** | Dev/pilot debugging; selected unit tests. |
| **Inputs** | Uploaded frames, session config. |
| **Outputs** | JSON results; optional baseline JSON snapshots. |
| **Owns** | Simulator session UX and harness wiring. |
| **Does not own** | Stage A isolation invariants (enforced under `ml/runtime/stage_a/`). |
| **Dependencies** | Flask stack (sim), ML packages as configured. |
| **Files** | `tools/simulation/` |

---

### L3 — Runtime services

#### Stage A types + runner protocol

| | |
|---|---|
| **Purpose** | Hard-isolate safety-critical inference from connectivity. |
| **What it does** | Defines frozen `CameraFrame` / `HazardResult` and `StageARunner.infer(frame) -> HazardResult` with **no** network parameters. |
| **Why it exists** | On-device hazard path must remain callable when offline. |
| **What it is used for** | Every on-device hazard tick; Stage A CI contract tests. |
| **Inputs** | `CameraFrame(image: np.ndarray, frame_id: str, timestamp: float)`. |
| **Outputs** | `HazardResult(event_type, urgency, direction, distance_zone, confidence, uncertainty, latency_ms, model_version, model_hash, condition_mode, timestamp_source, timestamp_emit, distance_meters?)`. |
| **Owns** | Stage A wire types and protocol. |
| **Does not own** | Training, OTA download, therapy decisions. |
| **Dependencies** | NumPy for frame buffer. |
| **Files** | `ml/runtime/stage_a/types.py` |

#### Model handle resolution

| | |
|---|---|
| **Purpose** | Resolve a trustworthy on-device model for Stage A. |
| **What it does** | State machine `ACTIVE → LAST_KNOWN_GOOD → REFUSED` with hash/signature and optional smoke probe. |
| **Why it exists** | Never run missing or untrusted weights for hazards. |
| **What it is used for** | Startup and post-OTA activation. |
| **Inputs** | Artifact paths, expected hashes, `SmokeProbe` protocol implementations. |
| **Outputs** | `ModelHandle`, `ResolutionState`; refused path surfaces `HAZARD_UNAVAILABLE_MESSAGE`. |
| **Owns** | Resolution policy and handle metadata. |
| **Does not own** | Training; **must not** import `MaxSightCNN` here. |
| **Dependencies** | Artifact signing/verify helpers. |
| **Files** | `ml/runtime/stage_a/model_handle.py`, `ml/runtime/stage_a/messages.py`, `ml/runtime/stage_a/verify.py`, `ml/infra/artifact_signing.py` |

#### Torch Stage A runner

| | |
|---|---|
| **Purpose** | Canonical Python Stage A inference for CI and reference devices. |
| **What it does** | Preprocesses `CameraFrame`, runs local torch artifact, maps outputs to `HazardResult` including zone/direction names. |
| **Why it exists** | Executable Stage A behind the protocol for `torch_ref` certification. |
| **What it is used for** | Stage A tests; `torch_ref` safety-gate platform. |
| **Inputs** | `CameraFrame`; resolved model path/handle. |
| **Outputs** | `HazardResult` with `distance_zone ∈ {near, medium, far}`, `direction ∈ {left, center, right}`. |
| **Owns** | Torch-backed Stage A adaption. |
| **Does not own** | Certification scoring; OTA. |
| **Dependencies** | torch; preprocess helpers. |
| **Files** | `ml/runtime/stage_a/torch_runner.py`, `ml/runtime/stage_a/preprocess.py` |

#### Runtime contracts / tiers / degraded modes

| | |
|---|---|
| **Purpose** | Shared event and health vocabulary across orchestrator, simulator, and device bridges. |
| **What it does** | Defines `CriticalEvent`, `SecondaryEvent`, `ComputeTier` (bronze/silver/gold), `DegradedMode` (D0–D3), MVP output filtering. |
| **Why it exists** | Prevent divergent event shapes and undefined overload behavior. |
| **What it is used for** | Event emission, tier routing, degradation policy. |
| **Inputs** | Model/perception dicts; tier YAML via `TierRouter`. |
| **Outputs** | Typed events; selected tier; filtered MVP keys (`classifications`, `boxes`, `objectness`, `urgency_scores`, `distance_zones`, …). |
| **Owns** | Contract dataclasses and tier routing rules. |
| **Does not own** | Head implementations. |
| **Dependencies** | `ml/runtime_constants.py` MVP keys. |
| **Files** | `ml/runtime/contracts.py`, `ml/runtime/tier_router.py`, `ml/runtime/mode.py` |

#### Safety gate evaluation + CI runner

| | |
|---|---|
| **Purpose** | Fail-closed release certification per (condition_mode, platform) cell. |
| **What it does** | Evaluates thresholds; builds certification manifest; CI wrapper writes JSON artifact. |
| **Why it exists** | Block false “all passed” without hazard ground truth (SG-01/02). |
| **What it is used for** | CI `safety-gates` job; product certify scripts. |
| **Inputs** | `platform` (`torch_ref` / `onnx` / `coreml`), optional metrics dict, flags `--hazard-gt`, `--tools-missing`, `--force-xfail`; thresholds from YAML/constants. |
| **Outputs** | Manifest `{cells, summary, all_passed}` with statuses `passed` / `failed` / `blocked_missing_hazard_labels` / `skipped_tools_missing` / `xfail_known_issue`. |
| **Owns** | Gate evaluation and manifest shape (`SCHEMA_VERSION`). |
| **Does not own** | Model training or device rendering. |
| **Dependencies** | PyYAML; `CONDITION_MODE_IDS` for clinical modes. |
| **Files** | `ml/evaluation/safety_gates.py`, `ml/config/safety_gates.yaml`, `scripts/infra/run_safety_gate_ci.py` |

#### RAG reliability service

| | |
|---|---|
| **Purpose** | Make retrieval failures observable without blocking hazards. |
| **What it does** | Emits structured reliability/degradation events for the advisory retrieval path. |
| **Why it exists** | Retrieval is non-blocking enrichment; silent failure is unacceptable. |
| **What it is used for** | CI `rag-reliability`; training observability. |
| **Inputs** | Retrieval attempt outcomes; `ml/config/rag_slo.yaml`. |
| **Outputs** | Structured events via `ml.training.observability.emit_event`. |
| **Owns** | RAG reliability event semantics. |
| **Does not own** | Stage A / CriticalEvent authority. |
| **Dependencies** | Observability module (lazy training package init keeps contracts slim). |
| **Files** | `ml/retrieval/rag_reliability.py`, `ml/config/rag_slo.yaml` |

---

### L4 — Domain

#### MaxSightCNN + heads

| | |
|---|---|
| **Purpose** | Single multi-head vision model family for detection, urgency, distance, therapy-related, OCR/scene, and related tasks. |
| **What it does** | Shared backbone/FPN with condition-aware head gating. |
| **Why it exists** | Share representation across assistive tasks while allowing tiered capability. |
| **What it is used for** | Training, export, Stage A/B inference depending on tier. |
| **Inputs** | RGB `[B,3,H,W]` (typically 224); optional audio features; condition tensor of width `CONDITION_TENSOR_WIDTH` (14). |
| **Outputs** | Dict of head tensors (`objectness`, `classifications`, `boxes`, `urgency_scores`, `distance_zones`, depth/motion/therapy/OCR/scene as enabled). |
| **Owns** | Neural forward graph and head wiring. |
| **Does not own** | Device I/O, certification scoring, OTA. |
| **Dependencies** | torch; backbone under `ml/models/backbone/`; heads under `ml/models/heads/`. |
| **Files** | `ml/models/maxsight_cnn.py`, `ml/models/backbone/vit_backbone.py`, `ml/models/heads/*` |

#### Condition modes

| | |
|---|---|
| **Purpose** | Explicit one-hot condition-mode contract (inspectable input, not a latent embedding). |
| **What it does** | Maps mode strings to indices in `CONDITION_MODE_IDS` (`none` + 13 clinical modes). |
| **Why it exists** | Condition-specific preprocessing and head emphasis must be auditable. |
| **What it is used for** | Forward passes, certification matrix cells, condition attribution. |
| **Inputs** | Mode string or tensor index. |
| **Outputs** | Index / one-hot slice; `CONDITION_TENSOR_WIDTH`. |
| **Owns** | Mode vocabulary and width constant. |
| **Does not own** | Clinical diagnosis. |
| **Dependencies** | Schema `docs/contracts/schemas/condition_tensor.json`. |
| **Files** | `ml/runtime_constants.py` |

#### Therapy engine

| | |
|---|---|
| **Purpose** | Closed-loop intervention decisions on top of perception context. |
| **What it does** | Decides whether/how to intervene; safety module can suppress under high uncertainty or rate limits. |
| **Why it exists** | Adaptive assistance without replacing hazard authority. |
| **What it is used for** | Optional intervention generation after perception. |
| **Inputs** | Situation features (e.g. environment stress, cognitive load, uncertainty); thresholds such as `THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD`, `THERAPY_MAX_PROMPTS_PER_MINUTE`. |
| **Outputs** | Therapy decision (intervene?, type, strength, reason); may emit suppress events. |
| **Owns** | Therapy decision and adaptation policy surface. |
| **Does not own** | `HazardResult` / `CriticalEvent` fields. |
| **Dependencies** | Perception outputs; `ml/therapy/therapy_safety.py`. |
| **Files** | `ml/therapy/therapy_engine.py`, `therapy_safety.py`, adaptation/session modules under `ml/therapy/` |

#### Retrieval stack

| | |
|---|---|
| **Purpose** | Advisory multi-vector retrieval and optional depth for scene/knowledge context. |
| **What it does** | Encodes, ANN-searches (FAISS when installed), reranks; optional MiDaS depth via `torch.hub`. |
| **Why it exists** | Enrich descriptions; never authority for hazard correctness. |
| **What it is used for** | Secondary/advisory path. |
| **Inputs** | Embeddings/queries; optional index path. |
| **Outputs** | Candidate ids/scores; depth maps when MiDaS available. |
| **Owns** | Retrieval pipeline mechanics. |
| **Does not own** | Hazard urgency/direction/distance correctness. |
| **Dependencies** | Optional `faiss` (lazy in `stage1_ann`); optional MiDaS hub; torch. |
| **Files** | `ml/retrieval/retrieval/stage1_ann.py`, `stage2_rerank.py`, `ml/retrieval/encoders/*`, `ml/retrieval/indexing/*`, `ml/retrieval/encoders/midas_loader.py` |

#### Evaluation metrics (offline)

| | |
|---|---|
| **Purpose** | Research/offline metrics and condition-tensor sensitivity — distinct from fail-closed cert gates. |
| **What it does** | Computes multi-modal / accessibility / robustness metric dataclasses; finite-difference condition sensitivity. |
| **Why it exists** | Analysis beyond SG thresholds. |
| **What it is used for** | Offline eval notebooks/scripts; attribution experiments. |
| **Inputs** | Predictions/targets or `(model, images, condition_tensor)`. |
| **Outputs** | Metric dataclasses; `{output_key, grad_l1, grad_per_mode}`. |
| **Owns** | Offline metric definitions. |
| **Does not own** | CI certification manifest. |
| **Dependencies** | torch/numpy (metrics package is torch-eager; evaluation package `__init__` lazy-loads metrics). |
| **Files** | `ml/evaluation/metrics.py`, `ml/evaluation/condition_attribution.py` |

---

### L5–L6 — Data

#### Checkpoints

| | |
|---|---|
| **Purpose** | Per-condition weight storage for deploy and OTA. |
| **Inputs** | Training/export artifacts. |
| **Outputs** | Loadable paths under `checkpoints/<condition>/`. |
| **Owns** | On-disk layout conventions. |
| **Does not own** | Handle resolution policy. |
| **Files** | `checkpoints/` |

#### Medallion datasets

| | |
|---|---|
| **Purpose** | Progressive bronze → silver → gold dataset refinement for reproducible training. |
| **Inputs** | Raw sources; ops scripts in `scripts/ops/`. |
| **Outputs** | Gold manifests; registry resolution via `ml/training/configs/registry/`. |
| **Owns** | Dataset layout and manifests. |
| **Does not own** | Live inference. |
| **Files** | `datasets/`, `ml/training/configs/registry/` |

#### Redis cache (optional)

| | |
|---|---|
| **Purpose** | Optional shared cache for repeated lookups. |
| **Inputs** | Redis URL/connection. |
| **Outputs** | Cached values; factory returns `None` when `REDIS_AVAILABLE` is false. |
| **Owns** | Cache client behavior. |
| **Does not own** | Model weights. |
| **Degrades** | `ImportError` / disabled path when `redis` not installed (`REDIS_AVAILABLE = False`). |
| **Files** | `ml/cache/redis_cache.py` |

#### S3 / model registry

| | |
|---|---|
| **Purpose** | Cloud artifact bridge for training/deploy. |
| **Inputs** | `MAXSIGHT_S3_BUCKET`, SageMaker role ARNs, object keys. |
| **Outputs** | Uploaded objects; registry records; CI dry-run support. |
| **Owns** | S3 client wrappers and registry client surface. |
| **Does not own** | On-device Stage A. |
| **Files** | `ml/infra/s3_client.py`, `ml/infra/model_registry.py` |

---

### L7 — Externals

| External | Required? | Role | Degradation |
|---|---|---|---|
| torch / torchvision / torchaudio | Required on torch CI / device torch path | Training and inference | CPU wheel in CI (`torch-cpu` profiles) |
| sentence-transformers | Soft-required for OCR/text encoders | Text/global encoding | Encoder paths fail if missing |
| faiss | Optional | Stage-1 ANN | Lazy import; clear `ImportError` on ANN use without install |
| redis | Optional | Cache | `REDIS_AVAILABLE`; cache disabled |
| coremltools / onnx / executorch | Optional | Export targets | Export functions return `None` / warn |
| MiDaS (`torch.hub` intel-isl/MiDaS) | Optional | Depth | `RuntimeError` with diagnostic if offline |
| boto3 / sagemaker | Ops | Cloud train/deploy | Dry-run / ImportError guards in ops paths |

---

## 5. Five Questions Method

### Stage A (`ml/runtime/stage_a/`)

| Question | Answer |
|---|---|
| **Why** | Guarantee offline, network-free hazard inference with trusted weights. |
| **Called by** | Device frame loop; Stage A CI; `torch_ref` certification path. |
| **Calls** | Preprocess; resolved local artifact; maps to `HazardResult`. |
| **Owns** | `CameraFrame`/`HazardResult`/`StageARunner`; handle resolution; torch runner. |
| **Fails** | Missing/bad artifact → `REFUSED` + `HAZARD_UNAVAILABLE_MESSAGE`; never blocks waiting on network. |
| **I/O** | In: `CameraFrame`. Out: `HazardResult` (urgency, direction, distance_zone, confidence, uncertainty, latency_ms, model_hash, …). |

### Therapy (`ml/therapy/`)

| Question | Answer |
|---|---|
| **Why** | Offer adaptive assistance without overriding hazard. |
| **Called by** | Orchestrator/simulator after perception context is available. |
| **Calls** | Safety suppress rules; adaptation/session helpers. |
| **Owns** | Intervention decision surface and therapy rate limits. |
| **Fails** | High uncertainty (`THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD`) or rate limits → suppress; never mutates Stage A outputs. |
| **I/O** | In: situation features. Out: intervene?/type/strength/reason. |

### Retrieval / RAG reliability (`ml/retrieval/`)

| Question | Answer |
|---|---|
| **Why** | Enrich context; keep failures visible and non-blocking. |
| **Called by** | Secondary path; CI rag-reliability; observability consumers. |
| **Calls** | Encoders, ANN/rerank, optional MiDaS; `emit_event`. |
| **Owns** | Advisory retrieval pipeline and reliability events. |
| **Fails** | Index/hub/import errors → degraded/advisory events; Stage A continues. |
| **I/O** | In: embeddings/queries/SLO config. Out: candidates/scores; structured events. |

### Safety gates / certification

| Question | Answer |
|---|---|
| **Why** | Prevent shipping with false-green certification. |
| **Called by** | CI `safety-gates`; product certify scripts. |
| **Calls** | Threshold checks (`SG-01`…); manifest builder. |
| **Owns** | Cell status semantics and manifest schema version. |
| **Fails** | No hazard GT → `blocked_missing_hazard_labels`, `all_passed=false`; tools missing → skip (never counts as pass). |
| **I/O** | In: platform, metrics?, flags. Out: manifest JSON (`cells`, `summary`, `all_passed`). |

### Training loop + export (`ml/training/`)

| Question | Answer |
|---|---|
| **Why** | Produce reproducible weights and deployable artifacts. |
| **Called by** | Ops train scripts; SageMaker entrypoints; export tooling. |
| **Calls** | Model forward/backward; observability health summaries; optional CoreML/ONNX/ExecuTorch/JIT exporters. |
| **Owns** | Train loop contracts (skipped batch ratio, health summary); export adapters. |
| **Fails** | Export target missing → return `None`; train loop aborts on skip-ratio breach. |
| **I/O** | In: configs (`ml/training/configs/`), datasets/registry. Out: checkpoints; optional `.mlpackage` / ONNX / `.pte` / JIT. |

---

## 6. Display / UX

| User | Context | Primary channels | Notes |
|---|---|---|---|
| Wearer (novice) | Daily mobility | Haptic urgency + short voice | Defaults favor low cognitive load; critical urgency (≥ `CRITICAL_URGENCY_THRESHOLD` = 3) always surfaced |
| Wearer (power) | Tuned preferences | Personal mode scaling; denser secondary cues | Must not weaken Stage A contract |
| Pilot / clinician operator | Evaluation sessions | Overlay + logs + cert summaries | Uses simulator and certification manifests |
| ML engineer | Training/debug | Desktop consoles, pytest, tensorboard/logs | Desktop-first |
| On-call infra | Deploy/rollback | Ops scripts, CloudWatch-oriented metrics defs | IAM-scoped roles |

**Accessibility:** Non-visual channels (haptic/voice) are first-class for hazards. Overlay is secondary. Therapy prompts are rate-limited separately from hazard alerts (`THERAPY_MAX_PROMPTS_PER_MINUTE`, `THERAPY_MIN_GAP_BETWEEN_PROMPTS_S`). Channel pacing uses `MIN_CHANNEL_INTERVAL_S` (0.3s) for non-emergency outputs.

---

## 7. Platform overview

### ASCII system diagram

```
 CameraFrame
     |
     v
 +------------------+     hash/sign      +------------------+
 | ModelHandle      | <--------------- | OTA / artifacts  |
 | resolve ACTIVE/  |                  +------------------+
 | LKG/REFUSED      |
 +--------+---------+
          |
          v
 +------------------+     HazardResult      +--------------------+
 | Stage A Runner   | --------------------> | Haptic/Voice/UI    |
 | (local, no net)  |                       +--------------------+
 +--------+---------+
          |
          | (optional, non-blocking)
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
   Stage B client     Therapy engine     Retrieval/RAG
   (connectivity)     (suppressible)     (advisory only)
          |                  |                  |
          v                  v                  v
     SecondaryEvent    therapy prompts    rag.* events
```

### Local companion topology

```
 [Laptop / CI]
    train_loop, export, safety_gate_ci, validators
         |
         | artifacts (hash/sign)
         v
 [Phone / runtime]
    Stage A + app UI + OTA client
         |
         | optional
         v
 [Cloud]
    SageMaker train/deploy (dry-run in CI), S3 registry
```

### Actors

| Actor | Responsibility |
|---|---|
| Wearer | Receives hazard and optional assistance cues |
| Pilot operator | Runs eval sessions, reviews cert cells |
| Model engineer | Trains, exports, interprets metrics |
| Release signer | Ensures artifact hash/signature before activate |
| CI | Enforces gate ownership and fail-closed cert |

### Layer map

| Layer | Directory / surface |
|---|---|
| L1 Client | `app/` |
| L2 Contracts / edge | `docs/contracts/`, `tools/simulation/` |
| L3 Runtime services | `ml/runtime/`, `ml/evaluation/safety_gates.py`, `scripts/infra/run_safety_gate_ci.py`, `ml/retrieval/rag_reliability.py` |
| L4 Domain | `ml/models/`, `ml/therapy/`, `ml/retrieval/`, `ml/evaluation/` |
| L5–L6 Data | `checkpoints/`, `datasets/`, `ml/cache/`, `ml/infra/` |
| L7 Externals | torch, faiss, redis, MiDaS hub, coremltools/onnx/executorch, boto3 |

---

## 8. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.10 (CI) |
| ML | torch ≥2.9.1, torchvision, torchaudio |
| Encoding / ANN | sentence-transformers; optional faiss |
| Cache | optional redis |
| Config | PyYAML; JSON Schema |
| Quality | ruff, flake8, black (advisory), basedpyright |
| Tests | pytest; JUnit artifacts in CI |
| Cloud | boto3; SageMaker scripts (ops profile) |
| CI | GitHub Actions; `.github/workflows/ci.yml` + `reusable-python.yml`; profiles `lint` / `contracts` / `torch-cpu` / `torch-cpu-full` / `ops` / `drift` |
| Export (optional) | CoreML, ONNX, ExecuTorch, JIT via `ml/training/export.py` |

---

## 9. Runtime topology & graceful degradation

| Missing / fault | Behavior |
|---|---|
| Network down | Stage A continues; Stage B/OTA gated by connectivity monitor |
| Model artifact missing/bad hash | Handle → `REFUSED`; user-facing `HAZARD_UNAVAILABLE_MESSAGE` |
| `redis` not installed | `REDIS_AVAILABLE=False`; cache factory returns `None` / raises on forced construct |
| `faiss` not installed | Package import OK; ANN API raises clear `ImportError` |
| MiDaS hub offline | `midas_loader` raises `RuntimeError` with diagnostic |
| coremltools / onnx / executorch missing | Export returns `None` (warn) |
| Hazard GT absent in CI | Cert cells `blocked_missing_hazard_labels`; `all_passed=false` |
| Retrieval failure | Advisory `rag.*` events; Stage A unaffected |
| High therapy uncertainty | Suppress therapy prompts |
| Overload | `ALERTS_PER_MINUTE_CAP`; `DegradedMode` D1–D3 per runtime boundary spec |

---

## 10. Background jobs

| Job | Entry | Purpose |
|---|---|---|
| Safety gate CI | `scripts/infra/run_safety_gate_ci.py` | Emit certification manifest |
| Runtime / train-loop validators | `scripts/infra/validate_*.py` | Contract drift |
| Stage A isolation | `scripts/infra/validate_stage_a_isolation.py` | Enforce no-network Stage A |
| Pre-SageMaker gate | `scripts/infra/pre_sagemaker_gate.py` | Preflight before cloud train |
| Quality audit | `scripts/infra/run_quality_audit.py` | Quality workflow |
| SageMaker train/deploy | `scripts/ops/sagemaker_train.py`, `sagemaker_deploy.py` | Cloud lifecycle (dry-run in CI) |
| Medallion build/sync | `scripts/ops/medallion_build.py`, `sync_medallion_s3.py` | Dataset pipeline |
| Checkpoint maintenance | `scripts/ops/ensure_checkpoint_layout.py`, `cleanup_cloud_checkpoints.py` | Layout hygiene |
| Export for Xcode | `scripts/ops/export_for_xcode.py` | Mobile export packaging |

There is no long-running in-repo daemon; jobs are invoked scripts (local or CI).

---

## 11. Key data flows

### 1. Frame → Stage A → wearer cues

```
CameraFrame(image, frame_id, timestamp)
  -> ModelHandle.resolve()
  -> StageARunner.infer(frame)
  -> HazardResult(urgency, direction, distance_zone, ...)
  -> haptic_feedback / voice_feedback / overlay_engine
```

### 2. Condition tensor forward (train/eval)

```
images [B,3,H,W] + condition_tensor [B, CONDITION_TENSOR_WIDTH]
  -> MaxSightCNN.forward(...)
  -> output dict (urgency_scores, distance_zones, boxes, ...)
  -> optional condition_tensor_sensitivity(...)
```

### 3. OTA activate / rollback

```
signed URI + expected hash
  -> downloader -> staging
  -> verify hash/signature
  -> activation (ACTIVE) OR rollback (LAST_KNOWN_GOOD)
  -> ModelHandle resolution on next infer
```

### 4. Certification

```
per clinical mode x platform
  -> evaluate_condition_platform_cell(metrics?, hazard_gt?, tools?)
  -> cell status (PASS/FAIL/BLOCKED/SKIP/XFAIL)
  -> build_certification_manifest
  -> artifacts/ci_manifest_torch_ref.json (CI)
```

### 5. RAG advisory (parallel)

```
query embedding
  -> Stage1ANN (optional faiss) -> Stage2Reranker
  -> advisory context
  -> rag_reliability / emit_event on success or degrade
  (Stage A path independent)
```

---

## 12. Auth, roles & access control

| Mechanism | Scope | Notes |
|---|---|---|
| HMAC session tokens | `ml/auth/token.py` | Secret `MAXSIGHT_SECRET_KEY`; TTL via `MAXSIGHT_SESSION_TIMEOUT` (default ~1h) |
| Image upload validation | `ml/security/validation.py`, `magic.py` | Base64 + magic-byte checks |
| Security headers / error sanitizer | `ml/middleware/*` | HTTP hardening for simulator/API surfaces |
| IAM SageMaker execution | `infra/iam/sagemaker_execution_role.json` | Cloud train |
| IAM model release export | `infra/iam/model_release_export_role.json` | Export pipeline |
| IAM phone readonly | `infra/iam/model_release_phone_readonly_role.json` | Device pull |
| Role assert bypass | `MAXSIGHT_SKIP_ROLE_ASSERT=1` | CI dry-run only |

There is **no** full product RBAC (no resident/clerk-style role matrix). Access is token + IAM-scoped deployment paths.

---

## 13. Architectural principles

1. **Stage A isolation:** `StageARunner.infer` accepts only a `CameraFrame` — no network client or connectivity flag.
2. **Hazard authority:** Retrieval and therapy are advisory/suppressible; they must not rewrite CriticalEvent/HazardResult semantics.
3. **Fail-closed certification:** Without hazard ground truth, SG-01/02 cells are `blocked_missing_hazard_labels`; `all_passed` must be false.
4. **Skip/XFAIL never count as pass** in certification summaries.
5. **Trusted weights only:** Model handle resolution prefers ACTIVE, then LAST_KNOWN_GOOD, else REFUSED.
6. **Explicit condition tensor:** Condition is an inspectable one-hot of width `CONDITION_TENSOR_WIDTH`, not a hidden embedding.
7. **Latency budget:** Critical path targets `LATENCY_MEDIAN_MS` / `LATENCY_P95_MS` (80 ms) per productization gates.
8. **Overload guardrails:** `ALERTS_PER_MINUTE_CAP` and channel interval `MIN_CHANNEL_INTERVAL_S`.
9. **CI gate ownership:** Path filters in `.github/workflows/ci.yml` own which jobs run; slim `contracts` profile must not require torch except where jobs opt into `torch-cpu`.
10. **Artifact integrity:** Hash/signature verification participates in resolution and OTA activation.
11. **MVP output discipline:** Runtime consumers should depend only on `MVP_MODEL_OUTPUT_KEYS` unless explicitly tier-gated.
12. **Degraded modes are explicit:** D0–D3 in `DegradedMode` — no silent half-failure.

---

## 14. Related documentation

| Doc | Role |
|---|---|
| `docs/architecture.md` | Shorter model/system overview (superseded as canonical by this file) |
| `docs/therapy_architecture.md` | Therapy closed-loop detail |
| `docs/training_architecture.md` | Training architecture |
| `docs/productization/01_product_scope_and_claims.md` | Scope and claims |
| `docs/productization/02_safety_first_release_gates.md` | Safety gates narrative |
| `docs/productization/04_runtime_boundary_spec.md` | Degraded modes / boundaries |
| `docs/productization/05_pilot_validation_protocol.md` | Pilot protocol |
| `docs/contracts/` | OpenAPI + JSON schemas |
| `docs/mlopwalkthrough/` | Ops runbooks / remediation / repro gate |
| `docs/caching.md` | Cache behavior |
| `docs/ml_lifecycle_s3.md` | S3 lifecycle |
| `notes/governance_*.md`, `notes/invariants_from_tests.md` | Working governance / invariant notes |

---

## 15. Testing (architecture-relevant)

| Tier | Command (representative) | Protects |
|---|---|---|
| Stage A contracts | `pytest tests/test_stage_a_*.py tests/test_model_handle_resolution.py tests/test_signature_in_resolution.py tests/test_artifact_signing.py -q` | Isolation, types, handle/signature |
| Safety gates | `pytest tests/test_safety_gates_*.py tests/test_run_safety_gate_ci.py tests/test_certification_manifest_schema.py tests/test_per_platform_certification.py -q` + `python scripts/infra/run_safety_gate_ci.py --platform torch_ref --output artifacts/ci_manifest_torch_ref.json` | Fail-closed cert |
| Phone app layer | `pytest tests/test_connectivity_state_machine.py tests/test_stage_b_timeout.py tests/test_ota_staging.py tests/test_local_rollback.py tests/test_haptic_urgency.py -q` | Connectivity/OTA/haptics |
| RAG reliability | `pytest tests/test_rag_reliability.py -q` | Advisory SLO / events |
| Torch model | `pytest tests/test_model.py tests/test_condition_tensor_contract.py tests/test_condition_tensor_forward.py tests/test_timing_enforcement.py -q` | Forward + latency |
| Infra IAM | `pytest tests/test_model_release_iam_scope.py tests/test_infra_validate_stubs.py -q` | IAM stub integrity |
| Drift / train contracts | `python scripts/infra/validate_runtime_contracts.py` ; `python scripts/infra/validate_train_loop_contracts.py` | Schema + train-loop observability |
| Unit heavy | `pytest tests/ -q` (CI profile `torch-cpu-full`) | Broad regression |
| Therapy safety | `pytest tests/test_therapy_safety.py -q` | Therapy suppress rules |

CI job ↔ profile mapping lives in `.github/workflows/ci.yml` (`contracts` vs `torch-cpu` vs `torch-cpu-full` vs `ops`).

---

## 16. Schema / migration checklist

Use this checklist when changing existing deployed contracts or layouts.

1. **Condition tensor width**
   - If adding/removing a mode in `CONDITION_MODE_IDS`, update `CONDITION_TENSOR_WIDTH`, `docs/contracts/schemas/condition_tensor.json`, cert matrix assumptions (13 clinical + `none`), and any stored tensors/checkpoints that assume prior width.
   - Gate: `tests/test_condition_tensor_contract.py`, `tests/test_condition_tensor_forward.py`.

2. **HazardResult / CriticalEvent fields**
   - Additive fields only unless a versioned bump is agreed.
   - Gate: Stage A type tests; runtime contract validators; phone consumers of urgency/direction/distance_zone.

3. **Certification manifest**
   - Preserve `SCHEMA_VERSION` compatibility or bump with consumer updates.
   - Gate: `tests/test_certification_manifest_schema.py`; schema `model_certification_manifest.json`.

4. **safety_gates.yaml platforms**
   - Adding `litert` (or others) requires CI cert-matrix assertions to include it when present.
   - Do not treat SKIP/XFAIL as pass.

5. **Checkpoint layout**
   - Keep `checkpoints/<condition>/` conventions; run `scripts/ops/ensure_checkpoint_layout.py` after moves.
   - Ensure OTA and handle resolution still find ACTIVE/LKG paths.

6. **Medallion / registry YAML**
   - Update `ml/training/configs/registry/` and gold manifests together; validate with `tests/test_dataset_registry.py`.

7. **MVP model output keys**
   - Expanding runtime dependence beyond `MVP_MODEL_OUTPUT_KEYS` requires explicit tier/product decision and contract updates (`model_output.json`).

8. **Do not break Stage A wire format**
   - `StageARunner.infer` must remain `(CameraFrame) -> HazardResult` with no connectivity parameter.
   - Gate: `scripts/infra/validate_stage_a_isolation.py` and Stage A CI job.

9. **IAM / env contracts**
   - Changes to required env (`MAXSIGHT_S3_BUCKET`, roles) must update ops dry-run scripts and IAM stubs under `infra/iam/`.

10. **Optional dependency boundaries**
    - Prefer lazy imports for faiss/metrics package side-effects so slim CI profiles keep collecting; document new optional deps in section 9.
