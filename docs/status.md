# Project status and health

*Last updated: Nov 7 2026 — integrity and security remediation (PRs #35, #38, #39, #40). Detailed reference: [architecture/MAXSIGHT_ARCHITECTURE.md](architecture/MAXSIGHT_ARCHITECTURE.md).*

This document summarizes the current status of the MaxSight repo: what is implemented, what is tested, and what to watch when running or deploying.

## Implementation status

### Model and training
- **Model (MaxSightCNN):** `ml/models/maxsight_cnn.py` — tiered T0–T5, ~393M params, stage-A/B latency gating (80ms threshold), CI size bounds in `ml/runtime_constants.py`.
- **Training:** Production loop in `ml/training/train_loop.py` with checkpoint resume, skipped-batch guards, `health_summary` logs, and deterministic seeding via `ml/training/reproducibility.py` (manifest written to checkpoint dir on each run).
- **Reproducibility:** `set_deterministic_seed()` + `reproducibility_manifest` wired into `ProductionTrainLoop.__init__`. Gate: `python scripts/infra/validate_train_loop_contracts.py`.
- **Export:** JIT, ExecuTorch (.pte), CoreML, ONNX in `ml/training/export.py`.
- **Quantization:** INT8/FP16 QAT pipeline in `ml/training/qat_pipeline.py` and `ml/training/quantization.py`.

### Compute tier routing
- **Tier architecture:** Bronze / Silver / Gold profiles in `ml/config/tiers/`. Router: `ml/runtime/tier_router.py`.
- **Runtime mode:** `ml/runtime/mode.py` — `get_runtime_mode()`, `resolve_compute_tier()`, and `RuntimeOrchestrator`.

### Runtime orchestration (SCRUM-18/19)
- **`RuntimeOrchestrator`** in `ml/runtime/mode.py`: one `process(RuntimeRequest) → RuntimeResponse` call that wires tier routing → `TherapyEngine` → `HardenedRagPipeline` → `RuntimeResponse`.
- **Score traces:** Every `TherapyRecommendation` carries a `score_trace` dict from `TherapyScoringModel` (stress, effectiveness, safety penalty, final score) for SCRUM-19 explainability.
- **Contracts:** JSON schema + OpenAPI spec in `docs/contracts/`. Validated by `scripts/infra/validate_runtime_contracts.py`.

### Therapy system
- **Closed-loop engine:** `ml/therapy/therapy_engine.py` — `update()` + `on_user_response()` with full adaptation-memory feedback.
- **Scoring model:** `ml/therapy/scoring.py` — hybrid deterministic + learned scorer; wired into `TherapyDecisionEngine` for intervention selection.
- **Ontology routing:** `ml/data/ontology/disability_ontology.json` (7 disabilities) — wired into `TherapyTaskIntegrator.generate_task_from_scene()` and `TherapyScoringModel.recommend_intervention_type()`.
- **Therapy constraints:** `ml/config/therapy_constraints.yaml` loaded by `ml/therapy/constraints_loader.py` — rate limits, disallowed phrases, disability routing map.
- **Safety layer:** `ml/therapy/therapy_safety.py` — suppression rules and content sanitization.
- **Personal mode:** `app/personal_mode.py` — `get_therapy_recommendations()` enriches perception with disability_id/preferred_channel then calls TherapyEngine.

### RAG
- **Hardened pipeline:** `ml/retrieval/rag_hardening.py` — debounce → hallucination guard → timeout → SLO tracking.
- **Offline pipeline:** `ml/pipeline/pipeline_runner.py` — wraps AdvisoryRetriever in `HardenedRagPipeline`; falls back to advisory-only path when retriever unavailable.
- **Advisory protocol:** `ml/pipeline/rag_advisory.py` — `AdvisoryRetriever` protocol for retrieval backends.

### Infrastructure and security
- **SageMaker entry:** `ml/training/sagemaker_entry.py` — S3 path safety pre-flight check on startup.
- **Security policy:** `ml/infra/security_policy.py` — validates IAM stubs against overly permissive principal patterns.
- **Pre-SageMaker gate:** `scripts/infra/pre_sagemaker_gate.py` — 7-check gate (tiers, contracts, security, reproducibility, ontology, constraints, deprecations). Run before any SM job.
- **Canonical runner:** `scripts/product/run.py` — `train | validate | export | package | transfer | smoke | gate | certify` (`certify --sign` signs all-pass artifacts).
- **Model artifact signatures:** Ed25519 instance contract (`docs/contracts/schemas/artifact_signature.json`) under `configs/security/artifact_signing.yaml`; devices refuse unsigned/tampered artifacts. The former HMAC keys were public and are no longer trusted. **No public keys are provisioned yet**, so devices refuse every model until the key ceremony (`ml/runtime/stage_a/keys/README.md`).
- **Strict weight loading:** inference, export and deploy use `ml/training/checkpoint_io.load_model_weights` (all-or-nothing); the Stage A runner and SageMaker `model_fn` refuse unloadable or missing checkpoints.
- **Data integrity:** split manifests and the audit refuse the same image in two splits; training verifies gold shard `sha256`; the legacy COCO splitter groups byte-identical images.

### Haptics and app
- **Haptic backends:** `app/ui/haptic_backends.py` — `MAXSIGHT_HAPTIC_BACKEND` selects Darwin / Linux / Log / Noop.
- **Feature transforms:** `ml/data/feature_transform.py` — PCA + scaling with `FeatureTransformArtifact` persistence.
- **Video ingest validation:** `ml/data/video_ingest_validator.py`.

## Tests

| Suite | File | Coverage |
|---|---|---|
| Runtime contracts + orchestrator | `tests/test_phase0_contracts.py` | 28 tests |
| Foundations (ontology, constraints, PCA, scoring) | `tests/test_phase1_foundations.py` | 24 tests |
| Production remediation (checkpoint, therapy, haptics) | `tests/test_production_remediation.py` | 14 tests |
| Therapy engine | `tests/test_therapy.py` | varies |
| Runtime mode + orchestrator | `tests/test_runtime_mode.py` | varies |
| Integration structure (GradNorm, timing) | `tests/test_integration_structure.py` | 7 tests |

| Integrity and security (2026-11) | `tests/test_strict_weight_loading.py`, `tests/test_artifact_signature_contract.py`, `tests/unit/test_split_content_leakage.py`, `tests/unit/test_coco_splitter_content.py`, `tests/test_gold_shard_verification.py`, `tests/test_checkpoint_content_hash.py`, `tests/test_hybrid_backbone_integration.py`, `tests/test_dynamic_conv_flag.py`, `tests/test_scene_graph_relation_index.py`, `tests/test_personal_mode_user_rows.py`, `tests/test_auth_token_secret.py` | targeted, all passing locally |

Run key suite: `pytest tests/test_phase0_contracts.py tests/test_phase1_foundations.py tests/test_production_remediation.py tests/test_therapy.py -q`

Full suite: last clean local baseline 1301 passed / 5 skipped / 12 xfailed / 0 failed; the 2026-11-07 revision must be re-verified in AWS (the local run was invalidated by a full disk).

Pre-SageMaker gate: `python scripts/product/run.py gate` (must exit 0 before any SM launch)

## Known limitations and risks

- **Model size:** ~375 MB INT8 / ~393M params. Tiered or quantized deployment uses smaller checkpoints. Bounds in `ml/runtime_constants.py`.
- **RAG retriever:** `HardenedRagPipeline` ships with a null retriever by default — advisory scores are low until a real retrieval backend is configured.
- **Haptics:** macOS requires PyObjC/Swift; Linux requires `evdev` with FF-capable hardware. Use `log` backend in CI.
- **JIT / ExecuTorch export:** Tracing can hit unsupported ops; use CPU + JIT-only for stability.
- **CLIP:** Unavailable in this environment (CVE-2025-32434 / torch.load restriction) — falls back to internal encoder. Upgrade torch ≥ 2.6 or use safetensors checkpoints.
- **Hybrid tiers:** T2_HYBRID_VIT, T3, T4 and T5 now run the hybrid CNN–ViT branch (it previously failed silently on every call); checkpoints trained before 2026-11-07 carry untrained hybrid weights.
- **Dynamic convolution:** real layer behind `use_dynamic_conv`, off in every tier and training YAML until retrained.
- **Silver tier RAG:** Gold tier enables RAG; Bronze tier disables it. Silver depends on `enable_rag` in `ml/config/tiers/silver.yaml`.

## How to use this doc

- **Before training:** Run `python scripts/product/run.py gate` — all 7 checks must pass.
- **Before SM launch:** Same gate, plus ensure channel dirs exist and security policy stubs are updated with real account IDs.
- **Before export:** Run `python scripts/product/run.py validate --skip-export-tests` then `run.py export`.
- **After changes:** `pytest tests/` and, for training changes, `python scripts/infra/validate_train_loop_contracts.py`.

For deeper technical detail: `docs/architecture.md`, `docs/training_architecture.md`, `docs/therapy_system.md`, `docs/contracts/openapi.yaml`.
