# MaxSight Declutter Map: Canonical Product Pipeline

## Purpose
Reduce current script sprawl into a stable, reproducible product command surface:
- `train`
- `validate`
- `export`
- `package`
- `smoke`

All scripts should map to one of:
- Keep as canonical entrypoint.
- Merge into canonical command.
- Archive (research/legacy/one-off).
- Keep as ops utility (non-release-critical).

## Canonical Command Targets

| Canonical command | Product intent | Primary target entrypoint |
|---|---|---|
| `train` | Train production model/checkpoint | `scripts/train_maxsight.py` |
| `validate` | Run model quality + safety checks | `scripts/run_checkpoint_inference.py` + test suite wrappers |
| `export` | Convert checkpoint to deployable formats | `python -m ml.training.export` |
| `package` | Produce glasses-ready bundle(s) | `run.py package` or `scripts/ops/export_for_xcode.py` |
| `smoke` | Fast end-to-end sanity and regressions | `scripts/smoke_train.py` + minimal inference/export smoke |

## Script Consolidation Matrix

### Training

| Current script | Action | Reason |
|---|---|---|
| `scripts/train_maxsight.py` | Keep (canonical) | Main production training entrypoint |
| `scripts/train_alive_models.py` | Merge into `train` modes | Condition-specific variant should be a flag/profile |
| `scripts/train_t5_fast_colab.py` | Keep as preset profile wrapper | Colab profile, not separate flow |
| `scripts/check_and_train_colab.py` | Merge into `train --check-env` | Avoid duplicate workflow logic |
| `scripts/AutoMLType.py` | Keep as optional tuning utility | Useful but not release-critical |
| `scripts/diagnose_training_speed.py` | Keep as ops utility | Performance debugging |

### Validation and inference checks

| Current script | Action | Reason |
|---|---|---|
| `scripts/run_checkpoint_inference.py` | Keep (canonical validate subcommand target) | Core checkpoint validation |
| `scripts/run_inference_on_inference_datasets.py` | Merge into `validate --datasets` | Dataset validation profile |
| `scripts/sanity_check_inference.py` | Merge into `smoke --inference` | Quick sanity belongs to smoke flow |
| `scripts/compare_condition_models.py` | Keep as report utility | Decision support for model selection |
| `scripts/get_top7_by_map.py` | Merge into model-selection module | Avoid standalone script proliferation |
| `scripts/improve_map_all_models.py` | Archive (research workflow) | One-off tuning workflow |

### Export and packaging

| Current script | Action | Reason |
|---|---|---|
| `python -m ml.training.export` | Keep (canonical export core) | Single source of export truth |
| `scripts/ops/export_for_xcode.py` | Keep (canonical package entrypoint) | Deployment bundle packaging |
| `scripts/deploy_top7.py` | Merge under `package --top7` | Deployment flavor, not separate pipeline |
| `scripts/research_archive/export_top7_to_xcode.py` | Optional / research | Multi-condition export wrapper |
| `scripts/export_7_coreml_only.py` | Merge into `export --coreml-only --top7` | Duplicate functionality |
| `scripts/export_one_model.py` | Merge into `export --single` | Duplicate functionality |
| `scripts/convert_pt_to_coreml.py` | Merge into export module | Duplicate conversion path |
| `scripts/find_and_convert_coreml.py` | Archive (condition-specific one-off) | Too narrow for product pipeline |
| `scripts/colab_convert_coreml.py` | Keep as docs sample wrapper | External notebook flow aid |
| `scripts/check_export_status.py` | Keep as release utility | Valuable for release readiness |
| `scripts/verify_coreml.py` | Keep as validate/export check | Critical artifact verification |

### Data and dataset prep

| Current script | Action | Reason |
|---|---|---|
| `scripts/gather_training_data.py` | Keep (train prerequisite) | Main data prep path |
| `scripts/validate_data_pipeline.py` | Keep as `validate --data` | Data correctness gate |
| `scripts/download_inference_datasets.py` | Keep as ops utility | Required datasets |
| `scripts/download_open_images_direct.py` | Archive | Duplicate download path |
| `scripts/download_open_images_fiftyone.py` | Keep preferred path | Recommended method |
| `scripts/download_open_images_s3.py` | Archive | Alternative legacy path |
| `scripts/reorganize_open_images.py` | Keep as migration utility | Existing dataset maintenance |
| `scripts/patch_missing_images.py` | Keep as recovery utility | Common failure recovery |
| `scripts/find_annotation_images.py` | Keep as debug utility | Fast path resolution aid |
| `scripts/archive/setup_coco_splits.py` | Keep archived | Already archived legacy |

### Ops and maintenance

| Current script | Action | Reason |
|---|---|---|
| `scripts/ensure_checkpoint_layout.py` | Keep (release utility) | Ensures deploy structure |
| `scripts/find_trained_checkpoints.py` | Keep (release utility) | Discovery for packaging |
| `scripts/list_saved_models.py` | Merge into discovery utility | Overlap with checkpoint finder |
| `scripts/create_minimal_checkpoint.py` | Keep for CI/smoke fixtures | Useful synthetic checkpoint path |
| `scripts/cleanup_cloud_checkpoints.py` | Keep as ops utility | Storage maintenance |
| `scripts/monitor_download.py` | Keep as ops utility | Long download monitoring |
| `scripts/test_systems_comprehensive.py` | Merge into test runner/profile | Should be under tests harness |
| `scripts/test_therapy_effectiveness.py` | Move under pilot eval utilities | Not release-critical path |
| `scripts/normalize_comments.py` | Archive tooling | Not product runtime concern |
| `scripts/clean_comments.py` | Archive tooling | Not product runtime concern |

## Proposed Folder/Ownership Shape
- `scripts/product/` -> canonical command wrappers only.
- `scripts/ops/` -> maintenance and environment helpers.
- `scripts/research_archive/` -> non-product one-offs.
- `scripts/pilot_eval/` -> user-study and therapy effectiveness scripts.

## 30-Day Declutter Deliverables
1. Canonical CLI map implemented and documented.
2. Deprecated scripts moved with clear redirect notes.
3. CI invokes canonical commands only.
4. Release runbook updated to use only canonical pipeline.
