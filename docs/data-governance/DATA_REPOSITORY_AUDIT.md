# Data repository audit

- **Audited:** 2026-11-03, branch `pr/6-data-governance`, commit `1d11ff2` (stacked on `pr/2-stage-a-schemas`, PR #22).
- **Scope:** everything that acquires, converts, cleans, splits, registers or loads data for training and evaluation.
- **Method:** file inventory, import graph (`git grep`), registry path checks, the torch-free test suites run locally,
  and the CI log of run `41767501868`. Every claim below names its evidence. Nothing here was assumed from a plan.

Status words follow the plan's task workflow: *Implemented* means code exists; *Verified* means tests or a real run
prove the behaviour; *Partial* means some required behaviour is missing.

## 1. What already exists

### 1.1 Two dataset stacks

| Stack | Location | Size | Who uses it |
|---|---|---|---|
| **Current** (PRs #18–#20) | [ml/datasets/](../../ml/datasets/) | 14 modules | `scripts/acquire_data.py`, `scripts/data_sources.py`, `scripts/validate_data.py`, 7 unit test files |
| **Legacy** | [ml/data/](../../ml/data/) | 34 modules incl. `gold/`, `ontology/` | 45 files: `ml/training/runner.py`, `ml/training/run_config.py`, `ml/pipeline/*`, 17 `scripts/ops/*` scripts (incl. `sagemaker_train.py`, `sync_medallion_s3.py`), 14 test files |

The two stacks do not talk to each other. The current stack writes `maxsight.manifest/1` JSONL
([manifest.py](../../ml/datasets/manifest.py)); training reads gold JSONL `schema_version 1.0`
([ml/data/gold/schema.py](../../ml/data/gold/schema.py)) through the legacy registry. **No converter connects them.**

### 1.2 Registries and label maps

| File | What it defines |
|---|---|
| [configs/data/sources.yaml](../../configs/data/sources.yaml) | The eight public sources: role, licence, access, URLs, sizes |
| [configs/data/label_map.yaml](../../configs/data/label_map.yaml) | 17 detector classes (DR-202 default) and per-source name mappings, plus ignore lists |
| [ml/training/configs/registry/datasets.yaml](../../ml/training/configs/registry/datasets.yaml) | 12 legacy training datasets, label space, tier compatibility, split paths |
| [ml/training/configs/registry/label_spaces.yaml](../../ml/training/configs/registry/label_spaces.yaml) | `coco_80` (80 classes) and `accessibility_622` (622 classes) |

### 1.3 Data on disk (`datasets/`, git-ignored except where noted)

| Path | Contents | Evidence |
|---|---|---|
| `datasets/raw/open_images_v7/` | 6,278 validation images, 1.9 GB, CSVs, manifest of 5,951 records | `ACQUIRED.json`, acquired 2026-11-03 19:39Z |
| `datasets/processed/open_images_v7/` | `validation_clean.jsonl` | No `VALIDATED.json` |
| `datasets/reports/open_images_v7_validation_audit.json` | Audit **failed**: vehicle 34.7 %, person 24.1 %, animal 16.2 % exceed the 15 % share | Report file |
| `datasets/raw/coco2017/annotations/` | `annotations_trainval2017.zip` (241 MB), `instances_train2017.json` | Fetched 2026-11-03 |
| `datasets/train/`, `val/`, `test/` | Synthetic images (200 per `generation_stats.json`; impairment probability 0.7) | **2,606 files tracked in git** despite the `/datasets/` ignore rule |
| `datasets/cleaned_splits/` | `class_distribution_report.json`, `class_weights.json` only | No split files |

### 1.4 Model and architecture records

- [ADR-008](../adr/008-ssd-mobilenet-is-the-initial-detection-benchmark.md): SSD-MobileNet is the initial detection benchmark.
- [Decision register](../architecture/v1/01-decision-register.md): DR-112 (dataset list) **Proposed**, final list Open;
  DR-202 (taxonomy) **Open**; DR-111 (pretrain on public mix, fine-tune and evaluate on MaxSight data) **Proposed**.
- `ml/models/detector/` contains only `.gitkeep`. No SSD, SSDLite or MobileNet detector code exists anywhere (`git grep`).

### 1.5 CI

Run `41767501868` (PR #16, after the SageMaker CLI fix): `sagemaker-ops` and 13 other jobs pass. `unit-tests-cpu`
fails with 9 of 655 tests, none in data code:

- 2 × `ModuleNotFoundError: ml.models.attention.cross_task_attention` (module missing)
- 5 × `ModuleNotFoundError: faiss` (dependency not installed in CI)
- 2 × `tests/test_port_binding.py` (port reuse assertions)

Locally, torch fails to load (`libtorch_cpu.dylib` missing), so torch-dependent tests run only in CI.

## 2. Implemented and functional (verified)

| Capability | Where | Evidence |
|---|---|---|
| Source registry with role and access validation | `ml/datasets/sources.py` | `test_public_sources.py` |
| Status ladder from on-disk evidence (`NOT_DOWNLOADED` → `VALIDATED`, purpose-named) | `sources.py` | `test_acquisition_provenance.py` |
| Detector-mix guard: only `pretrain` sources, BDD100K and research sets rejected | `assert_detector_training_sources()` | `test_only_pretrain_sources_may_enter_detector_mix` |
| Commercial-release guard (Vistas) | `commercial_release_blockers()` | `test_vistas_blocks_commercial_release` |
| Resumable HTTP download, truncation detection, HTTP 416 handling, MD5 check, safe tar/zip extraction | `acquisition/fetch.py` | fetch tests against a local HTTP server |
| COCO class-capped selection (rare classes first), byte budget, per-image SHA-256, per-image licence, per-split manifest | `acquisition/coco.py` | unit tests; real val2017 and train2017 runs (see `DATASET_STATUS.md`) |
| Open Images selection from official CSVs (trainable boxes, ignore list, per-class cap), anonymous S3 download, per-image SHA-256 | `acquisition/openimages_manual.py` | unit tests; real validation run 2026-11-03 |
| Per-split `ACQUIRED.json` with source URL, code revision, selection and archive hashes | `write_acquired_marker()` | `test_marker_keeps_every_split_instead_of_overwriting` |
| Raw integrity check (present, decodes, SHA-256 unchanged) | `ml/datasets/integrity.py`, `data_sources.py verify` | `test_verify_raw_*`, CLI test |
| Cleaning: decode check, boxes < 8 px short side → ignore, dHash near-duplicates (Hamming ≤ 4), per-class cap | `cleaning.py` | `test_dataset_core_regressions.py` |
| Group-based splits, held-out groups | `splits.py` | regression tests |
| Audit: class share ≤ 15 %, cross-split near-duplicates, missing files | `audit.py`, `scripts/validate_data.py` | regression tests; real Open Images run (failed as designed) |
| YAML house style and registry/label-map consistency | `tests/unit/test_yaml_standard.py`, `test_config_consistency.py` | CI and local |

## 3. Partially implemented

| Item | What exists | What is missing |
|---|---|---|
| BDD100K | Converter `annotation/bdd100k.py` (eval split, rider → ignore) | Never run on real data; loader-level training exclusion only through the registry guard |
| DIODE | Scanner `annotation/diode.py`, planned fetch with MD5 | Never downloaded; no depth-evaluation subset |
| Negative examples | Cleaning and audit | Both selectors keep only images with ≥ 1 mapped box (`coco.py:81`, `openimages_manual.py:224`), so **no negative pool exists** |
| Per-image licence and attribution | COCO records carry the dataset licence string | COCO per-image Flickr licence IDs are in the annotation JSON but not copied; Open Images per-image author/licence metadata is not downloaded |
| Gold format and training registry | Legacy `ml/data/gold/*`, strict registry loader | No path from current manifests into gold; registry entries reference files that do not exist (section 8) |

## 4. Missing

1. **SSD-MobileNet detector, loader and export contract.** Blocks any training-ready export (prompt phase 10).
2. **`docs/contracts/hazard_taxonomy.md`.** Referenced by the execution prompt; does not exist. DR-202 is Open.
3. Converters for Mapillary Vistas (polygons → `pole_bollard`), PIE (tracks, crossing intention), SANPO, Ego4D.
4. Licence register, acquisition plan and status documents (added in this change: see section 6).
5. MaxSight Hazard v1 schema, capture metadata, reviewer/adjudication records.
6. Assistance-necessity (P2) data schema. No public source provides these labels (plan §5); none may be derived from
   detection annotations.
7. `scripts/download_open_images_fiftyone.py` (named in the prompt) never existed; the Open Images route is
   `scripts/acquire_data.py openimages`.
8. `datasets/synthetic_legacy/` does not exist; the synthetic set lives in `datasets/train|val|test` (section 7).

## 5. Authoritative files

| Concern | Authoritative | Not authoritative |
|---|---|---|
| Public-source identity, roles, licences, access | `configs/data/sources.yaml` | Legacy `datasets.yaml` source rows (`bdd100k`, `kinetics700`, …) |
| Detector classes and source mappings | `configs/data/label_map.yaml` (a DR-202 *default*, not frozen) | `accessibility_622` label space |
| Acquisition, cleaning, splitting, audit | `ml/datasets/*` (decision 2026-11-03: the current stack is authoritative). Since 2026-11-07 the split manifest and audit refuse the same image (sha256 or near-equal dhash) in two splits. | `ml/data/download_datasets.py`, `dataset_cleaning.py`, `coco_dataset_splitter.py` (still called by `scripts/ops/gather_training_data.py` and `medallion_build.py`; the splitter now groups byte-identical images but has no near-duplicate check) |
| Architecture decisions | `docs/architecture/v1/01-decision-register.md`, ADRs | Older text that treats IMU/LiDAR as open (plan §12) |
| Training-time loading (today) | `ml/data/gold/*` + legacy registry | — (to be bridged, not replaced blindly) |

## 6. What must be reused

- Registry: extend `sources.yaml` and `load_registry()`; do not add `configs/data/dataset_registry.yaml`.
- CLI: extend `scripts/data_sources.py` (status, verify, plan, fetch) and `scripts/acquire_data.py`; do not add
  `python -m ml.data.cli`.
- Manifests: `maxsight.manifest/1` records already carry source, group, SHA-256, dHash, split, synthetic flag and
  licence; extend them rather than adding a second sample manifest.
- Documentation: `docs/data-governance/` (the plan's layout), not a new `docs/data/`.
- Training side: bridge into `ml/data/gold` and the legacy registry, because the training runner and SageMaker
  scripts depend on them.

## 7. Data that must not be overwritten or moved

- `datasets/raw/open_images_v7/` and its `ACQUIRED.json` (folded into the per-split format, timestamp kept).
- `datasets/raw/coco2017/` (annotations and images).
- `datasets/train|val|test` and `generation_stats.json`: tracked in git and referenced by the
  `maxsight-synthetic-224` registry entry. Moving them to `synthetic_legacy/` would break that entry; propose the
  move as a separate reviewed change.
- `datasets/cleaned_splits/*` reports and both registries.

## 8. Conflicts for review (none changed by this audit)

| # | Conflict | Evidence | Proposed resolution | Severity |
|---|---|---|---|---|
| C1 | **BDD100K is trainable in the legacy registry.** DR-112 makes it evaluation-only. | `datasets.yaml` `bdd100k`: `tier_compatibility` includes `T2_DETECTOR`, `T3_MULTI_TASK`, `T5_TEMPORAL` | Remove training tiers or add an `eval_only` flag the legacy loader enforces; add a test | High |
| C2 | Legacy label space has **622 classes**; the plan's detector default has 17. | `label_spaces.yaml`, `label_map.yaml` | Decide with DR-202; the SSD-MobileNet export must use one class map | High |
| C3 | Active legacy entries point at files that do not exist. | `maxsight-coco-cleaned` 4/4 paths missing; `coco-bronze` 3/3; `maxsight-accessibility-coco` 3/6 | Mark them `raw` or repoint them at the new COCO manifests through a bridge | High |
| C4 | The only active entry with all files present is **synthetic**. | `maxsight-synthetic-224`, 0/6 missing | Keep it for smoke tests only (plan §8.2); block it from real-world training exports | Medium |
| C5 | Legacy code **derives assistive and urgency labels from detections**. | `assistive_supervision.py` ("class priors, bbox proximity, centre bias"), `generate_annotations.py` ("assign urgency, estimate distance") | Do not use these outputs as hazard or P2 ground truth (prompt §1); quarantine or retire them | High |
| C6 | Legacy loader applies **condition-specific augmentation** and a seven-disability ontology. | `ml/data/dataset.py` `condition_mode`; `ontology/loader.py` | Plan §2: perception is condition-independent and simulations are not evidence; restrict to labelled experiments | Medium |
| C7 | Legacy registry and downloader include sources outside DR-112. | `kinetics700`, `youtube8m`, `howto100m`, `webvid10m`, `epic_kitchens`, `mose`, `youtube_vos`; Objects365, Visual Genome, LVIS in `download_datasets.py` | Leave them at `raw`; add to `sources.yaml` only through a DR-112 decision | Low |
| C8 | 2,606 synthetic files are tracked despite the `/datasets/` ignore rule. | `git ls-files datasets` | Keep (tests depend on them) and document; revisit with the `synthetic_legacy` move | Low |
| C9 | The execution prompt names paths that differ from the plan's layout. | `ml/data/cli`, `docs/data/`, `configs/data/dataset_registry.yaml` | Follow the layout (section 6) | Low |
