# Governance check (Block 2.4)

Rules from [`notes/governance_rules.txt`](governance_rules.txt). Each rule checked with one targeted command against imports / repo layout.

| # | Rule (short) | Check command | Result | Evidence |
|---|--------------|---------------|--------|----------|
| 1 | No parallel governance trees under `docs/` for L1–L9 | `rg -l 'TB system governance\|L1 — Repo core' docs; ls docs/systems` | **pass** | No `docs/systems/`; no docs file matches TB governance section title (README remains single source) |
| 2 | Do not collapse layers (process rule) | n/a — process/meta rule, not import-graph enforceable | **n/a** | Requires PR/review discipline; not visible as static import edge |
| 3 | L1 owns models/data/training/evaluation/export | `ls ml/models ml/data ml/training ml/evaluation ml/training/export.py` | **pass** | Packages and `ml/training/export.py` exist as L1 surface |
| 4 | No boto3/SageMaker SDK under `ml/training`, `ml/models`, `ml/data` | `rg -n '^(import boto3\|from boto3\|import sagemaker\|from sagemaker)' ml/training ml/models ml/data` | **pass** | Zero matches |
| 5 | No therapy/C1 event semantics in datasets/gold IR | `rg -n 'TherapyTaskIntegrator\|therapy_copy\|event semantics' ml/data --glob '*.py'` | **pass** | Zero matches under `ml/data` |
| 6 | Gold Layer A schema; no registry required in meta gold load | `test -f ml/data/gold/schema.py` | **pass** | `ml/data/gold/schema.py` present (Layer A home) |
| 7 | Batches use `images` key; gold plane supported in run_config | `rg -n 'KEY_IMAGES\|images' ml/data/sample_contract.py`; `rg -n data_plane ml/training/run_config.py` | **pass** | `KEY_IMAGES = "images"`; `data_plane=gold` validation paths present |
| 8 | `inference_handler` must not import offline entrypoints | `rg -n 'sagemaker_entrypoint\|pipeline_runner' ml/infra/inference_handler.py` | **pass** | Zero matches |
| 9 | Offline pipeline vs realtime contract (process) | n/a — architectural role split | **n/a** | Files exist separately (`pipeline_runner.py` vs `inference_handler.py`); not an import-edge fail |
| 10 | RAG advisory-only; no gold schema writes from therapy/retrieval | `rg -n 'ml\.data\.gold\|gold/schema' ml/therapy ml/retrieval --glob '*.py'`; `rg -n advisory-only ml/pipeline/rag_advisory.py` | **pass** | No gold imports from therapy/retrieval; rag_advisory documents advisory-only |
| 11 | L3/L4 ops vs infra ownership | `ls scripts/ops ml/infra` | **pass** | Both trees exist; cross edges show `ml.infra → ml.middleware` and limited `ml.training → ml.infra` only in SageMaker entry |
| 12 | Deploy registry gate; skip forbidden in production | `rg -n 'skip.registry\|MAXSIGHT_ENV' scripts/ops/sagemaker_deploy.py` | **pass** | `--skip-registry-check` blocked when `MAXSIGHT_ENV` is `production`/`prod` |
| 13 | L6–L7 must not change L1 gold IR without review | n/a — process rule | **n/a** | Not verifiable via import grep alone |
| 14 | L9 infra stubs not runtime imports for training | `rg -n 'from ml\.infra\|import ml\.infra' ml/training --glob '*.py'` | **pass\*** | No imports of `infra/*.json` stubs into training; *note:* `ml/training/sagemaker_entry.py` imports `ml.infra.experiment_tracker` / `security_policy` (AWS entry seam, not L9 JSON stubs) |
| 15 | Canonical gold JSONL+meta; medallion legacy | `rg -n data_plane ml/training/run_config.py` | **pass** | `data_plane=gold` is first-class in run_config; medallion index not required for gold plane |

## Cross-package notes vs L1 AWS boundary

From [`notes/cross_package_edges.md`](cross_package_edges.md):

- `ml.training → ml.infra` appears via `ml/training/sagemaker_entry.py` (expected L5/entry seam, not boto3 inside train loop core).
- `ml.pipeline → ml.data` / `ml.training` consistent with offline Processing.
- `ml.models → ml.therapy` / `ml.retrieval` are product edges to watch for L2 “advisory / no authority override” (covered by tests in Block 3, not import absence).

## Summary

- **pass:** 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14*, 15  
- **n/a (process):** 2, 9, 13  
- **fail:** none from these mechanical checks
