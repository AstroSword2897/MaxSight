# Block 1.1 — Package inventory (guesses)

**Total `.py` files** (excluding `.venv`, `__pycache__`, `.mypy_cache`, `.pytest_cache`): **414**

**Command:** `find . -name "*.py" -not -path "./.venv/*" -not -path "*/__pycache__/*" -not -path "./.mypy_cache/*" -not -path "./.pytest_cache/*" | wc -l`

## `ml/` top-level packages (guessed one-liners)

| Package | Guessed role (correct later) |
|---|---|
| `ml/auth` | Session/token auth helpers for API or simulator access. |
| `ml/cache` | Redis or in-process caching for inference outputs. |
| `ml/config` | YAML/config loading for tiers, therapy, safety gates. |
| `ml/data` | Datasets, gold/medallion loaders, augmentation, video utils. |
| `ml/evaluation` | Metrics, safety-gate eval, condition attribution. |
| `ml/infra` | SageMaker/S3/registry/signing cloud and ops helpers. |
| `ml/middleware` | HTTP error sanitization and security headers. |
| `ml/models` | MaxSightCNN, backbone, heads, fusion, temporal. |
| `ml/optimization` | Mobile/quantization/pruning helpers. |
| `ml/pipeline` | Offline/SageMaker processing and advisory RAG wiring. |
| `ml/retrieval` | Encoders, indexing, RAG hardening/reliability. |
| `ml/runtime` | Runtime contracts, tier routing, Stage A hard contract. |
| `ml/security` | Security policy checks for infra/IAM. |
| `ml/therapy` | Closed-loop therapy engine, safety, scoring, sessions. |
| `ml/tools` | Small ML utility scripts packaged under ml. |
| `ml/training` | Train loop, losses, export, QAT, SageMaker entry. |
| `ml/utils` | Preprocessing, schedulers, shared utilities. |

## Outside `ml/` (noted for inventory)

| Path | Guessed role (correct later) |
|---|---|
| `app/` | Personal mode, overlays, haptics, connectivity, Stage B, OTA. |
| `tools/simulation` | Flask web simulator and related test harnesses. |

## Correct later (Block 7)

_Leave blank until mental-model pass._
