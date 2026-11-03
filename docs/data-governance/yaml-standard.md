# Configuration YAML standard

Applies to `configs/`, `ml/config/`, `ml/training/configs/` and `scripts/product/*.yaml`. Enforced by
`tests/unit/test_yaml_standard.py`. CI workflows, GitHub actions, `environment.yml` and `docs/contracts/openapi.yaml`
follow their own tool formats and are exempt.

1. First line is a `#` header: purpose, loader path, owner.
2. Top-level `schema_version: "MAJOR.MINOR.PATCH"` (a quoted string). Bump MAJOR for a breaking change to keys or meaning.
   Loaders may accept the legacy `version` key for one release; new files never use it.
3. Top-level keys are `snake_case`. Data maps keyed by external names (label names, dataset class names) are exempt.
4. Block style for mappings (no `{a: b}`). Short flow lists of scalars are allowed.
5. No secrets, credentials or machine paths. Product rules live in tests: no IMU/LiDAR/GPS/accelerometer keys,
   BDD100K stays `eval_only`, Vistas stays non-commercial.
