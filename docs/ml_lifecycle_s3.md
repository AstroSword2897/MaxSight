# S3 utilities (large-scale data lifecycle)

MaxSight S3 helpers live in `ml/infra/s3_client.py` and `ml/infra/s3_validation.py`. They target **large medallion syncs**, **checkpoint storage**, and **run artefact upload** with explicit validation, **structured JSON event logs**, and **retries** on transient AWS errors.

## Design principles

1. **Validate before I/O** — Bucket names, object keys (≤ 1024 UTF-8 bytes), and local paths are checked so failures are immediate and loggable (`S3ValidationError`).
2. **Structured events** — Each major step logs a single JSON line with `event` like `s3.upload_ok` or `s3.sync_upload_done` (grep-friendly for CloudWatch / ELK).
3. **Bounded memory** — `list_keys(..., max_keys=N)` caps listing for huge prefixes; `sync_medallion_s3.py` exposes `--max-list-keys`.
4. **Partial success** — `sync_upload` / `sync_download` can set `continue_on_error=True` and return `SyncUploadResult` / `SyncDownloadResult` with per-file `failed` entries instead of failing the whole tree.
5. **Retries** — Transient `ClientError` codes (throttling, 503, timeouts) retry with exponential backoff + jitter; `botocore` adaptive retries are enabled on the client.

## API sketch

```python
from pathlib import Path
from ml.infra.s3_client import S3Client, S3OperationError

client = S3Client("my-valid-bucket", prefix="maxsight", max_attempts=5)
result = client.sync_upload(Path("datasets/medallion/silver"), "maxsight/medallion/silver", continue_on_error=True)
# result.uris, result.failed, result.bytes_uploaded, result.skipped
```

## CLI

See `scripts/ops/sync_medallion_s3.py` — use `--max-list-keys` on `status` or `download --dry-run` when buckets may contain millions of objects.

## Exceptions

| Type | When |
|------|------|
| `S3ValidationError` | Invalid bucket/key/URI/local path |
| `S3OperationError` | Retries exhausted or non-retryable AWS error |

## Related

- `docs/git_workflow.md` — branching and merging expectations
- `scripts/ops/sagemaker_train.py` — uploads gold index via `S3Client`
