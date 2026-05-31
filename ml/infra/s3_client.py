"""S3 utilities for the medallion data lifecycle and checkpoint management.

Large-scale sync assumes flaky networks and strict AWS limits; validation runs
before I/O, operations emit structured events, and transient errors retry with backoff.

Usage
-----
from ml.infra.s3_client import S3Client, SyncUploadResult

client = S3Client(bucket="my-maxsight-bucket")
result = client.sync_upload(Path("data"), "prefix")
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ml.infra.s3_validation import (
    MAX_SINGLE_PUT_BYTES,
    S3ValidationError,
    sanitize_relative_key,
    validate_bucket_name,
    validate_local_dir,
    validate_local_file,
    validate_object_key,
    validate_prefix,
)
from ml.infra.s3_validation import (
    parse_s3_uri as _parse_s3_uri_strict,
)

logger = logging.getLogger(__name__)

MEDALLION_LAYERS = ("bronze", "silver", "gold")

# Retry only codes that commonly succeed on retry (large listings / uploads).
_TRANSIENT_CODES = frozenset(
    {
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "Throttling",
        "SlowDown",
        "InternalError",
        "ServiceUnavailable",
        "503",
    }
)

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BASE_DELAY_S = 0.5
_DEFAULT_MAX_DELAY_S = 30.0


class S3OperationError(RuntimeError):
    """Raised when S3 returns a non-retryable error or retries are exhausted."""


def _boto3_session(session=None):
    import boto3  # type: ignore

    return session or boto3.Session()


def _emit(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log one JSON line so large-scale pipelines can grep and ship to centralized logs."""

    payload = {"event": f"s3.{event}", **fields}
    try:
        line = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        line = json.dumps({"event": f"s3.{event}", "serialization": "fallback"}, default=str)
    logger.log(level, "%s", line)


def _is_transient_client_error(exc: BaseException) -> bool:
    try:
        from botocore.exceptions import ClientError  # type: ignore
    except ImportError:
        return False
    if not isinstance(exc, ClientError):
        return False
    err = exc.response.get("Error", {}) or {}
    code = str(err.get("Code", "") or "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if code in _TRANSIENT_CODES:
        return True
    if status == 503:
        return True
    return False


def _call_with_retry(
    operation: str,
    fn: Callable[[], Any],
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = _DEFAULT_BASE_DELAY_S,
    max_delay_s: float = _DEFAULT_MAX_DELAY_S,
) -> Any:
    """Run fn with exponential backoff and jitter on transient failures."""

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except S3ValidationError:
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_transient_client_error(exc) or attempt >= max_attempts:
                _emit(
                    "operation_failed",
                    level=logging.ERROR,
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                )
                raise S3OperationError(
                    f"{operation} failed after {attempt} attempt(s): {exc}"
                ) from exc
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random() * 0.5
            _emit(
                "operation_retry",
                level=logging.WARNING,
                operation=operation,
                attempt=attempt,
                max_attempts=max_attempts,
                sleep_s=round(delay, 3),
                error_type=type(exc).__name__,
            )
            time.sleep(delay)
    raise S3OperationError(f"{operation} failed: {last_exc}") from last_exc


@dataclass
class SyncUploadResult:
    """Aggregate outcome for large directory uploads (partial success supported)."""

    uris: list[str] = field(default_factory=list)
    skipped: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    bytes_uploaded: int = 0


@dataclass
class SyncDownloadResult:
    """Aggregate outcome for prefix downloads."""

    paths: list[Path] = field(default_factory=list)
    skipped: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)


# Backward-compatible helpers (strict validation).
def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from s3://bucket/key; raises S3ValidationError if invalid."""

    return _parse_s3_uri_strict(uri)


def is_s3_uri(path_or_uri: str) -> bool:
    from ml.infra.s3_validation import is_s3_uri as _is

    return _is(path_or_uri)


def s3_uri_to_local(uri: str, base: Path) -> Path:
    _, key = parse_s3_uri(uri)
    return base / key


class S3Client:
    """S3 wrapper with validation, structured events, and retries for large-scale use."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "maxsight",
        region: str | None = None,
        session=None,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        max_upload_bytes: int | None = MAX_SINGLE_PUT_BYTES,
    ) -> None:
        self.bucket = validate_bucket_name(bucket)
        self.prefix = validate_prefix(prefix)
        self._max_attempts = max(1, int(max_attempts))
        self._max_upload_bytes = max_upload_bytes
        self._session = _boto3_session(session)
        try:
            from botocore.config import Config  # type: ignore

            _bcfg = Config(
                retries={"max_attempts": self._max_attempts, "mode": "adaptive"},
                connect_timeout=60,
                read_timeout=300,
            )
        except Exception:
            _bcfg = None
        self._s3 = self._session.client(
            "s3",
            region_name=region,
            config=_bcfg,
        )
        self._region = region or self._session.region_name or "us-east-1"
        _emit(
            "client_init",
            bucket=self.bucket,
            prefix=self.prefix,
            region=self._region,
            max_attempts=self._max_attempts,
        )

    def _s3_key(self, *parts: str) -> str:
        segments = [self.prefix] + list(parts)
        key = "/".join(s for s in segments if s)
        return validate_object_key(key, field="composed key")

    def _s3_uri(self, *parts: str) -> str:
        key = self._s3_key(*parts)
        return f"s3://{self.bucket}/{key}"

    def upload_file(self, local: Path, s3_key: str, *, overwrite: bool = True) -> str:
        """Upload a single file; optional skip when ETag matches local MD5."""

        s3_key = validate_object_key(s3_key)
        lp = validate_local_file(
            local,
            must_exist=True,
            max_size_bytes=self._max_upload_bytes,
        )
        if not overwrite:
            from botocore.exceptions import ClientError  # type: ignore

            try:

                def _head():
                    return self._s3.head_object(Bucket=self.bucket, Key=s3_key)

                head = _call_with_retry("head_object", _head, max_attempts=self._max_attempts)
                remote_etag = head["ETag"].strip('"')
                local_md5 = _md5(lp)
                if remote_etag == local_md5:
                    _emit("upload_skip_unchanged", key=s3_key, bytes=lp.stat().st_size)
                    return f"s3://{self.bucket}/{s3_key}"
            except S3OperationError as op_err:
                cause = op_err.__cause__
                if isinstance(cause, ClientError):
                    c_err = cast(Any, cause)
                    err_body = getattr(c_err, "response", None) or {}
                    code = str((err_body.get("Error") or {}).get("Code", "") or "")
                    if code in ("404", "NoSuchKey", "NotFound"):
                        _emit("upload_head_missing", key=s3_key)
                    else:
                        raise
                else:
                    raise
            except ClientError as head_exc:
                hb = getattr(head_exc, "response", None) or {}
                code = str((hb.get("Error") or {}).get("Code", "") or "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    _emit("upload_head_missing", key=s3_key)
                else:
                    raise

        sz = lp.stat().st_size

        def _upload():
            self._s3.upload_file(str(lp), self.bucket, s3_key)

        _call_with_retry("upload_file", _upload, max_attempts=self._max_attempts)
        uri = f"s3://{self.bucket}/{s3_key}"
        _emit("upload_ok", key=s3_key, bytes=sz, uri=uri)
        return uri

    def download_file(self, s3_key: str, local: Path, *, overwrite: bool = True) -> Path:
        s3_key = validate_object_key(s3_key)
        local = Path(local)
        if local.exists() and not overwrite:
            _emit("download_skip_existing", key=s3_key, path=str(local))
            return local
        local.parent.mkdir(parents=True, exist_ok=True)

        def _download():
            self._s3.download_file(self.bucket, s3_key, str(local))

        _call_with_retry("download_file", _download, max_attempts=self._max_attempts)
        _emit("download_ok", key=s3_key, path=str(local))
        return local

    def exists(self, s3_key: str) -> bool:
        s3_key = validate_object_key(s3_key)
        from botocore.exceptions import ClientError  # type: ignore

        try:

            def _head():
                self._s3.head_object(Bucket=self.bucket, Key=s3_key)

            _call_with_retry("head_object", _head, max_attempts=self._max_attempts)
            return True
        except S3OperationError as op_err:
            cause = op_err.__cause__
            if isinstance(cause, ClientError):
                c_err = cast(Any, cause)
                err_body = getattr(c_err, "response", None) or {}
                code = str((err_body.get("Error") or {}).get("Code", "") or "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    return False
            raise
        except ClientError as e:
            err_body = getattr(e, "response", None) or {}
            code = str((err_body.get("Error") or {}).get("Code", "") or "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            _emit("exists_error", key=s3_key, code=code, level=logging.ERROR)
            raise

    def list_keys(self, prefix: str, *, max_keys: int | None = None) -> list[str]:
        """List object keys under prefix; cap max_keys to bound memory on huge buckets."""

        if prefix:
            prefix = validate_object_key(prefix, field="list prefix")
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append(obj["Key"])
                if max_keys is not None and len(keys) >= max_keys:
                    _emit(
                        "list_keys_truncated",
                        prefix=prefix,
                        returned=len(keys),
                        max_keys=max_keys,
                    )
                    return keys

        _emit("list_keys_ok", prefix=prefix, count=len(keys))
        return keys

    def sync_upload(
        self,
        local_dir: Path,
        s3_prefix: str,
        *,
        extensions: list[str] | None = None,
        overwrite: bool = False,
        on_progress: Callable[[str], None] | None = None,
        continue_on_error: bool = True,
        max_files: int | None = None,
    ) -> SyncUploadResult:
        """Upload files under local_dir; collect per-file failures when continue_on_error."""

        local_dir = validate_local_dir(local_dir, must_exist=True)
        s3_prefix = validate_object_key(s3_prefix.strip("/") or "root", field="s3_prefix")
        result = SyncUploadResult()
        ext_set = None
        if extensions is not None:
            ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

        paths = sorted(local_dir.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if ext_set is not None and path.suffix.lower() not in ext_set:
                result.skipped += 1
                continue
            if max_files is not None and len(result.uris) + len(result.failed) >= max_files:
                _emit("sync_upload_cap", max_files=max_files, uploaded=len(result.uris))
                break
            try:
                rel_key = sanitize_relative_key(path, base=local_dir)
                key = f"{s3_prefix}/{rel_key}" if rel_key else s3_prefix
                key = validate_object_key(key)
                uri = self.upload_file(path, key, overwrite=overwrite)
                result.uris.append(uri)
                result.bytes_uploaded += path.stat().st_size
                if on_progress:
                    on_progress(uri)
            except (S3ValidationError, S3OperationError, OSError) as exc:
                err = {"path": str(path), "error": str(exc)[:500]}
                result.failed.append(err)
                _emit("sync_upload_file_error", **err, level=logging.ERROR)
                if not continue_on_error:
                    raise
        _emit(
            "sync_upload_done",
            uploaded=len(result.uris),
            skipped=result.skipped,
            failed=len(result.failed),
            bytes=result.bytes_uploaded,
        )
        return result

    def sync_download(
        self,
        s3_prefix: str,
        local_dir: Path,
        *,
        overwrite: bool = False,
        on_progress: Callable[[str], None] | None = None,
        continue_on_error: bool = True,
        max_keys: int | None = None,
    ) -> SyncDownloadResult:
        s3_prefix = validate_object_key(s3_prefix, field="s3_prefix") if s3_prefix else ""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        result = SyncDownloadResult()
        keys = self.list_keys(s3_prefix, max_keys=max_keys)
        for key in keys:
            rel = key[len(s3_prefix) :].lstrip("/") if s3_prefix else key
            local_path = local_dir / rel
            try:
                if local_path.exists() and not overwrite:
                    result.skipped += 1
                    continue
                self.download_file(key, local_path, overwrite=overwrite)
                result.paths.append(local_path)
                if on_progress:
                    on_progress(str(local_path))
            except (S3ValidationError, S3OperationError, OSError) as exc:
                err = {"key": key, "error": str(exc)[:500]}
                result.failed.append(err)
                _emit("sync_download_key_error", **err, level=logging.ERROR)
                if not continue_on_error:
                    raise
        _emit(
            "sync_download_done",
            downloaded=len(result.paths),
            skipped=result.skipped,
            failed=len(result.failed),
        )
        return result

    def medallion_s3_prefix(self, layer: str) -> str:
        if layer not in MEDALLION_LAYERS:
            raise S3ValidationError(f"unknown medallion layer: {layer!r}")
        return self._s3_key("medallion", layer)

    def upload_medallion_layer(
        self,
        layer: str,
        local_root: Path,
        *,
        overwrite: bool = False,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        target = Path(local_root) / layer
        if not target.exists():
            raise S3ValidationError(f"medallion layer directory missing: {target}")
        if not target.is_dir():
            raise S3ValidationError(f"medallion layer path is not a directory: {target}")
        s3_prefix = self.medallion_s3_prefix(layer)
        result = self.sync_upload(
            target,
            s3_prefix,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
        )
        _emit(
            "medallion_upload_layer",
            layer=layer,
            files=len(result.uris),
            failed=len(result.failed),
            bytes=result.bytes_uploaded,
        )
        return {
            "layer": layer,
            "files_uploaded": len(result.uris),
            "files_failed": len(result.failed),
            "bytes_uploaded": result.bytes_uploaded,
            "failures": result.failed[:100],
            "s3_prefix": f"s3://{self.bucket}/{s3_prefix}",
        }

    def download_medallion_layer(
        self,
        layer: str,
        local_root: Path,
        *,
        overwrite: bool = False,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        s3_prefix = self.medallion_s3_prefix(layer)
        dest = Path(local_root) / layer
        dest.mkdir(parents=True, exist_ok=True)
        result = self.sync_download(
            s3_prefix,
            dest,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
        )
        return {
            "layer": layer,
            "files_downloaded": len(result.paths),
            "files_skipped": result.skipped,
            "files_failed": len(result.failed),
            "failures": result.failed[:100],
        }

    def upload_gold_index(self, gold_index_path: Path) -> str:
        lp = validate_local_file(gold_index_path, must_exist=True)
        key = self._s3_key("medallion", "gold", "training_index.json")
        return self.upload_file(lp, key, overwrite=True)

    def download_gold_index(self, dest: Path) -> Path:
        key = self._s3_key("medallion", "gold", "training_index.json")
        return self.download_file(key, dest, overwrite=True)

    def upload_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        run_id: str,
        tag: str = "latest",
    ) -> str:
        if not run_id or not str(run_id).strip():
            raise S3ValidationError("run_id must be non-empty")
        tag = validate_object_key(str(tag), field="tag")
        cp = validate_local_file(
            checkpoint_path, must_exist=True, max_size_bytes=self._max_upload_bytes
        )
        key = self._s3_key("checkpoints", run_id.strip(), tag, cp.name)
        uri = self.upload_file(cp, key, overwrite=True)
        _emit("checkpoint_upload_ok", run_id=run_id, tag=tag, uri=uri)
        return uri

    def download_checkpoint(
        self,
        run_id: str,
        dest_dir: Path,
        *,
        tag: str = "latest",
        filename: str = "best.pt",
    ) -> Path:
        if not run_id or not str(run_id).strip():
            raise S3ValidationError("run_id must be non-empty")
        validate_object_key(filename, field="filename")
        key = self._s3_key("checkpoints", run_id.strip(), tag, filename)
        dest_dir = validate_local_dir(dest_dir, must_exist=False)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        return self.download_file(key, dest, overwrite=True)

    def list_checkpoints(self, run_id: str | None = None) -> list[str]:
        if run_id is None:
            prefix = self._s3_key("checkpoints")
        else:
            if not str(run_id).strip():
                raise S3ValidationError("run_id must be non-empty when provided")
            prefix = self._s3_key("checkpoints", run_id.strip())
        return self.list_keys(prefix)

    def upload_run_artefacts(self, run_dir: Path, run_id: str) -> list[str]:
        rd = validate_local_dir(run_dir, must_exist=True)
        if not run_id or not str(run_id).strip():
            raise S3ValidationError("run_id must be non-empty")
        key_prefix = self._s3_key("runs", run_id.strip())
        result = self.sync_upload(rd, key_prefix, overwrite=True, continue_on_error=True)
        return result.uris


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()
