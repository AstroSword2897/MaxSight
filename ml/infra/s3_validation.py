"""Input validation for S3 paths, keys, and local files used by large-scale sync.

S3 object keys are UTF-8 bytes capped at 1024; bucket names follow DNS rules.
Callers validate before network I/O so failures are explicit and loggable.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

# AWS limits relevant to this client (see S3 docs).
MAX_OBJECT_KEY_BYTES = 1024
MAX_BUCKET_NAME_LEN = 63
MIN_BUCKET_NAME_LEN = 3
# Single PUT max; multipart used above this by boto3 upload_file.
MAX_SINGLE_PUT_BYTES = 5 * 1024 * 1024 * 1024 - 1

_BUCKET_LABEL = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$")


class S3ValidationError(ValueError):
    """Raised when bucket, key, URI, or local path fails validation."""


def validate_bucket_name(name: str) -> str:
    """Return stripped bucket name or raise if it violates S3 DNS naming rules."""

    if name is None or not isinstance(name, str):
        raise S3ValidationError("bucket name must be a non-empty string")
    n = name.strip().lower()
    if not n:
        raise S3ValidationError("bucket name is empty")
    if len(n) < MIN_BUCKET_NAME_LEN or len(n) > MAX_BUCKET_NAME_LEN:
        raise S3ValidationError(
            f"bucket name length must be {MIN_BUCKET_NAME_LEN}-{MAX_BUCKET_NAME_LEN}, got {len(n)}"
        )
    if not re.match(r"^[a-z0-9][a-z0-9\-.]*[a-z0-9]$", n) or ".." in n:
        raise S3ValidationError(f"invalid bucket name: {name!r}")
    labels = n.split(".")
    for label in labels:
        if not _BUCKET_LABEL.match(label) or label.startswith("-") or label.endswith("-"):
            raise S3ValidationError(f"invalid bucket label in {name!r}")
    if n.startswith("xn--") or n.endswith("-s3alias"):
        raise S3ValidationError(f"reserved bucket pattern: {name!r}")
    return n


def _key_byte_length(key: str) -> int:
    return len(key.encode("utf-8"))


def validate_object_key(key: str, *, field: str = "key") -> str:
    """Return key or raise if empty, too long, or contains disallowed control bytes."""

    if key is None or not isinstance(key, str):
        raise S3ValidationError(f"{field} must be a string")
    k = key.strip()
    if not k:
        raise S3ValidationError(f"{field} is empty")
    blen = _key_byte_length(k)
    if blen > MAX_OBJECT_KEY_BYTES:
        raise S3ValidationError(
            f"{field} exceeds {MAX_OBJECT_KEY_BYTES} UTF-8 bytes ({blen} bytes)"
        )
    for ch in k:
        code = ord(ch)
        if code < 32 or code == 127:
            raise S3ValidationError(f"{field} contains disallowed control character")
    if k.startswith("\\") or "..\\" in k or k.startswith("../") or "/../" in k:
        raise S3ValidationError(f"{field} must not contain path-traversal segments")
    return k


def validate_prefix(prefix: str) -> str:
    """Normalize prefix: no leading slash, strip trailing slashes except root."""

    if prefix is None:
        return ""
    p = prefix.strip().strip("/")
    if not p:
        return ""
    validate_object_key(p, field="prefix")
    return p


def validate_local_file(
    path: Path,
    *,
    must_exist: bool = True,
    max_size_bytes: Optional[int] = None,
) -> Path:
    """Ensure path is a readable file and optionally under a size cap."""

    p = Path(path).resolve()
    if must_exist and not p.exists():
        raise S3ValidationError(f"local path does not exist: {p}")
    if must_exist and not p.is_file():
        raise S3ValidationError(f"local path is not a file: {p}")
    if must_exist and max_size_bytes is not None:
        sz = p.stat().st_size
        if sz > max_size_bytes:
            raise S3ValidationError(
                f"file exceeds max_size_bytes={max_size_bytes}: {p} ({sz} bytes)"
            )
    return p


def validate_local_dir(path: Path, *, must_exist: bool = True) -> Path:
    """Ensure path is a directory when must_exist is True."""

    p = Path(path).resolve()
    if must_exist:
        if not p.exists():
            raise S3ValidationError(f"local directory does not exist: {p}")
        if not p.is_dir():
            raise S3ValidationError(f"local path is not a directory: {p}")
    return p


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key); key may be empty."""

    if uri is None or not isinstance(uri, str):
        raise S3ValidationError("URI must be a string")
    u = uri.strip()
    if not u.lower().startswith("s3://"):
        raise S3ValidationError(f"not an S3 URI: {uri!r}")
    rest = u[5:]
    if "/" not in rest:
        bucket = validate_bucket_name(rest.rstrip("/"))
        return bucket, ""
    bucket_part, key_part = rest.split("/", 1)
    bucket = validate_bucket_name(bucket_part)
    key = key_part.lstrip("/")
    if key:
        validate_object_key(key, field="URI key")
    return bucket, key


def is_s3_uri(path_or_uri: str) -> bool:
    """True if the string looks like s3:// (does not fully validate)."""

    s = str(path_or_uri).strip().lower()
    return s.startswith("s3://") and len(s) > 5


def sanitize_relative_key(path: Path, *, base: Path) -> str:
    """Build an S3-safe key segment for a file path under base; blocks escapes."""

    try:
        resolved_rel = path.resolve().relative_to(base.resolve())
    except ValueError as e:
        raise S3ValidationError(f"path is not under base: {path}") from e
    parts = []
    for part in resolved_rel.parts:
        if part in (".", ".."):
            raise S3ValidationError(f"invalid path segment in key: {part!r}")
        norm = unicodedata.normalize("NFC", part)
        parts.append(norm)
    key = "/".join(parts).replace("\\", "/")
    if key:
        validate_object_key(key, field="relative key")
    return key
