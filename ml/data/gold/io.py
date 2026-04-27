"""URI-agnostic shard I/O for the gold data plane.

``ShardReader`` is the primary interface.  It encapsulates buffering, backend
selection (local vs S3), and structured error reporting.  All other code
should go through ``ShardReader``; the bare functions below are thin wrappers
kept for internal use by the indexer.

Design contracts
----------------
- A URI is an absolute/relative POSIX path or ``s3://bucket/key``.
- Gold shards are **write-once immutable**: offsets computed at index time must
  remain stable.  The reader does not defend against live mutation.
- Encoding is strictly UTF-8 with LF (``\n``) line endings — no CRLF, no BOM.
- S3 reads use byte-range GETs so only the bytes for a specific line are
  transferred (avoids full-shard downloads on every ``__getitem__`` call).
- The S3 client is created lazily per-process via ``threading.local`` so
  ``DataLoader`` forked workers each get their own client with no contention.
"""

from __future__ import annotations

import hashlib
import threading
from typing import List, Optional, Tuple

_CHUNK = 8192

_s3_local = threading.local()


# ── URI helpers ────────────────────────────────────────────────────────────────

def is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _s3_client():
    if not hasattr(_s3_local, "client"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required to read gold shards from S3 (pip install boto3)"
            ) from exc
        _s3_local.client = boto3.client("s3")
    return _s3_local.client


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    without = uri[len("s3://"):]
    slash = without.find("/")
    if slash == -1:
        return without, ""
    return without[:slash], without[slash + 1:]


# ── GoldIOError ────────────────────────────────────────────────────────────────

class GoldIOError(RuntimeError):
    """Raised when shard I/O fails; carries structured context for pinpointing
    the exact record that caused the failure.

    Attributes
    ----------
    uri :
        Shard URI (local path or s3://).
    idx :
        Zero-based index of the record within the dataset (not the shard).
    offset :
        Byte offset within the shard file.
    raw_prefix :
        Up to 80 bytes of raw data at the offset (empty on I/O-level errors).
    reason :
        Human-readable description of the failure.
    shard_sha256 :
        Expected SHA-256 from meta (if available); helps confirm shard integrity.
    line_schema_version :
        Expected JSONL schema version (if available).
    """

    def __init__(
        self,
        *,
        uri: str,
        idx: int,
        offset: int,
        raw_prefix: bytes,
        reason: str,
        shard_sha256: Optional[str] = None,
        line_schema_version: Optional[str] = None,
    ) -> None:
        self.uri = uri
        self.idx = idx
        self.offset = offset
        self.raw_prefix = raw_prefix
        self.reason = reason
        self.shard_sha256 = shard_sha256
        self.line_schema_version = line_schema_version
        preview = repr(raw_prefix[:80]) if raw_prefix else "<none>"
        meta_ctx = ""
        if shard_sha256:
            meta_ctx += f" expected_shard_sha256={shard_sha256[:12]}…"
        if line_schema_version:
            meta_ctx += f" schema={line_schema_version!r}"
        super().__init__(
            f"GoldIOError: {reason} | "
            f"uri={uri!r} idx={idx} offset={offset}{meta_ctx} raw_prefix={preview}"
        )


# ── ShardReader ────────────────────────────────────────────────────────────────

class ShardReader:
    """Reads a single shard URI; encapsulates local-file vs S3 backend.

    Usage
    -----
    reader = ShardReader(uri, shard_sha256="...", line_schema_version="1.0")
    offsets = reader.index_line_starts()
    raw_bytes = reader.read_at(idx=42, offset=offsets[42])

    Parameters
    ----------
    uri :
        Local path or ``s3://`` URI.
    shard_sha256 :
        Expected SHA-256 from meta (embedded in ``GoldIOError`` on failure).
    line_schema_version :
        Expected schema version (embedded in ``GoldIOError`` on failure).
    """

    def __init__(
        self,
        uri: str,
        *,
        shard_sha256: Optional[str] = None,
        line_schema_version: Optional[str] = None,
    ) -> None:
        self.uri = str(uri)
        self.shard_sha256 = shard_sha256
        self.line_schema_version = line_schema_version
        self._is_s3 = is_s3(self.uri)

    def index_line_starts(self) -> List[int]:
        """Return the byte offset of every non-empty line in the shard."""
        if self._is_s3:
            return _index_s3(self.uri)
        return _index_local(self.uri)

    def read_at(self, idx: int, offset: int) -> bytes:
        """Return raw bytes of the line at ``offset``.

        Raises ``GoldIOError`` with full context on corruption or I/O errors.
        """
        try:
            if self._is_s3:
                raw = _readline_s3(self.uri, offset)
            else:
                raw = _readline_local(self.uri, offset)
        except GoldIOError:
            raise
        except Exception as exc:
            raise GoldIOError(
                uri=self.uri,
                idx=idx,
                offset=offset,
                raw_prefix=b"",
                reason=str(exc),
                shard_sha256=self.shard_sha256,
                line_schema_version=self.line_schema_version,
            ) from exc
        if not raw.strip():
            raise GoldIOError(
                uri=self.uri,
                idx=idx,
                offset=offset,
                raw_prefix=raw[:80],
                reason="empty or whitespace-only line at offset",
                shard_sha256=self.shard_sha256,
                line_schema_version=self.line_schema_version,
            )
        return raw

    def verify_sha256(self, expected: str) -> None:
        """Re-hash the entire shard; raise ``GoldIOError`` on mismatch.

        This is an optional pre-training integrity gate — not called on
        every read.  Use ``verify_shards=True`` in ``GoldManifestDataset``
        to trigger it at startup.
        """
        h = hashlib.sha256()
        if self._is_s3:
            client = _s3_client()
            bucket, key = _parse_s3_uri(self.uri)
            obj = client.get_object(Bucket=bucket, Key=key)
            stream = obj["Body"]
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        else:
            with open(self.uri, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise GoldIOError(
                uri=self.uri,
                idx=-1,
                offset=-1,
                raw_prefix=b"",
                reason=f"sha256 mismatch: expected {expected!r}, got {actual!r}",
                shard_sha256=expected,
                line_schema_version=self.line_schema_version,
            )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _index_local(uri: str) -> List[int]:
    offsets: List[int] = []
    with open(uri, "rb") as f:
        while True:
            p = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(p)
    return offsets


def _index_s3(uri: str) -> List[int]:
    client = _s3_client()
    bucket, key = _parse_s3_uri(uri)
    obj = client.get_object(Bucket=bucket, Key=key)
    stream = obj["Body"]
    offsets: List[int] = []
    pos = 0
    buf = b""
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            if buf.strip():
                offsets.append(pos)
            break
        for byte in chunk:
            b = bytes([byte])
            if b == b"\n":
                if buf.strip():
                    offsets.append(pos)
                pos += len(buf) + 1
                buf = b""
            else:
                buf += b
    return offsets


def _readline_local(uri: str, offset: int) -> bytes:
    with open(uri, "rb") as f:
        f.seek(offset)
        return f.readline()


def _readline_s3(uri: str, offset: int) -> bytes:
    client = _s3_client()
    bucket, key = _parse_s3_uri(uri)
    result = b""
    start = offset
    while True:
        range_end = start + _CHUNK - 1
        response = client.get_object(
            Bucket=bucket, Key=key, Range=f"bytes={start}-{range_end}"
        )
        data: bytes = response["Body"].read()
        if not data:
            break
        combined = result + data
        nl = combined.find(b"\n", len(result))
        if nl != -1:
            result = combined[: nl + 1]
            break
        result = combined
        if len(data) < _CHUNK:
            break
        start = offset + len(result)
    return result


# ── Convenience wrappers (used by dataset.py) ──────────────────────────────────

def index_jsonl_line_starts(uri: str) -> List[int]:
    """Return byte offsets of every non-empty line in a shard URI."""
    return ShardReader(uri).index_line_starts()


def readline_at_offset(uri: str, idx: int, offset: int) -> bytes:
    """Return raw bytes at ``offset`` in ``uri``, raising ``GoldIOError`` on failure."""
    return ShardReader(uri).read_at(idx, offset)


def verify_shard_sha256(uri: str, expected_sha256: str) -> None:
    """Verify shard integrity; raise ``GoldIOError`` on mismatch."""
    ShardReader(uri, shard_sha256=expected_sha256).verify_sha256(expected_sha256)
