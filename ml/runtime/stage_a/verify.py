"""On-device signature verification for Stage A artifacts."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

KEYS_DIR = Path(__file__).resolve().parent / "keys"


def _artifact_digest(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.digest()


def load_trust_window_hmac_keys(keys_dir: Path | None = None) -> list[bytes]:
    root = keys_dir or KEYS_DIR
    keys: list[bytes] = []
    for name in ("current.hmac", "next.hmac"):
        path = root / name
        if path.is_file():
            keys.append(path.read_bytes().strip())
    return keys


def verify_artifact_signature(
    artifact_path: Path | str,
    signature_path: Path | str | None = None,
    *,
    keys_dir: Path | None = None,
    signature_bytes: bytes | None = None,
) -> bool:
    """Return True if signature verifies against any key in the trust window."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        return False
    if signature_bytes is None:
        if signature_path is None:
            signature_path = Path(str(artifact) + ".sig")
        sig_path = Path(signature_path)
        if not sig_path.is_file():
            return False
        signature_bytes = sig_path.read_bytes()
    digest = _artifact_digest(artifact)
    for key in load_trust_window_hmac_keys(keys_dir):
        expected = hmac.new(key, digest, hashlib.sha256).digest()
        if hmac.compare_digest(expected, signature_bytes):
            return True
    return False
