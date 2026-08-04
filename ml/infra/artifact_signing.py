"""Artifact signing: fail-closed unless certification manifest all_passed."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any


class ManifestNotAllPassedError(RuntimeError):
    """Raised when signing is attempted without an all-pass certification manifest."""


def _require_all_passed(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ManifestNotAllPassedError("manifest must be a dict")
    if manifest.get("all_passed") is not True:
        raise ManifestNotAllPassedError(
            "refuse to sign: certification manifest all_passed is not True "
            f"(got {manifest.get('all_passed')!r})"
        )
    cells = manifest.get("cells") or []
    for cell in cells:
        if cell.get("status") != "passed":
            raise ManifestNotAllPassedError(
                f"refuse to sign: cell status {cell.get('status')!r} is not passed"
            )


def _artifact_digest(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.digest()


def _local_hmac_key(key_path: Path | None = None) -> bytes:
    if key_path and key_path.is_file():
        return key_path.read_bytes().strip()
    return secrets.token_bytes(32)


def sign_artifact(
    artifact_path: Path | str,
    manifest: dict[str, Any],
    *,
    output_dir: Path | str | None = None,
    private_key_pem: Path | str | None = None,
) -> Path:
    """Sign artifact only when manifest is all-pass. Returns signature path.

    Local mode uses HMAC-SHA256 (stdlib). Set MAXSIGHT_SIGNING_MODE=kms for AWS KMS.
    """
    _require_all_passed(manifest)
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    out = Path(output_dir) if output_dir else artifact.parent
    out.mkdir(parents=True, exist_ok=True)

    mode = os.environ.get("MAXSIGHT_SIGNING_MODE", "local").strip().lower()
    digest = _artifact_digest(artifact)

    if mode == "kms":
        import boto3  # type: ignore[import-not-found]  # optional AWS dep stubs missing in CI venv

        key_id = os.environ.get("MAXSIGHT_KMS_SIGNING_KEY_ID", "")
        if not key_id:
            raise RuntimeError("MAXSIGHT_KMS_SIGNING_KEY_ID required for kms mode")
        client = boto3.client("kms")
        resp = client.sign(
            KeyId=key_id,
            Message=digest,
            MessageType="DIGEST",
            SigningAlgorithm="ECDSA_SHA_256",
        )
        signature = resp["Signature"]
        meta: dict[str, Any] = {"mode": "kms", "key_id": key_id}
    else:
        key = _local_hmac_key(Path(private_key_pem) if private_key_pem else None)
        signature = hmac.new(key, digest, hashlib.sha256).digest()
        meta = {"mode": "local", "algorithm": "hmac-sha256"}

    sig_path = out / f"{artifact.name}.sig"
    man_path = out / f"{artifact.name}.certification.json"
    tmp_sig = out / f".{artifact.name}.sig.tmp"
    tmp_man = out / f".{artifact.name}.certification.json.tmp"
    tmp_sig.write_bytes(signature)
    payload = {"manifest": manifest, "signing": meta, "artifact_sha256": digest.hex()}
    tmp_man.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_sig.replace(sig_path)
    tmp_man.replace(man_path)
    return sig_path


def verify_local_hmac(artifact_path: Path, signature: bytes, key: bytes) -> bool:
    digest = _artifact_digest(Path(artifact_path))
    expected = hmac.new(key, digest, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)
