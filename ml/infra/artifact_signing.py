"""Artifact signing: fail-closed unless the certification manifest is all-pass.

Writes ``<artifact>.sig.json``, an instance of docs/contracts/schemas/artifact_signature.json, signed with
the Ed25519 private key named by configs/security/artifact_signing.yaml (build.private_key_env). Devices
verify it with public keys only (ml/runtime/stage_a/signature_contract.py).

Removed on purpose: the HMAC mode (a verifying device could also sign, and the committed keys were public),
the random-key fallback (it signed with a key nobody had), and KMS ECDSA (devices cannot verify it).
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ml.runtime.stage_a.signature_contract import (
    ALGORITHM,
    INSTANCE_SUFFIX,
    SCHEMA_VERSION,
    SigningPolicy,
    canonical_payload,
    file_sha256,
    key_id,
    load_policy,
    validate_instance,
)


class ManifestNotAllPassedError(RuntimeError):
    """Raised when signing is attempted without an all-pass certification manifest."""


class SigningKeyError(RuntimeError):
    """The build private key is missing or is not an Ed25519 key."""


def _require_all_passed(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ManifestNotAllPassedError("manifest must be a dict")
    if manifest.get("all_passed") is not True:
        raise ManifestNotAllPassedError(
            "refuse to sign: certification manifest all_passed is not True "
            f"(got {manifest.get('all_passed')!r})"
        )
    for cell in manifest.get("cells") or []:
        if cell.get("status") != "passed":
            raise ManifestNotAllPassedError(f"refuse to sign: cell status {cell.get('status')!r} is not passed")


def _load_private_key(private_key_path: Path | str | None, policy: SigningPolicy) -> Ed25519PrivateKey:
    location = private_key_path or os.environ.get(policy.private_key_env, "")
    if not location or not Path(location).is_file():
        raise SigningKeyError(
            f"no signing key: pass private_key_path or set {policy.private_key_env} to a PKCS#8 PEM file"
        )
    key = serialization.load_pem_private_key(Path(location).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningKeyError(f"signing key at {location} is not an Ed25519 private key")
    return key


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sign_artifact(
    artifact_path: Path | str,
    manifest: dict[str, Any],
    *,
    output_dir: Path | str | None = None,
    private_key_path: Path | str | None = None,
    policy: SigningPolicy | None = None,
) -> Path:
    """Sign an all-pass artifact; return the path of ``<artifact>.sig.json``.

    The instance belongs next to the artifact (devices look there); ``output_dir`` is for staging only.
    """
    _require_all_passed(manifest)
    if os.environ.get("MAXSIGHT_SIGNING_MODE", "local").strip().lower() == "kms":
        raise RuntimeError("kms signing mode is not supported: devices verify Ed25519 instances only")
    policy = policy or load_policy()
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    private_key = _load_private_key(private_key_path, policy)

    instance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": artifact.name,
        "artifact_sha256": file_sha256(artifact),
        "algorithm": ALGORITHM,
        "key_id": key_id(private_key.public_key()),
        "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "certification": manifest,
    }
    instance["signature"] = base64.b64encode(private_key.sign(canonical_payload(instance))).decode("ascii")
    errors = validate_instance(instance)
    if errors:
        raise ValueError(f"refuse to write a signature instance that breaks the contract: {errors}")

    out = Path(output_dir) if output_dir else artifact.parent
    out.mkdir(parents=True, exist_ok=True)
    instance_path = out / f"{artifact.name}{INSTANCE_SUFFIX}"
    _write_atomic(instance_path, json.dumps(instance, indent=2, sort_keys=True) + "\n")
    return instance_path
