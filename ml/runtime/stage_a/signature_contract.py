"""Artifact signature contract: policy (YAML), instance validation, Ed25519 verification.

An artifact ``model.pt`` is trusted only if ``model.pt.sig.json`` is a valid instance of
docs/contracts/schemas/artifact_signature.json whose Ed25519 signature verifies under a public key in the
trust window of configs/security/artifact_signing.yaml. Devices hold public keys only; the private key
stays in the build environment. No network access (Stage A isolation).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = REPO_ROOT / "configs" / "security" / "artifact_signing.yaml"
POLICY_ENV = "MAXSIGHT_SIGNING_POLICY"
INSTANCE_SUFFIX = ".sig.json"
SCHEMA_VERSION = "1.0.0"
ALGORITHM = "ed25519"

_INSTANCE_KEYS = frozenset(
    {"schema_version", "artifact_name", "artifact_sha256", "algorithm", "key_id", "signed_at", "certification", "signature"}
)
_PATTERNS = {
    "artifact_sha256": re.compile(r"^[0-9a-f]{64}$"),
    "key_id": re.compile(r"^[0-9a-f]{16}$"),
    "signed_at": re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"),
    "signature": re.compile(r"^[A-Za-z0-9+/]{86}==$"),
}


class SignatureError(Exception):
    """The artifact may not be trusted; the message says why."""


@dataclass(frozen=True)
class SigningPolicy:
    algorithm: str
    contract: Path
    private_key_env: str
    public_keys_dir: Path
    trust_window: tuple[str, ...]
    require_signature: bool
    unsigned_allowed: tuple[str, ...]


def _expect_keys(section: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(section, dict) or set(section) != keys:
        got = sorted(section) if isinstance(section, dict) else type(section).__name__
        raise ValueError(f"signing policy {where}: expected keys {sorted(keys)}, got {got}")
    return section


def load_policy(path: Path | str | None = None) -> SigningPolicy:
    """Read and strictly validate the signing policy (MAXSIGHT_SIGNING_POLICY overrides the default path)."""
    policy_path = Path(path or os.environ.get(POLICY_ENV) or DEFAULT_POLICY_PATH)
    raw = _expect_keys(yaml.safe_load(policy_path.read_text(encoding="utf-8")),
                       {"schema_version", "algorithm", "contract", "build", "device"}, "top level")
    if raw["schema_version"] != 1:
        raise ValueError(f"signing policy schema_version must be 1, got {raw['schema_version']!r}")
    if raw["algorithm"] != ALGORITHM:
        raise ValueError(f"signing policy algorithm must be {ALGORITHM!r} (devices verify nothing else)")
    build = _expect_keys(raw["build"], {"private_key_env"}, "build")
    device = _expect_keys(raw["device"], {"public_keys_dir", "trust_window", "require_signature", "unsigned_allowed"}, "device")
    if not isinstance(device["require_signature"], bool):
        raise ValueError("signing policy device.require_signature must be true or false")
    for name in ("trust_window", "unsigned_allowed"):
        if not isinstance(device[name], list) or not all(isinstance(v, str) for v in device[name]):
            raise ValueError(f"signing policy device.{name} must be a list of strings")
    return SigningPolicy(
        algorithm=raw["algorithm"],
        contract=REPO_ROOT / raw["contract"],
        private_key_env=str(build["private_key_env"]),
        public_keys_dir=REPO_ROOT / device["public_keys_dir"],
        trust_window=tuple(device["trust_window"]),
        require_signature=device["require_signature"],
        unsigned_allowed=tuple(device["unsigned_allowed"]),
    )


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def canonical_payload(instance: dict[str, Any]) -> bytes:
    """The bytes the signature covers: every field except ``signature``, canonical JSON."""
    body = {k: v for k, v in instance.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def validate_instance(instance: Any) -> list[str]:
    """Runtime mirror of artifact_signature.json (kept dependency-free); empty list means valid."""
    if not isinstance(instance, dict):
        return ["instance must be a JSON object"]
    errors = [f"missing field {k}" for k in sorted(_INSTANCE_KEYS - set(instance))]
    errors += [f"unexpected field {k}" for k in sorted(set(instance) - _INSTANCE_KEYS)]
    if instance.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if instance.get("algorithm", ALGORITHM) != ALGORITHM:
        errors.append(f"algorithm must be {ALGORITHM}")
    name = instance.get("artifact_name", "x")
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        errors.append("artifact_name must be a bare file name")
    for field, pattern in _PATTERNS.items():
        value = instance.get(field)
        if field in instance and not (isinstance(value, str) and pattern.match(value)):
            errors.append(f"{field} is malformed")
    cert = instance.get("certification", {"all_passed": True})
    if not isinstance(cert, dict) or cert.get("all_passed") is not True:
        errors.append("certification.all_passed must be true")
    return errors


def trusted_public_keys(policy: SigningPolicy) -> dict[str, Ed25519PublicKey]:
    keys: dict[str, Ed25519PublicKey] = {}
    for name in policy.trust_window:
        path = policy.public_keys_dir / name
        if not path.is_file():
            continue
        loaded = serialization.load_pem_public_key(path.read_bytes())
        if not isinstance(loaded, Ed25519PublicKey):
            raise SignatureError(f"trust-window key {name} is not an Ed25519 public key")
        keys[key_id(loaded)] = loaded
    return keys


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unsigned_allowed(artifact: Path, policy: SigningPolicy) -> bool:
    resolved = artifact.resolve()
    return any((REPO_ROOT / entry).resolve() == resolved for entry in policy.unsigned_allowed)


def _read_instance(path: Path) -> dict[str, Any]:
    try:
        instance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SignatureError(f"unreadable signature instance {path.name}: {exc}") from exc
    errors = validate_instance(instance)
    if errors:
        raise SignatureError(f"signature instance breaks the contract: {errors[:5]}")
    return instance


def check_artifact_signature(artifact_path: Path | str, *, policy: SigningPolicy | None = None) -> None:
    """Raise SignatureError unless the artifact may be trusted under the policy."""
    policy = policy or load_policy()
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise SignatureError(f"artifact not found: {artifact}")
    instance_path = Path(str(artifact) + INSTANCE_SUFFIX)
    if not instance_path.is_file():
        if not policy.require_signature or _unsigned_allowed(artifact, policy):
            return
        raise SignatureError(f"unsigned artifact refused: {artifact.name} has no {INSTANCE_SUFFIX}")
    instance = _read_instance(instance_path)
    if instance["artifact_name"] != artifact.name:
        raise SignatureError(f"artifact_name {instance['artifact_name']!r} does not match {artifact.name!r}")
    if instance["artifact_sha256"] != file_sha256(artifact):
        raise SignatureError("artifact sha256 does not match the signed instance (modified after signing)")
    keys = trusted_public_keys(policy)
    if not keys:
        raise SignatureError("no trusted public keys provisioned in the trust window")
    public_key = keys.get(instance["key_id"])
    if public_key is None:
        raise SignatureError(f"signing key {instance['key_id']} is not in the trust window")
    try:
        public_key.verify(base64.b64decode(instance["signature"], validate=True), canonical_payload(instance))
    except (InvalidSignature, ValueError) as exc:
        raise SignatureError("Ed25519 signature does not verify") from exc


def verify_artifact_signature(artifact_path: Path | str, *, policy: SigningPolicy | None = None) -> bool:
    try:
        check_artifact_signature(artifact_path, policy=policy)
    except SignatureError:
        return False
    return True
