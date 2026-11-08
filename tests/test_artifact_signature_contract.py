"""Model artifact signatures: Ed25519 instance contract + YAML policy (F7/F8).

Every signed artifact ships ``<artifact>.sig.json``, an instance of
docs/contracts/schemas/artifact_signature.json. The build signs with a private key that never enters
the repository; devices verify with public keys only. configs/security/artifact_signing.yaml decides
whether unsigned artifacts are refused and which dev fixtures are the explicit exceptions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from ml.infra.artifact_signing import ManifestNotAllPassedError, SigningKeyError, sign_artifact  # noqa: E402
from ml.runtime.stage_a.signature_contract import (  # noqa: E402
    SignatureError,
    check_artifact_signature,
    key_id,
    load_policy,
    validate_instance,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "docs" / "contracts" / "schemas" / "artifact_signature.json"


def _manifest(all_passed: bool = True) -> dict:
    return {
        "schema_version": "1.0.0", "artifact_hash": "x", "platform": "torch_ref", "all_passed": all_passed,
        "cells": [{"condition_mode": "glaucoma", "platform": "torch_ref",
                   "status": "passed" if all_passed else "failed", "metrics": {}, "gates_failed": []}],
        "summary": {"passed": 1, "failed": 0, "blocked": 0, "skipped": 0, "xfailed": 0},
    }


@pytest.fixture
def keys(tmp_path: Path):
    """A build private key outside the 'repo' and a device keys dir holding only its public key."""
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "secrets" / "signing.pem"
    private_path.parent.mkdir()
    private_path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    (keys_dir / "current.ed25519.pub").write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    policy = replace(load_policy(), public_keys_dir=keys_dir, unsigned_allowed=())
    return private, private_path, policy


def _artifact(tmp_path: Path, name: str = "active.pt", data: bytes = b"weights-v1") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _signed(tmp_path, keys) -> tuple[Path, Path]:
    _, private_path, policy = keys
    artifact = _artifact(tmp_path)
    instance = sign_artifact(artifact, _manifest(), private_key_path=private_path, policy=policy)
    return artifact, instance


# --- the YAML policy ------------------------------------------------------------------------------

def test_committed_policy_requires_signatures_with_ed25519_and_has_no_unsigned_exceptions():
    policy = load_policy()
    assert policy.algorithm == "ed25519"
    assert policy.require_signature is True
    assert policy.unsigned_allowed == ()
    assert policy.contract == SCHEMA


def test_policy_with_unknown_keys_or_another_algorithm_is_refused(tmp_path):
    text = (REPO / "configs" / "security" / "artifact_signing.yaml").read_text()
    for bad in (text + "\nsurprise: true\n", text.replace("algorithm: ed25519", "algorithm: hmac-sha256")):
        path = tmp_path / "policy.yaml"
        path.write_text(bad)
        with pytest.raises(ValueError):
            load_policy(path)


# --- the instance contract ------------------------------------------------------------------------

def test_signed_instance_satisfies_the_json_schema_and_the_runtime_validator(tmp_path, keys):
    _, instance_path = _signed(tmp_path, keys)
    instance = json.loads(instance_path.read_text())
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(instance)
    assert validate_instance(instance) == []
    assert instance_path.name == "active.pt.sig.json"


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("signature"),
    lambda d: d.update(algorithm="hmac-sha256"),
    lambda d: d.update(artifact_sha256="abc"),
    lambda d: d.update(key_id="not-hex"),
    lambda d: d.update(extra="field"),
    lambda d: d["certification"].update(all_passed=False),
], ids=["no_signature", "hmac", "short_sha", "bad_key_id", "extra_field", "not_certified"])
def test_schema_and_runtime_validator_agree_on_broken_instances(tmp_path, keys, mutate):
    _, instance_path = _signed(tmp_path, keys)
    instance = json.loads(instance_path.read_text())
    mutate(instance)
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))
    assert not validator.is_valid(instance)
    assert validate_instance(instance) != []


# --- signing at build time ------------------------------------------------------------------------

def test_signing_without_a_private_key_refuses_instead_of_inventing_one(tmp_path, keys, monkeypatch):
    _, _, policy = keys
    monkeypatch.delenv(policy.private_key_env, raising=False)
    with pytest.raises(SigningKeyError):
        sign_artifact(_artifact(tmp_path), _manifest(), policy=policy)


def test_signing_reads_the_key_path_from_the_policy_env_var(tmp_path, keys, monkeypatch):
    _, private_path, policy = keys
    monkeypatch.setenv(policy.private_key_env, str(private_path))
    artifact = _artifact(tmp_path)
    sign_artifact(artifact, _manifest(), policy=policy)
    check_artifact_signature(artifact, policy=policy)


def test_signing_refuses_an_uncertified_artifact(tmp_path, keys):
    _, private_path, policy = keys
    with pytest.raises(ManifestNotAllPassedError):
        sign_artifact(_artifact(tmp_path), _manifest(all_passed=False), private_key_path=private_path, policy=policy)


def test_kms_mode_is_refused_because_devices_cannot_verify_it(tmp_path, keys, monkeypatch):
    _, private_path, policy = keys
    monkeypatch.setenv("MAXSIGHT_SIGNING_MODE", "kms")
    with pytest.raises(RuntimeError, match="kms"):
        sign_artifact(_artifact(tmp_path), _manifest(), private_key_path=private_path, policy=policy)


# --- verification on the device -------------------------------------------------------------------

def test_a_correctly_signed_artifact_verifies(tmp_path, keys):
    artifact, _ = _signed(tmp_path, keys)
    check_artifact_signature(artifact, policy=keys[2])


def test_an_unsigned_artifact_is_refused(tmp_path, keys):
    with pytest.raises(SignatureError, match="unsigned"):
        check_artifact_signature(_artifact(tmp_path), policy=keys[2])


def test_an_unsigned_artifact_listed_as_a_dev_exception_is_allowed(tmp_path, keys):
    artifact = _artifact(tmp_path)
    policy = replace(keys[2], unsigned_allowed=(str(artifact),))
    check_artifact_signature(artifact, policy=policy)


def test_a_modified_artifact_is_refused(tmp_path, keys):
    artifact, _ = _signed(tmp_path, keys)
    artifact.write_bytes(b"weights-v2")
    with pytest.raises(SignatureError, match="sha256"):
        check_artifact_signature(artifact, policy=keys[2])


def test_edited_signed_metadata_is_refused(tmp_path, keys):
    artifact, instance_path = _signed(tmp_path, keys)
    instance = json.loads(instance_path.read_text())
    instance["signed_at"] = "2030-01-01T00:00:00Z"
    instance_path.write_text(json.dumps(instance))
    with pytest.raises(SignatureError, match="signature"):
        check_artifact_signature(artifact, policy=keys[2])


def test_a_signature_from_an_untrusted_key_is_refused(tmp_path, keys):
    _, _, policy = keys
    other = tmp_path / "other.pem"
    other.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    artifact = _artifact(tmp_path)
    sign_artifact(artifact, _manifest(), private_key_path=other, policy=policy)
    with pytest.raises(SignatureError, match="key"):
        check_artifact_signature(artifact, policy=policy)


def test_a_forged_signature_under_a_trusted_key_id_is_refused(tmp_path, keys):
    private, _, policy = keys
    artifact, instance_path = _signed(tmp_path, keys)
    instance = json.loads(instance_path.read_text())
    assert instance["key_id"] == key_id(private.public_key())
    instance["signature"] = instance["signature"][::-1][:86] + "=="
    instance_path.write_text(json.dumps(instance))
    with pytest.raises(SignatureError):
        check_artifact_signature(artifact, policy=policy)


def test_an_instance_moved_to_another_artifact_is_refused(tmp_path, keys):
    artifact, instance_path = _signed(tmp_path, keys)
    other = _artifact(tmp_path, "other.pt", data=artifact.read_bytes())
    (tmp_path / "other.pt.sig.json").write_text(instance_path.read_text())
    with pytest.raises(SignatureError, match="artifact_name"):
        check_artifact_signature(other, policy=keys[2])


def test_no_provisioned_public_keys_refuses_everything(tmp_path, keys):
    artifact, _ = _signed(tmp_path, keys)
    empty = tmp_path / "empty_keys"
    empty.mkdir()
    with pytest.raises(SignatureError, match="no trusted public keys"):
        check_artifact_signature(artifact, policy=replace(keys[2], public_keys_dir=empty))


def test_malformed_instance_json_is_refused(tmp_path, keys):
    artifact = _artifact(tmp_path)
    (tmp_path / "active.pt.sig.json").write_text("{not json")
    with pytest.raises(SignatureError):
        check_artifact_signature(artifact, policy=keys[2])


# --- default resolution hook ----------------------------------------------------------------------

def test_default_verify_hook_refuses_an_unsigned_artifact(tmp_path):
    from ml.runtime.stage_a.model_handle import default_verify_signature

    assert default_verify_signature(_artifact(tmp_path)) is False


# --- key provisioning -----------------------------------------------------------------------------

def test_keypair_script_refuses_a_private_key_inside_the_repo(tmp_path):
    from scripts.infra.generate_signing_keypair import generate

    with pytest.raises(ValueError, match="inside the repository"):
        generate("current", REPO / "leak.pem", keys_dir=tmp_path)
    assert not (REPO / "leak.pem").exists()


def test_keypair_script_output_signs_and_verifies(tmp_path, keys):
    from scripts.infra.generate_signing_keypair import generate

    keys_dir = tmp_path / "fresh_keys"
    keys_dir.mkdir()
    private_out = tmp_path / "secret" / "current.pem"
    kid = generate("current", private_out, keys_dir=keys_dir)
    assert oct(private_out.stat().st_mode & 0o777) == "0o600"
    policy = replace(keys[2], public_keys_dir=keys_dir)
    artifact = _artifact(tmp_path)
    instance = json.loads(sign_artifact(artifact, _manifest(), private_key_path=private_out, policy=policy).read_text())
    assert instance["key_id"] == kid
    check_artifact_signature(artifact, policy=policy)
