"""Artifact signing fail-closed tests (MAXS-201b); full contract coverage in test_artifact_signature_contract.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from ml.infra.artifact_signing import ManifestNotAllPassedError, sign_artifact  # noqa: E402


def _pass_manifest() -> dict:
    return {
        "schema_version": "1.0.0",
        "artifact_hash": "x",
        "platform": "torch_ref",
        "cells": [
            {
                "condition_mode": "glaucoma",
                "platform": "torch_ref",
                "status": "passed",
                "metrics": {"hazard_recall": 0.99},
                "gates_failed": [],
            }
        ],
        "all_passed": True,
        "summary": {"passed": 1, "failed": 0, "blocked": 0, "skipped": 0, "xfailed": 0},
    }


def test_sign_refuses_blocked_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    bad = _pass_manifest()
    bad["all_passed"] = False
    bad["cells"][0]["status"] = "blocked_missing_hazard_labels"
    with pytest.raises(ManifestNotAllPassedError):
        sign_artifact(artifact, bad, output_dir=tmp_path)


def test_sign_succeeds_on_all_pass(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    key = tmp_path / "signing.pem"
    key.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    instance = sign_artifact(artifact, _pass_manifest(), private_key_path=key)
    assert instance == tmp_path / "model.bin.sig.json"
    assert json.loads(instance.read_text())["certification"]["all_passed"] is True
