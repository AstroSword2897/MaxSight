"""Artifact signing fail-closed tests (MAXS-201b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.infra.artifact_signing import ManifestNotAllPassedError, sign_artifact


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
    key = tmp_path / "hmac.key"
    key.write_bytes(b"unit-test-hmac-key-32-bytes-long!!")
    sig = sign_artifact(artifact, _pass_manifest(), output_dir=tmp_path, private_key_pem=key)
    assert sig.is_file()
    assert (tmp_path / "model.bin.certification.json").is_file()
