"""Signature verification inside model resolution (MAXS-203b), Ed25519 instance contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from ml.infra.artifact_signing import sign_artifact  # noqa: E402
from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE  # noqa: E402
from ml.runtime.stage_a.model_handle import ResolutionState, resolve_model_handle  # noqa: E402
from ml.runtime.stage_a.signature_contract import load_policy, verify_artifact_signature  # noqa: E402
from ml.runtime.stage_a.types import CameraFrame, HazardResult  # noqa: E402

MANIFEST = {"all_passed": True, "cells": []}


def _ok_probe(frame: CameraFrame, artifact_path: Path) -> HazardResult:
    return HazardResult(
        event_type="none", urgency=0, direction="center", distance_zone="far", confidence=0.5,
        uncertainty=0.5, latency_ms=1.0, model_version="probe", model_hash=artifact_path.name,
        condition_mode="none", timestamp_source=frame.timestamp, timestamp_emit=frame.timestamp,
    )


@pytest.fixture
def signer(tmp_path: Path):
    private = Ed25519PrivateKey.generate()
    key_file = tmp_path / "signing.pem"
    key_file.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    (keys_dir / "current.ed25519.pub").write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    policy = replace(load_policy(), public_keys_dir=keys_dir, unsigned_allowed=())

    def sign(path: Path) -> None:
        sign_artifact(path, MANIFEST, private_key_path=key_file, policy=policy)

    return sign, (lambda p: verify_artifact_signature(p, policy=policy))


def test_signed_active_is_selected(tmp_path: Path, signer) -> None:
    sign, verify = signer
    active = tmp_path / "active.bin"
    active.write_bytes(b"good-active")
    sign(active)
    handle = resolve_model_handle(active_path=active, lkg_path=tmp_path / "none", smoke_probe=_ok_probe,
                                  verify_signature=verify)
    assert handle.state is ResolutionState.ACTIVE


def test_tampered_active_falls_to_signed_lkg(tmp_path: Path, signer) -> None:
    sign, verify = signer
    active, lkg = tmp_path / "active.bin", tmp_path / "lkg.bin"
    active.write_bytes(b"good-active")
    lkg.write_bytes(b"lkg-bytes")
    sign(active)
    sign(lkg)
    active.write_bytes(b"hood-active")  # one byte changed after signing
    assert verify(active) is False
    handle = resolve_model_handle(active_path=active, lkg_path=lkg, smoke_probe=_ok_probe, verify_signature=verify)
    assert handle.state is ResolutionState.LAST_KNOWN_GOOD


def test_unsigned_active_and_tampered_lkg_refuse(tmp_path: Path, signer) -> None:
    sign, verify = signer
    active, lkg = tmp_path / "active.bin", tmp_path / "lkg.bin"
    active.write_bytes(b"unsigned")
    lkg.write_bytes(b"lkg-bytes")
    sign(lkg)
    lkg.write_bytes(b"lkg-BYTEZ")
    handle = resolve_model_handle(active_path=active, lkg_path=lkg, smoke_probe=_ok_probe, verify_signature=verify)
    assert handle.state is ResolutionState.REFUSED
    assert handle.refusal_message == HAZARD_UNAVAILABLE_MESSAGE
