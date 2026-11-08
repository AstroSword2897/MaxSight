"""Stage A key bundle (MAXS-203a): public keys only, never secret material.

The directory used to hold HMAC keys in a public repository; an HMAC verification key is also a signing
key, so anyone could forge model signatures. This guard fails if secret material comes back.
"""

from __future__ import annotations

from pathlib import Path

KEYS = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "keys"
SECRET_MARKERS = (b"PRIVATE KEY", b"BEGIN OPENSSH", b"BEGIN EC PRIVATE")


def test_readme_documents_the_trust_window() -> None:
    text = (KEYS / "README.md").read_text(encoding="utf-8")
    assert "current.ed25519.pub" in text and "next.ed25519.pub" in text


def test_no_symmetric_or_private_key_material_is_committed() -> None:
    for path in KEYS.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".hmac", ".key", ".pem"}, f"secret-looking file committed: {path.name}"
        assert not any(marker in path.read_bytes() for marker in SECRET_MARKERS), f"private key in {path.name}"


def test_only_ed25519_public_keys_and_docs_are_present() -> None:
    names = {p.name for p in KEYS.iterdir() if p.is_file()}
    assert names <= {"README.md", "current.ed25519.pub", "next.ed25519.pub"}, names
