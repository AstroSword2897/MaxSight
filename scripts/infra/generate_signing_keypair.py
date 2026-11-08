#!/usr/bin/env python3
"""Generate an Ed25519 model-signing key pair (see ml/runtime/stage_a/keys/README.md).

Writes the PRIVATE key (PKCS#8 PEM, mode 0600) to --private-out, which must be outside the repository,
and the PUBLIC key to ml/runtime/stage_a/keys/<slot>.ed25519.pub. Prints the key id.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.runtime.stage_a.signature_contract import key_id, load_policy  # noqa: E402

SLOTS = ("current", "next")


def generate(slot: str, private_out: Path, keys_dir: Path | None = None) -> str:
    """Create the pair; refuse a private path inside the repo or an existing file. Return the key id."""
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    private_out = private_out.resolve()
    if REPO.resolve() in private_out.parents:
        raise ValueError(f"refuse to write a private key inside the repository: {private_out}")
    keys_dir = keys_dir or load_policy().public_keys_dir
    public_out = keys_dir / f"{slot}.ed25519.pub"
    for path in (private_out, public_out):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite {path}")

    key = Ed25519PrivateKey.generate()
    private_out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(private_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    public_out.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                         serialization.PublicFormat.SubjectPublicKeyInfo))
    return key_id(key.public_key())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=SLOTS, required=True)
    parser.add_argument("--private-out", type=Path, required=True, help="private PEM path OUTSIDE the repo")
    args = parser.parse_args(argv)
    try:
        kid = generate(args.slot, args.private_out)
    except (ValueError, FileExistsError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"generated {args.slot} key id={kid}; commit only the .pub file, store the private PEM in the secret store")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
