"""On-device signature verification for Stage A artifacts.

Ed25519 over the artifact signature instance contract; see ml/runtime/stage_a/signature_contract.py and
configs/security/artifact_signing.yaml. The former HMAC trust window was removed: an HMAC verification
key is also a signing key, and the committed keys were public.
"""

from __future__ import annotations

from ml.runtime.stage_a.signature_contract import (
    SignatureError,
    check_artifact_signature,
    verify_artifact_signature,
)

__all__ = ["SignatureError", "check_artifact_signature", "verify_artifact_signature"]
