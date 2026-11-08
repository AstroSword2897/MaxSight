"""Session tokens are signed only with a real secret.

The module fell back to the public string "change_me_in_production" when MAXSIGHT_SECRET_KEY was
unset, so anyone could mint tokens a misconfigured deployment would accept.
"""

from __future__ import annotations

import base64
import json

import pytest
from ml.auth import token

STRONG = "k" * 48


def test_round_trip_with_a_configured_secret(monkeypatch):
    monkeypatch.setenv("MAXSIGHT_SECRET_KEY", STRONG)
    assert token.verify_token(token.make_token({"sub": "u1"}))["sub"] == "u1"


@pytest.mark.parametrize("value", [None, "", "change_me_in_production", "short"], ids=["unset", "empty", "placeholder", "too_short"])
def test_signing_refuses_a_missing_or_weak_secret(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MAXSIGHT_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("MAXSIGHT_SECRET_KEY", value)
    with pytest.raises(RuntimeError, match="MAXSIGHT_SECRET_KEY"):
        token.make_token({"sub": "u1"})


def test_verification_refuses_when_no_secret_is_configured(monkeypatch):
    monkeypatch.setenv("MAXSIGHT_SECRET_KEY", STRONG)
    signed = token.make_token({"sub": "u1"})
    monkeypatch.delenv("MAXSIGHT_SECRET_KEY")
    with pytest.raises(RuntimeError, match="MAXSIGHT_SECRET_KEY"):
        token.verify_token(signed)


def test_a_token_forged_with_the_old_default_is_rejected(monkeypatch):
    import hashlib
    import hmac

    payload = json.dumps({"sub": "attacker", "exp": 4102444800}, separators=(",", ":")).encode()
    sig = hmac.new(b"change_me_in_production", payload, hashlib.sha256).digest()
    forged = (base64.urlsafe_b64encode(payload).rstrip(b"=") + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()
    monkeypatch.setenv("MAXSIGHT_SECRET_KEY", STRONG)
    with pytest.raises(ValueError, match="Bad signature"):
        token.verify_token(forged)


def test_tampered_payload_is_rejected(monkeypatch):
    monkeypatch.setenv("MAXSIGHT_SECRET_KEY", STRONG)
    payload_b64, sig_b64 = token.make_token({"sub": "u1"}).split(".")
    other = base64.urlsafe_b64encode(json.dumps({"sub": "admin", "exp": 4102444800}).encode()).rstrip(b"=").decode()
    with pytest.raises(ValueError):
        token.verify_token(f"{other}.{sig_b64}")
