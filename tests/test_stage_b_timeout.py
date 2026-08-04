"""Stage B timeout / offline contract tests (MAXS-402)."""

from __future__ import annotations

import time

from app.connectivity import ConnectivityState
from app.stage_b import STAGE_B_OFFLINE_MESSAGE, STAGE_B_TIMEOUT_MESSAGE, StageBClient


def test_offline_short_circuits_without_transport() -> None:
    called = {"n": 0}

    def transport(cid: str, body: dict) -> dict:
        called["n"] += 1
        return {"ok": True}

    client = StageBClient(transport=transport)
    result = client.request("describe", connectivity=ConnectivityState.OFFLINE)
    assert result.offline is True
    assert result.message == STAGE_B_OFFLINE_MESSAGE
    assert called["n"] == 0
    assert result.correlation_id


def test_timeout_no_silent_retry() -> None:
    calls = {"n": 0}
    clock = {"t": 0.0}

    def transport(cid: str, body: dict) -> dict:
        calls["n"] += 1
        clock["t"] += 5.0
        return {"ok": True}

    client = StageBClient(timeout_s=4.0, transport=transport, clock=lambda: clock["t"])
    result = client.request("describe", connectivity=ConnectivityState.ONLINE_DEGRADED)
    assert result.timed_out is True
    assert result.message == STAGE_B_TIMEOUT_MESSAGE
    assert calls["n"] == 1  # no automatic retry
