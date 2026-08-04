"""Stage B client: explicit timeout, no silent retry, correlation_id always."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.connectivity.monitor import ConnectivityState
from app.stage_b.messages import STAGE_B_OFFLINE_MESSAGE, STAGE_B_TIMEOUT_MESSAGE


@dataclass
class StageBResult:
    ok: bool
    correlation_id: str
    message: str | None = None
    payload: dict[str, Any] | None = None
    timed_out: bool = False
    offline: bool = False


class StageBClient:
    """Client-side Stage B trigger with 4s timeout and no automatic retry."""

    def __init__(
        self,
        *,
        timeout_s: float = 4.0,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self._transport = transport
        self._clock = clock or time.monotonic

    def request(
        self,
        prompt: str,
        *,
        connectivity: ConnectivityState,
        payload: dict[str, Any] | None = None,
    ) -> StageBResult:
        correlation_id = str(uuid.uuid4())
        if connectivity is ConnectivityState.OFFLINE:
            return StageBResult(
                ok=False,
                correlation_id=correlation_id,
                message=STAGE_B_OFFLINE_MESSAGE,
                offline=True,
            )
        body = dict(payload or {})
        body["prompt"] = prompt
        body["correlation_id"] = correlation_id
        if self._transport is None:
            return StageBResult(
                ok=False,
                correlation_id=correlation_id,
                message=STAGE_B_TIMEOUT_MESSAGE,
                timed_out=True,
            )
        start = float(self._clock())
        try:
            result = self._transport(correlation_id, body)
        except Exception:  # noqa: BLE001
            return StageBResult(
                ok=False,
                correlation_id=correlation_id,
                message=STAGE_B_TIMEOUT_MESSAGE,
                timed_out=True,
            )
        elapsed = float(self._clock()) - start
        if elapsed > self.timeout_s:
            return StageBResult(
                ok=False,
                correlation_id=correlation_id,
                message=STAGE_B_TIMEOUT_MESSAGE,
                timed_out=True,
            )
        return StageBResult(ok=True, correlation_id=correlation_id, payload=result)
