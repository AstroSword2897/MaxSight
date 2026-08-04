"""App-layer connectivity state machine. Never imported by ml.runtime.stage_a."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum


class ConnectivityState(str, Enum):
    ONLINE_FULL = "ONLINE_FULL"
    ONLINE_DEGRADED = "ONLINE_DEGRADED"
    OFFLINE = "OFFLINE"


class ConnectivityMonitor:
    """Polled connectivity state with debounce. Stage A must not read this."""

    def __init__(
        self,
        *,
        debounce_s: float = 1.0,
        clock: Callable[[], float] | None = None,
        reachability: Callable[[], bool] | None = None,
        stage_b_health: Callable[[], bool] | None = None,
    ) -> None:
        self.debounce_s = debounce_s
        self._clock = clock or time.monotonic
        self._reachability = reachability or (lambda: True)
        self._stage_b_health = stage_b_health or (lambda: True)
        self._state = ConnectivityState.ONLINE_FULL
        self._pending: ConnectivityState | None = None
        self._pending_since = 0.0

    @property
    def state(self) -> ConnectivityState:
        return self._state

    def poll(self) -> ConnectivityState:
        desired = self._compute_desired()
        now = float(self._clock())
        if desired == self._state:
            self._pending = None
            return self._state
        if self._pending != desired:
            self._pending = desired
            self._pending_since = now
            if self.debounce_s <= 0:
                self._state = desired
                self._pending = None
            return self._state
        if now - self._pending_since >= self.debounce_s:
            self._state = desired
            self._pending = None
        return self._state

    def _compute_desired(self) -> ConnectivityState:
        if not bool(self._reachability()):
            return ConnectivityState.OFFLINE
        if not bool(self._stage_b_health()):
            return ConnectivityState.ONLINE_DEGRADED
        return ConnectivityState.ONLINE_FULL
