"""Connectivity state machine tests (MAXS-401)."""

from __future__ import annotations

from app.connectivity import ConnectivityMonitor, ConnectivityState


def test_debounce_prevents_flapping() -> None:
    clock = {"t": 0.0}
    reach = {"ok": True}

    mon = ConnectivityMonitor(
        debounce_s=1.0,
        clock=lambda: clock["t"],
        reachability=lambda: reach["ok"],
    )
    assert mon.poll() is ConnectivityState.ONLINE_FULL
    reach["ok"] = False
    assert mon.poll() is ConnectivityState.ONLINE_FULL  # pending
    clock["t"] = 0.5
    assert mon.poll() is ConnectivityState.ONLINE_FULL
    clock["t"] = 1.5
    assert mon.poll() is ConnectivityState.OFFLINE


def test_degraded_when_stage_b_unhealthy() -> None:
    mon = ConnectivityMonitor(
        debounce_s=0.0,
        reachability=lambda: True,
        stage_b_health=lambda: False,
    )
    assert mon.poll() is ConnectivityState.ONLINE_DEGRADED
