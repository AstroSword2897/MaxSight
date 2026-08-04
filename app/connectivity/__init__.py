"""App-layer connectivity monitor (isolated from Stage A)."""

from app.connectivity.monitor import ConnectivityMonitor, ConnectivityState

__all__ = ["ConnectivityMonitor", "ConnectivityState"]
