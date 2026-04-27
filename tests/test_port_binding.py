"""Tests for simulator port selection (no server start)."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.simulation.port_binding import port_is_available, resolve_listen_port  # noqa: E402


def test_resolve_returns_preferred_when_free() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()
    p, fb = resolve_listen_port("127.0.0.1", free_port, strict=False, scan_max=5)
    assert p == free_port
    assert fb is False


def test_resolve_scans_when_preferred_held() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    held = s.getsockname()[1]
    try:
        p, fb = resolve_listen_port("127.0.0.1", held, strict=False, scan_max=20)
        assert p != held
        assert fb is True
        assert port_is_available("127.0.0.1", p)
    finally:
        s.close()


def test_strict_raises_when_busy() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    held = s.getsockname()[1]
    try:
        with pytest.raises(OSError):
            resolve_listen_port("127.0.0.1", held, strict=True)
    finally:
        s.close()
