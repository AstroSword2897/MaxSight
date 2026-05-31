"""Choose a TCP port the Flask simulator can bind to."""

from __future__ import annotations

import socket


def port_is_available(host: str, port: int) -> bool:
    """Return True if nothing is listening on ``host:port`` (same bind shape Flask uses)."""

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bind_host = "" if host in ("0.0.0.0", "") else host
        s.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def resolve_listen_port(
    host: str,
    preferred: int,
    *,
    strict: bool = False,
    scan_max: int = 128,
) -> tuple[int, bool]:
    """Return ``(port, used_fallback)``.

    If ``preferred`` is free, use it. Otherwise scan upward up to ``scan_max`` attempts, then
    ask the OS for an ephemeral port. When ``strict`` is True, raise if ``preferred`` is taken.
    """

    if port_is_available(host, preferred):
        return preferred, False
    if strict:
        raise OSError(
            f"Port {preferred} is already in use. Choose another MAXSIGHT_PORT or unset MAXSIGHT_STRICT_PORT."
        )
    limit = min(preferred + scan_max, 65536)
    for p in range(preferred + 1, limit):
        if port_is_available(host, p):
            return p, True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_host = "" if host in ("0.0.0.0", "") else host
    s.bind((bind_host, 0))
    ephemeral = s.getsockname()[1]
    s.close()
    return ephemeral, True
