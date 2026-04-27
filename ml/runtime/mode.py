"""Distinguish local simulator runs from production-style deployment via environment."""

from __future__ import annotations

import os
from enum import Enum


class RuntimeMode(str, Enum):
    """High-level deployment profile; drives defaults in tools/simulation/config."""

    SIMULATOR = "simulator"
    PRODUCTION = "production"


def get_runtime_mode() -> RuntimeMode:
    """Resolve ``MAXSIGHT_RUNTIME`` (default: simulator). Accepts production|prod."""

    raw = os.getenv("MAXSIGHT_RUNTIME", "simulator").strip().lower()
    if raw in ("production", "prod"):
        return RuntimeMode.PRODUCTION
    return RuntimeMode.SIMULATOR


def is_production_runtime() -> bool:
    """True when running with production defaults (no Flask debug, dev routes off)."""

    return get_runtime_mode() == RuntimeMode.PRODUCTION
