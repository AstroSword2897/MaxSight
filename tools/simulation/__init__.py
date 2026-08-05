"""Simulation tools for MaxSight 3.0."""

from __future__ import annotations

from typing import Any

# Lazy so unrelated simulation tests do not pull faiss via retrieval.
__all__ = ["RetrievalIntegration"]


def __getattr__(name: str) -> Any:
    if name == "RetrievalIntegration":
        from .retrieval_integration import RetrievalIntegration

        return RetrievalIntegration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
