"""Indexing modules for retrieval."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS = {
    "IndexManager": "ml.retrieval.indexing.index_manager",
    "NeuralIndexBuilder": "ml.retrieval.indexing.neural_index_builder",
}

if TYPE_CHECKING:
    from .index_manager import IndexManager as IndexManager
    from .neural_index_builder import NeuralIndexBuilder as NeuralIndexBuilder

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
