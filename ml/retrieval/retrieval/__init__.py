"""Retrieval modules."""

from __future__ import annotations

from typing import Any

# Lazy exports so importing this package does not force faiss/torch at collection time.
__all__ = [
    "Stage1ANN",
    "Stage2Reranker",
    "ConceptRetrieval",
    "KnowledgeAugmentedRetrieval",
]

_EXPORT_MODULES: dict[str, tuple[str, str]] = {
    "Stage1ANN": (".stage1_ann", "Stage1ANN"),
    "Stage2Reranker": (".stage2_rerank", "Stage2Reranker"),
    "ConceptRetrieval": (".concept_retrieval", "ConceptRetrieval"),
    "KnowledgeAugmentedRetrieval": (".knowledge_augment", "KnowledgeAugmentedRetrieval"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        import importlib

        mod_name, attr = _EXPORT_MODULES[name]
        module = importlib.import_module(mod_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
