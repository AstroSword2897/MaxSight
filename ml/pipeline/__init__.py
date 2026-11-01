"""Production pipeline modules (SageMaker-ready). Public names load lazily (torch-heavy)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, str] = {
    "SageMakerPipelineConfig": "ml.pipeline.sagemaker_config",
    "RetrievalResult": "ml.pipeline.rag_advisory",
    "AdvisoryRetriever": "ml.pipeline.rag_advisory",
    "generate_therapy_advisory": "ml.pipeline.rag_advisory",
    "run_sagemaker_pipeline": "ml.pipeline.pipeline_runner",
}

if TYPE_CHECKING:  # pragma: no cover
    from ml.pipeline.pipeline_runner import run_sagemaker_pipeline
    from ml.pipeline.rag_advisory import (
        AdvisoryRetriever,
        RetrievalResult,
        generate_therapy_advisory,
    )
    from ml.pipeline.sagemaker_config import SageMakerPipelineConfig

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
