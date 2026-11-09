"""Runtime safeguards for FAISS and PyTorch sharing OpenMP on macOS."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any


def configure_environment(platform: str = sys.platform) -> None:
    """Limit OpenMP before FAISS is imported on macOS."""
    if platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")


def configure_faiss(faiss_module: Any, platform: str = sys.platform) -> None:
    """Also enforce the limit through FAISS after its extension is loaded."""
    if platform == "darwin" and hasattr(faiss_module, "omp_set_num_threads"):
        faiss_module.omp_set_num_threads(1)


def load_faiss() -> Any:
    """Import and configure FAISS in the only safe order for macOS."""
    configure_environment()
    faiss_module = import_module("faiss")
    configure_faiss(faiss_module)
    return faiss_module
