"""SageMaker Processing Job (offline) — adaptive temporal preprocessing + advisory.

This module is executed as a SageMaker Processing Job entrypoint, not at inference
or training time. Importing it from another module is a bug — the guard below
enforces the boundary so Cursor-assisted refactors cannot accidentally wire it
into predict_fn or a training loop.

Inference path: ml.infra.inference_handler
Training path:  ml.training.sagemaker_entry
"""

from __future__ import annotations

# Enforce offline-only boundary at import time, not just by convention.
if __name__ != "__main__":
    raise ImportError(
        "ml.pipeline.sagemaker_entrypoint is an offline SageMaker Processing entrypoint "
        "and must not be imported at inference or training time. "
        "Inference: ml.infra.inference_handler | Training: ml.training.sagemaker_entry"
    )

import json

from ml.pipeline.pipeline_runner import run_sagemaker_pipeline  # noqa: E402 — import after guard

if __name__ == "__main__":
    result = run_sagemaker_pipeline()
    print(json.dumps(result))
