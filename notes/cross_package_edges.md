# Cross-package import edges (ml.* → ml.*)

Derived from `notes/raw_imports.txt`. Intra-package imports omitted.

| From | To | Example files |
|------|----|---------------|
| `ml.data` | `ml.models` | `ml/data/generate_annotations.py`, `ml/data/dataset.py` |
| `ml.data` | `ml.utils` | `ml/data/dataset.py`, `ml/data/gold/dataset.py` |
| `ml.evaluation` | `ml.runtime_constants` | `ml/evaluation/safety_gates.py` |
| `ml.infra` | `ml.middleware` | `ml/infra/inference_handler.py` |
| `ml.infra` | `ml.training` | `ml/infra/sagemaker_utils.py` |
| `ml.models` | `ml.retrieval` | `ml/models/retrieval_heads_production.py`, `ml/models/retrieval_heads_production.py` |
| `ml.models` | `ml.therapy` | `ml/models/maxsight_cnn.py` |
| `ml.models` | `ml.utils` | `ml/models/maxsight_cnn.py`, `ml/models/maxsight_cnn.py` |
| `ml.pipeline` | `ml.data` | `ml/pipeline/pipeline_runner.py`, `ml/pipeline/sagemaker_config.py` |
| `ml.pipeline` | `ml.training` | `ml/pipeline/pipeline_runner.py`, `ml/pipeline/sagemaker_config.py` |
| `ml.retrieval` | `ml.models` | `ml/retrieval/encoders/scene_graph_encoder.py`, `ml/retrieval/retrieval/knowledge_augment.py` |
| `ml.retrieval` | `ml.training` | `ml/retrieval/rag_reliability.py` |
| `ml.runtime` | `ml.runtime_constants` | `ml/runtime/contracts.py` |
| `ml.therapy` | `ml.runtime_constants` | `ml/therapy/therapy_safety.py` |
| `ml.training` | `ml.data` | `ml/training/run_config.py`, `ml/training/runner.py` |
| `ml.training` | `ml.infra` | `ml/training/sagemaker_entry.py` |
| `ml.training` | `ml.models` | `ml/training/runner.py` |
| `ml.training` | `ml.utils` | `ml/training/sagemaker_entry.py` |
| `ml.utils` | `ml.runtime_constants` | `ml/utils/output_scheduler.py` |

**Unique edges:** 19
