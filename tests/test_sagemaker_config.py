import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.sagemaker_config import SageMakerPipelineConfig  # noqa: E402


def test_sagemaker_config_from_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SM_CHANNEL_TRAIN", "/tmp/train")
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", "/tmp/out")
    monkeypatch.setenv("SM_MODEL_DIR", "/tmp/model")

    monkeypatch.setattr(SageMakerPipelineConfig, "_read_hyperparameters", staticmethod(lambda: {}))

    cfg = SageMakerPipelineConfig.from_env()
    assert str(cfg.input_dir) == "/tmp/train"
    assert str(cfg.output_dir) == "/tmp/out"
    assert str(cfg.model_dir) == "/tmp/model"
    assert cfg.preprocessing.sampling.temporal_window >= 2

