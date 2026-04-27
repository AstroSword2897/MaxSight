import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.rag_advisory import RetrievalResult  # noqa: E402
from ml.pipeline.sagemaker_config import SageMakerPipelineConfig  # noqa: E402
from ml.pipeline.pipeline_runner import run_sagemaker_pipeline  # noqa: E402
from ml.training.loss_weighting import TemporalWeightSchedule  # noqa: E402
from ml.data.video_preprocessing import PreprocessingConfig  # noqa: E402
from ml.data.video_panoptic import (  # noqa: E402
    AdaptiveTemporalConfig,
    PseudoPanopticQualityConfig,
    VideoSamplingConfig,
)


class DummyRetriever:
    def retrieve(self, query: Dict, top_k: int = 5) -> List[RetrievalResult]:
        return [RetrievalResult(payload={"kind": "therapy_hint"}, score=0.8)]


def test_run_sagemaker_pipeline_with_precomputed_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "model"
    input_dir.mkdir(parents=True)

    manifest = [
        {
            "video_id": "vid-1",
            "frame_paths": [f"f{i}" for i in range(8)],
            "frames_segments": [
                [{"bbox": [i, 0, 10, 10], "score": 0.9, "area": 100}]
                for i in range(8)
            ],
        }
    ]
    (input_dir / "video_records.json").write_text(json.dumps(manifest))

    cfg = SageMakerPipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        preprocessing=PreprocessingConfig(
            sampling=VideoSamplingConfig(temporal_window=4, temporal_stride=1, temporal_overlap=2),
            quality=PseudoPanopticQualityConfig(min_confidence=0.5, min_area_pixels=10),
            chunk_size=4,
            segmentation_workers=2,
            temporal_lookback=2,
            temporal_iou_threshold=0.2,
            enable_adaptive_windowing=True,
            adaptive=AdaptiveTemporalConfig(t_min=3, t_max=5, smooth_factor=0.0, overlap_ratio=0.0),
        ),
        temporal_weight_schedule=TemporalWeightSchedule(
            start_epoch=0, warmup_epochs=4, start_weight=0.1, target_weight=0.6
        ),
        advisory_retrieval_top_k=3,
    )

    res = run_sagemaker_pipeline(cfg=cfg, retriever=DummyRetriever())
    assert res["num_videos"] == 1
    out_path = Path(res["output_path"])
    assert out_path.exists()

    out_data = json.loads(out_path.read_text())
    assert len(out_data) == 1
    assert out_data[0]["clips"]
    first_clip = out_data[0]["clips"][0]
    assert "advisory" in first_clip
    assert "temporal_weight_updates" in first_clip

