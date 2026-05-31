import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.loss_weighting import (  # noqa: E402
    TemporalWeightSchedule,
    build_temporal_weight_updates,
)
from ml.training.losses import MultiHeadLoss  # noqa: E402


def test_temporal_weight_schedule_warmup() -> None:
    sched = TemporalWeightSchedule(
        start_epoch=0, warmup_epochs=4, start_weight=0.1, target_weight=0.5
    )
    assert sched.at_epoch(0) == 0.1
    assert sched.at_epoch(2) > 0.1
    assert sched.at_epoch(10) == 0.5


def test_build_temporal_weight_updates() -> None:
    sched = TemporalWeightSchedule(
        start_epoch=0, warmup_epochs=2, start_weight=0.2, target_weight=1.0
    )
    updates = build_temporal_weight_updates(1, sched, {"motion": 0.6, "temporal_consistency": 0.4})
    assert set(updates.keys()) == {"motion", "temporal_consistency"}
    assert updates["motion"] > 0.0


def test_multihead_loss_weight_updates() -> None:
    loss = MultiHeadLoss(
        loss_functions={"objectness": nn.MSELoss(), "box": nn.L1Loss()},
        loss_weights={"objectness": 1.0, "box": 0.5},
    )
    before = loss.get_loss_weights()
    assert before["box"] == 0.5
    loss.set_loss_weights({"box": 0.9})
    after = loss.get_loss_weights()
    assert abs(after["box"] - 0.9) < 1e-6
