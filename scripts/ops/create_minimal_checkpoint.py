#!/usr/bin/env python3
"""Create minimal checkpoints so run_checkpoint_inference can be exercised for all conditions."""
import sys
from pathlib import Path

try:
    _repo = Path(__file__).resolve().parents[1]
except NameError:
    _repo = Path.cwd()
sys.path.insert(0, str(_repo))

import torch
from ml.models.maxsight_cnn import (
    COCO_CLASSES,
    CapabilityTier,
    TierConfig,
    create_model,
)

CONDITIONS = [
    "amblyopia", "amd", "astigmatism", "cataracts", "color_blindness",
    "cvi", "diabetic_retinopathy", "glaucoma", "hyperopia", "myopia",
    "presbyopia", "refractive_errors", "retinitis_pigmentosa", "strabismus",
]


def main():
    for condition in CONDITIONS:
        out_dir = _repo / "checkpoints" / f"checkpoints_{condition}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_model.pt"

        model = create_model(
            num_classes=len(COCO_CLASSES),
            use_audio=False,
            condition_mode=condition,
            tier_config=TierConfig.for_tier(CapabilityTier["T5_TEMPORAL"]),
        )
        state = model.state_dict()
        torch.save({"model_state_dict": state}, out_path)
        print(f"Saved {out_path}")
    print(f"Created {len(CONDITIONS)} minimal checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


