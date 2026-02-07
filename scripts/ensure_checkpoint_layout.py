#!/usr/bin/env python3
"""Create checkpoints_<condition>/ under a base dir so inference can find best_model.pt when added."""
import sys
from pathlib import Path

CONDITIONS = [
    "amblyopia", "amd", "astigmatism", "cataracts", "color_blindness",
    "cvi", "diabetic_retinopathy", "glaucoma", "hyperopia", "myopia",
    "presbyopia", "refractive_errors", "retinitis_pigmentosa", "strabismus",
]


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        d = base / f"checkpoints_{cond}"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()
    print(f"Layout ready: {base}")
    print("Add best_model.pt into each checkpoints_<condition>/ then run inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
