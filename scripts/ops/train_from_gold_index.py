#!/usr/bin/env python3
"""Train MaxSight using paths from datasets/medallion/gold/training_index.json (COCO section)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.medallion_layout import (  # noqa: E402
    default_gold_index_path,
    default_medallion_root,
    load_training_index,
    resolve_coco_for_train,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=REPO, help="Repository root (path resolution)"
    )
    parser.add_argument(
        "--gold-index",
        type=Path,
        default=None,
        help="training_index.json (default: <repo>/datasets/medallion/gold/training_index.json)",
    )
    args, rest = parser.parse_known_args()
    rr = args.repo_root.resolve()
    gold = args.gold_index
    if gold is None:
        gold = default_gold_index_path(default_medallion_root(rr))
    gold = gold.resolve()
    if not gold.exists():
        print(
            f"Gold index not found: {gold}. Run: python scripts/ops/medallion_build.py all",
            file=sys.stderr,
        )
        return 1

    idx = load_training_index(gold)
    data_dir, train_a, val_a, img_dir = resolve_coco_for_train(idx, rr)
    train_script = rr / "scripts" / "ops" / "train_maxsight.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data-dir",
        str(data_dir),
        "--train-annotation",
        str(train_a),
        "--val-annotation",
        str(val_a),
        "--image-dir",
        str(img_dir),
        *rest,
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=rr)


if __name__ == "__main__":
    sys.exit(main())
