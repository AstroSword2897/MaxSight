#!/usr/bin/env python3
"""Find where trained checkpoints live. Prints first base dir that has checkpoints_<condition>/best_model.pt.
Use: CHECKPOINTS_BASE=$(python scripts/find_trained_checkpoints.py) && echo $CHECKPOINTS_BASE
"""
import os
import sys
from pathlib import Path
from typing import Optional

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

# Canonical locations (order matters: env first, then repo, then common Drive mounts)
def _candidates():
    base = os.environ.get("CHECKPOINTS_BASE")
    if base:
        yield Path(base)
    yield REPO / "checkpoints"
    yield REPO / "backups"
    # Google Drive for Desktop (macOS common paths)
    home = Path.home()
    yield home / "Google Drive" / "My Drive" / "MaxSight"
    for gd in home.glob("Library/CloudStorage/GoogleDrive-*"):
        yield gd / "My Drive" / "MaxSight"
    # Colab (when run on Colab)
    yield Path("/content/drive/MyDrive/MaxSight")


def find_base() -> Optional[Path]:
    for base in _candidates():
        if not base.exists():
            continue
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.startswith("checkpoints_") and (d / "best_model.pt").exists():
                    return base.resolve()
        except OSError:
            continue
    return None


def main():
    base = find_base()
    if base is None:
        print("No trained checkpoints found. Tried: CHECKPOINTS_BASE, repo/checkpoints, repo/backups, "
              "~/Google Drive/My Drive/MaxSight, ~/Library/CloudStorage/GoogleDrive-*/My Drive/MaxSight, "
              "/content/drive/MyDrive/MaxSight.", file=sys.stderr)
        return 1
    print(str(base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
