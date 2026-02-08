#!/usr/bin/env python3
"""Get the top 7 conditions by mAP from inference_data.json.

Reads the JSON produced by run_checkpoint_inference (or improve_map_all_models).
Ranks conditions by mAP@0.5 (mAP_50) and returns the top 7. Used by deploy/train
when you want to deploy the best-performing models by validation mAP.

Usage:
  python scripts/get_top7_by_map.py --inference-data inference_data.json
  python scripts/get_top7_by_map.py --inference-data inference_data.json --k 5
  # Print one per line for shell:
  python scripts/get_top7_by_map.py --inference-data inference_data.json --print
"""

import argparse
import json
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

DEFAULT_K = 7
MAP_KEY = "mAP_50"  # mAP @ IoU 0.5; fallback to "mAP"


def get_top_conditions_by_map(
    inference_data_path: Path,
    k: int = DEFAULT_K,
    map_key: str = MAP_KEY,
) -> list[str]:
    """Return list of condition names with highest mAP (descending)."""
    path = Path(inference_data_path)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    # Build (condition, mAP) for entries that have metrics (no error)
    candidates = []
    for r in results:
        if isinstance(r, dict) and "error" not in r:
            cond = r.get("condition")
            m = r.get(map_key) is not None and r.get(map_key) or r.get("mAP", 0.0)
            if cond is not None:
                candidates.append((cond, float(m)))
    # Sort by mAP descending, then take top k
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:k]]


def main():
    p = argparse.ArgumentParser(
        description="Get top K conditions by mAP from inference_data.json."
    )
    p.add_argument("--inference-data", type=Path, default=REPO / "inference_data.json",
                   help="Path to inference_data.json from run_checkpoint_inference")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top conditions (default 7)")
    p.add_argument("--map-key", type=str, default=MAP_KEY,
                   help="JSON key for mAP (default mAP_50)")
    p.add_argument("--print", action="store_true",
                   help="Print one condition per line (for shell consumption)")
    args = p.parse_args()

    top = get_top_conditions_by_map(args.inference_data, k=args.k, map_key=args.map_key)
    if not top:
        print("No valid results in inference data or file missing.", file=sys.stderr)
        return 1
    if args.print:
        for c in top:
            print(c)
    else:
        print(" ".join(top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
