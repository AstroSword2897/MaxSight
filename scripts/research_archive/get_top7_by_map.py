#!/usr/bin/env python3
"""Get the top 7 conditions by mAP from inference_data.json."""

import argparse
import json
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

DEFAULT_K = 7
MAP_KEY = "mAP_50"  # MAP @ IoU 0.5; fallback to "mAP"


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
    # Collect (condition, mAP) for entries with valid metrics so we can rank by mAP.
    candidates = []
    for r in results:
        if isinstance(r, dict) and "error" not in r:
            cond = r.get("condition")
            m = r.get(map_key) is not None and r.get(map_key) or r.get("mAP", 0.0)
            if cond is not None:
                candidates.append((cond, float(m)))
    # Sort by mAP descending and take top k.
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:k]]


def main():
    p = argparse.ArgumentParser(description="Get top K conditions by mAP from inference_data.json.")
    p.add_argument(
        "--inference-data",
        type=Path,
        default=REPO / "inference_data.json",
        help="Path to inference_data.json from run_checkpoint_inference",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top conditions (default 7)")
    p.add_argument("--map-key", type=str, default=MAP_KEY, help="JSON key for mAP (default mAP_50)")
    p.add_argument(
        "--print", action="store_true", help="Print one condition per line (for shell consumption)"
    )
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
