#!/usr/bin/env python3
"""Find where images referenced in the val annotation JSON actually live on disk."""

import argparse
import json
import sys
from pathlib import Path

# Defaults for Colab.
DEFAULT_VAL_JSON = "/content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json"
DEFAULT_SEARCH_ROOT = "/content/drive/MyDrive"


def get_filenames_from_json(val_json: Path, max_filenames: int = 100) -> list[str]:
    """Collect unique image filenames from the annotation JSON."""
    with open(val_json) as f:
        data = json.load(f)
    filenames = []
    seen = set()
    if isinstance(data, list):
        for ann in data:
            if not ann.get("objects"):
                continue
            rel = ann.get("image_path", ann.get("file_name", ""))
            name = Path(rel).name
            if name and name not in seen:
                seen.add(name)
                filenames.append(name)
                if len(filenames) >= max_filenames:
                    break
    elif isinstance(data, dict) and "images" in data:
        for img in data["images"][:max_filenames]:
            name = img.get("file_name", "")
            if name and name not in seen:
                seen.add(name)
                filenames.append(name)
    return filenames


def find_file_in_tree(root: Path, filename: str, max_dirs: int = 10000) -> list[Path]:
    """BFS under root for filename; return list of full paths (usually 0 or 1)."""
    from collections import deque

    root = Path(root)
    if not root.exists():
        return []
    found = []
    queue = deque([root])
    dirs_done = 0
    while queue and dirs_done < max_dirs:
        d = queue.popleft()
        dirs_done += 1
        try:
            for e in d.iterdir():
                if e.is_file() and e.name == filename:
                    found.append(e.resolve())
                elif e.is_dir():
                    queue.append(e)
        except OSError:
            continue
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Find where annotation-referenced images live under a search root."
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path(DEFAULT_VAL_JSON),
        help="Path to val annotation JSON (MaxSight list or COCO)",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(DEFAULT_SEARCH_ROOT),
        help="Directory to search for image filenames (e.g. Drive root)",
    )
    parser.add_argument(
        "--max-filenames",
        type=int,
        default=30,
        help="Max number of filenames to look up (default 30)",
    )
    parser.add_argument(
        "--max-dirs",
        type=int,
        default=15000,
        help="Max dirs to scan per filename (default 15000)",
    )
    args = parser.parse_args()

    if not args.val_json.exists():
        print(f"Val JSON not found: {args.val_json}", file=sys.stderr)
        return 1
    if not args.search_root.exists():
        print(f"Search root not found: {args.search_root}", file=sys.stderr)
        return 1

    filenames = get_filenames_from_json(args.val_json, args.max_filenames)
    print(
        f"Looking up {len(filenames)} filenames from {args.val_json.name} under {args.search_root}\n"
    )

    found_count = 0
    first_found_path = None
    for i, name in enumerate(filenames):
        paths = find_file_in_tree(args.search_root, name, max_dirs=args.max_dirs)
        if paths:
            found_count += 1
            if first_found_path is None:
                first_found_path = paths[0]
            loc = str(paths[0])
            if len(paths) > 1:
                loc += f" (and {len(paths) - 1} more)"
            print(f"  [{i + 1}] {name}\n      -> {loc}")
        else:
            print(f"  [{i + 1}] {name}\n      -> NOT FOUND")

    print(f"\nSummary: {found_count}/{len(filenames)} found under {args.search_root}")
    if first_found_path is not None:
        parent = Path(first_found_path).parent
        print(f"\nSuggested IMAGE_DIR (parent of first found): {parent}")
        print("  Then in Colab: os.environ['IMAGE_DIR'] = str(that_path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
