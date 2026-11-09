"""COCO Dataset Splitter for MaxSight Creates train/test/validation splits from COCO dataset. Handles both COCO 2017 format and custom MaxSight annotations."""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_HASH_CHUNK_BYTES = 1 << 20


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_groups(items: list[Any], path_of: Callable[[Any], Path | None]) -> list[list[Any]]:
    """Group items whose image files hold identical bytes (first-appearance order).

    Different COCO ids can point at the same picture; splitting by id alone put such twins in train and
    test. An item whose file cannot be read cannot be cleared, so the split is refused.
    """
    resolved: list[tuple[Any, Path]] = []
    missing: list[str] = []
    for item in items:
        path = path_of(item)
        if path is None or not path.is_file():
            missing.append(str(path))
        else:
            resolved.append((item, path))
    if missing:
        raise ValueError(
            f"cannot check the content of {len(missing)} images (missing files or no image_dir), e.g. {missing[:3]}"
        )
    groups: dict[str, list[Any]] = {}
    for item, path in resolved:
        groups.setdefault(_file_sha256(path), []).append(item)
    duplicates = sum(len(g) - 1 for g in groups.values())
    if duplicates:
        print(f"  {duplicates} images duplicate another image's bytes; each set is kept in one split")
    return list(groups.values())


def _assign_groups(groups: list[list[Any]], train_end: int, val_end: int) -> tuple[list, list, list]:
    """Shuffle whole content groups (global numpy RNG, seeded by the caller) and fill train, val, test in order."""
    train: list[Any] = []
    val: list[Any] = []
    test: list[Any] = []
    filled = 0
    for index in np.random.permutation(len(groups)):
        group = groups[int(index)]
        target = train if filled < train_end else val if filled < val_end else test
        target.extend(group)
        filled += len(group)
    return train, val, test


def split_coco_dataset(
    coco_annotation_file: Path,
    output_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    min_objects_per_image: int = 1,
    *,
    image_dir: Path,
) -> tuple[Path, Path, Path]:
    """Split COCO dataset into train/val/test splits; byte-identical images always share a split."""
    # Validate splits.
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")

    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations.
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    with open(coco_annotation_file) as f:
        coco_data = json.load(f)

    # Validate COCO format.
    if "images" not in coco_data or "annotations" not in coco_data:
        raise ValueError("Invalid COCO format: missing 'images' or 'annotations'")

    # Build mappings.
    image_map = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id.
    image_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    # Filter images with minimum objects.
    valid_image_ids = [
        img_id
        for img_id in image_map
        if len(image_annotations.get(img_id, [])) >= min_objects_per_image
    ]

    print(f"Found {len(valid_image_ids)} valid images (with >= {min_objects_per_image} objects)")

    # Calculate split indices.
    total = len(valid_image_ids)
    if total == 0:
        raise ValueError("No valid images found after filtering.")
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    image_root = Path(image_dir)
    groups = _content_groups(valid_image_ids, lambda i: image_root / image_map[i]["file_name"])
    train_ids, val_ids, test_ids = _assign_groups(groups, train_end, val_end)

    print("Split sizes:")
    print(f"  Train: {len(train_ids)} ({len(train_ids) / total * 100:.1f}%)")
    print(f"  Val:   {len(val_ids)} ({len(val_ids) / total * 100:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({len(test_ids) / total * 100:.1f}%)")

    # Create split annotation files.
    def create_split_file(image_ids: list[int], split_name: str) -> Path:
        """Create annotation file for a split."""
        split_images = [image_map[img_id] for img_id in image_ids]
        split_annotations = []
        split_category_ids = set()

        for img_id in image_ids:
            for ann in image_annotations[img_id]:
                split_annotations.append(ann)
                split_category_ids.add(ann["category_id"])

        # Restrict categories to those present in the current split.
        split_categories = [
            cat for cat in coco_data.get("categories", []) if cat["id"] in split_category_ids
        ]

        split_data = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": split_categories,
        }
        output_file = output_dir / f"instances_{split_name}2017.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)

        print(
            f"Created {split_name} split: {output_file} ({len(split_images)} images, {len(split_annotations)} annotations)"
        )
        return output_file

    # Create all splits.
    train_file = create_split_file(train_ids, "train")
    val_file = create_split_file(val_ids, "val")
    test_file = create_split_file(test_ids, "test")

    return train_file, val_file, test_file


def create_maxsight_splits_from_coco(
    coco_annotation_file: Path,
    image_dir: Path | None,
    output_dir: Path,
    train_split: float | None = None,
    val_split: float | None = None,
    test_split: float | None = None,
    train_samples: int | None = None,
    val_samples: int | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    min_objects_per_image: int = 1,
) -> tuple[Path, Path, Path]:
    """Create MaxSight-format train/val/test splits from COCO dataset."""
    from ml.data.assistive_supervision import load_assistive_spec, object_distance_and_urgency
    from ml.data.generate_annotations import generate_scene_description, map_coco_to_environmental

    _spec = load_assistive_spec()

    # Validate: use either ratios OR absolute counts, not both.
    if train_samples is not None and val_samples is not None:
        # Using absolute counts - validate they're positive.
        if train_samples <= 0 or val_samples <= 0:
            raise ValueError(
                f"Sample counts must be positive: train={train_samples}, val={val_samples}"
            )
        use_absolute_counts = True
    elif train_split is not None and val_split is not None:
        # Using ratios - validate they sum to 1.0.
        test_split = test_split or (1.0 - train_split - val_split)
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
        use_absolute_counts = False
    else:
        # Default to ratios if nothing specified.
        train_split = train_split or 0.7
        val_split = val_split or 0.15
        test_split = test_split or 0.15
        use_absolute_counts = False

    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations.
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    with open(coco_annotation_file) as f:
        coco_data = json.load(f)

    image_map = {img["id"]: img for img in coco_data["images"]}
    category_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    image_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    # Filter images with minimum objects.
    valid_image_ids = [
        img_id
        for img_id in image_annotations
        if len(image_annotations.get(img_id, [])) >= min_objects_per_image
    ]

    print(f"Found {len(valid_image_ids)} valid images (with >= {min_objects_per_image} objects)")

    # Limit samples if specified.
    if num_samples and len(valid_image_ids) > num_samples:
        valid_image_ids = random.sample(valid_image_ids, num_samples)
        print(f"Limited to {num_samples} samples")

    image_ids = valid_image_ids
    print(f"Processing {len(image_ids)} images...")

    # Convert to MaxSight format.
    maxsight_annotations = []
    for image_id in image_ids:
        img_info = image_map.get(image_id)
        if img_info is None:
            continue

        anns = image_annotations[image_id]

        objects = []
        scene_urgency = 0

        for ann in anns:
            category_id = ann["category_id"]
            category_name = category_map.get(category_id, "unknown")
            env_category = map_coco_to_environmental(category_name)

            bbox = ann["bbox"]  # [x, y, w, h] in pixels.
            img_width = max(1, img_info.get("width", 224))
            img_height = max(1, img_info.get("height", 224))

            # Convert to center format and normalize.
            cx = (bbox[0] + bbox[2] / 2) / img_width
            cy = (bbox[1] + bbox[3] / 2) / img_height
            w = max(0.01, bbox[2] / img_width)
            h = max(0.01, bbox[3] / img_height)
            distance_zone, urgency = object_distance_and_urgency(cx, cy, w, h, env_category, _spec)
            scene_urgency = max(scene_urgency, urgency)

            objects.append(
                {
                    "box": [cx, cy, w, h],
                    "category": env_category,
                    "urgency": urgency,
                    "distance": distance_zone,
                    "confidence": ann.get("score", 1.0),
                }
            )

        # Skip images with zero objects.
        if not objects:
            continue

        # Generate scene description.
        scene_desc = generate_scene_description(objects)

        # Handle image path (check if image_dir is None)
        if image_dir is not None:
            image_path = str(image_dir / img_info.get("file_name", f"{image_id}.jpg"))
        else:
            image_path = img_info.get("file_name", f"{image_id}.jpg")

        # Create MaxSight annotation.
        maxsight_ann = {
            "image_id": image_id,
            "image_path": image_path,
            "objects": objects,
            "urgency": scene_urgency,
            "lighting": "normal",  # Will be detected during training.
            "description": scene_desc,
            "audio_path": None,  # No audio in COCO.
        }

        maxsight_annotations.append(maxsight_ann)

    # Calculate split indices.
    total = len(maxsight_annotations)
    groups = _content_groups(
        maxsight_annotations, lambda ann: Path(ann["image_path"]) if image_dir is not None else None
    )

    if use_absolute_counts:
        # Narrowed by the absolute-count branch above; assert for the type checker.
        assert train_samples is not None and val_samples is not None
        train_count = train_samples
        val_count = val_samples
        train_end = min(train_count, total)
        val_end = min(train_end + val_count, total)

        train_annotations, val_annotations, test_annotations = _assign_groups(groups, train_end, val_end)

        print("\nSplit sizes (absolute counts):")
        print(f"  Train: {len(train_annotations):,} samples (requested: {train_count:,})")
        print(f"  Val:   {len(val_annotations):,} samples (requested: {val_count:,})")
        print(f"  Test:  {len(test_annotations):,} samples (remaining)")
    else:
        # Defaults / validation above always leave concrete ratios here.
        train_ratio = 0.7 if train_split is None else train_split
        val_ratio = 0.15 if val_split is None else val_split
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_annotations, val_annotations, test_annotations = _assign_groups(groups, train_end, val_end)

        print("\nSplit sizes (ratios):")
        print(f"  Train: {len(train_annotations):,} ({len(train_annotations) / total * 100:.1f}%)")
        print(f"  Val:   {len(val_annotations):,} ({len(val_annotations) / total * 100:.1f}%)")
        print(f"  Test:  {len(test_annotations):,} ({len(test_annotations) / total * 100:.1f}%)")

    # Save split files.
    def save_split(annotations: list[dict], split_name: str) -> Path:
        output_file = output_dir / f"maxsight_{split_name}.json"
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved {split_name} split: {output_file}")
        return output_file

    train_file = save_split(train_annotations, "train")
    val_file = save_split(val_annotations, "val")
    test_file = save_split(test_annotations, "test")

    return train_file, val_file, test_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val/test")
    parser.add_argument(
        "--coco_annotations", type=Path, required=True, help="Path to COCO annotation JSON file"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing the COCO images (their bytes are hashed so duplicates share a split)",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory for split annotation files"
    )
    parser.add_argument(
        "--train_split", type=float, default=0.7, help="Fraction for training (default: 0.7)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.15, help="Fraction for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.15, help="Fraction for testing (default: 0.15)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Optional limit on total samples"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "maxsight"],
        default="maxsight",
        help="Output format: coco (original) or maxsight (converted)",
    )

    args = parser.parse_args()

    if args.format == "coco":
        train_file, val_file, test_file = split_coco_dataset(
            coco_annotation_file=args.coco_annotations,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            image_dir=args.image_dir,
        )
    else:
        train_file, val_file, test_file = create_maxsight_splits_from_coco(
            coco_annotation_file=args.coco_annotations,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            train_split=args.train_split if args.train_samples is None else None,
            val_split=args.val_split if args.val_samples is None else None,
            test_split=args.test_split if args.train_samples is None else None,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed,
            num_samples=args.num_samples,
            min_objects_per_image=1,  # Default: require at least 1 object per image.
        )

    print("\nOK Dataset splitting complete!")
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print(f"Test:  {test_file}")
