#!/usr/bin/env python3
"""
Fix Dataset Splits and Bounding Boxes

This script:
1. Merges existing train/val/test splits into a unified dataset
2. Fixes invalid bounding boxes (clips to boundaries, removes negatives)
3. Regenerates train/val/test splits with ZERO overlap
4. Generates class distribution report for weighted loss implementation

Usage:
    python scripts/fix_dataset_splits.py
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional

# --- CONFIG ---
INPUT_DIR = Path("datasets")
OUTPUT_DIR = Path("datasets/cleaned_splits")

# Sample counts (absolute numbers)
TRAIN_SAMPLES = 350000  # 350,000 training samples
VAL_SAMPLES = 70000     # 70,000 validation samples
# Test samples will be remaining after train + val

# Alternative: Use ratios (set to None to use absolute counts above)
TRAIN_RATIO = None  # Set to None to use absolute counts
VAL_RATIO = None    # Set to None to use absolute counts
TEST_RATIO = None   # Set to None to use absolute counts

RANDOM_SEED = 42

# Validate: either use ratios OR absolute counts, not both
if TRAIN_RATIO is not None:
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DATASET CLEANING AND RE-SPLITTING")
print("=" * 80)


def load_and_merge_splits() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load and merge all existing splits into unified dataset."""
    all_images = []
    all_annotations = []
    all_categories = []
    seen_image_ids = set()
    seen_category_ids = set()
    
    for split in ['train', 'val', 'test']:
        ann_file = INPUT_DIR / split / "annotations.json"
        if not ann_file.exists():
            print(f"⚠️  {split} split not found, skipping...")
            continue
        
        print(f"\nLoading {split} split...")
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'images' not in data:
            print(f"⚠️  {split} is not in COCO format, skipping...")
            continue
        
        # Merge images (avoid duplicates)
        for img in data.get('images', []):
            if img['id'] not in seen_image_ids:
                all_images.append(img)
                seen_image_ids.add(img['id'])
        
        # Merge annotations
        all_annotations.extend(data.get('annotations', []))
        
        # Merge categories (avoid duplicates)
        for cat in data.get('categories', []):
            if cat['id'] not in seen_category_ids:
                all_categories.append(cat)
                seen_category_ids.add(cat['id'])
        
        print(f"  Images: {len(data.get('images', []))}")
        print(f"  Annotations: {len(data.get('annotations', []))}")
    
    print(f"\n✅ Merged dataset:")
    print(f"  Total unique images: {len(all_images)}")
    print(f"  Total annotations: {len(all_annotations)}")
    print(f"  Total categories: {len(all_categories)}")
    
    return all_images, all_annotations, all_categories


def fix_bbox(ann: Dict, image_w: int, image_h: int) -> Optional[Dict]:
    """
    Fix bounding box: clip to image boundaries, remove invalid boxes.
    
    Returns:
        Fixed annotation dict, or None if box is invalid
    """
    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        return None  # Invalid format
    
    x, y, w, h = bbox
    
    # Remove negative coordinates
    if x < 0 or y < 0 or w < 0 or h < 0:
        # Try to fix by clipping
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
    
    # Clip to image bounds
    x = max(0, min(x, image_w))
    y = max(0, min(y, image_h))
    w = max(0, min(w, image_w - x))
    h = max(0, min(h, image_h - y))
    
    # Remove invalid boxes (zero or negative size)
    if w <= 0 or h <= 0:
        return None
    
    # Update annotation
    ann["bbox"] = [x, y, w, h]
    ann["area"] = w * h  # Update area
    
    return ann


def fix_all_bboxes(annotations: List[Dict], images: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Fix all bounding boxes and return statistics."""
    image_dict = {img["id"]: img for img in images}
    clean_annotations = []
    
    stats = {
        'original': len(annotations),
        'fixed': 0,
        'removed': 0,
        'out_of_bounds': 0,
        'negative_coords': 0,
        'invalid_size': 0
    }
    
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id not in image_dict:
            stats['removed'] += 1
            continue
        
        img = image_dict[img_id]
        original_bbox = ann.get("bbox", [])
        
        # Check for issues before fixing
        if len(original_bbox) == 4:
            x, y, w, h = original_bbox
            img_w, img_h = img.get('width', 224), img.get('height', 224)
            
            if x < 0 or y < 0 or w < 0 or h < 0:
                stats['negative_coords'] += 1
            if x + w > img_w or y + h > img_h:
                stats['out_of_bounds'] += 1
            if w <= 0 or h <= 0:
                stats['invalid_size'] += 1
        
        # Fix the box
        fixed = fix_bbox(ann.copy(), img.get('width', 224), img.get('height', 224))
        
        if fixed:
            clean_annotations.append(fixed)
            if fixed.get("bbox") != original_bbox:
                stats['fixed'] += 1
        else:
            stats['removed'] += 1
    
    return clean_annotations, stats


def split_dataset(images: List[Dict], annotations: List[Dict], 
                  train_ratio: Optional[float] = None, 
                  val_ratio: Optional[float] = None, 
                  test_ratio: Optional[float] = None,
                  train_samples: Optional[int] = None,
                  val_samples: Optional[int] = None,
                  seed: int = 42) -> Dict[str, Tuple[Set[int], List[Dict]]]:
    """
    Split dataset into train/val/test with zero overlap.
    
    Supports both ratio-based and absolute sample count splitting.
    
    Arguments:
        images: List of image dictionaries
        annotations: List of annotation dictionaries
        train_ratio: Fraction for training (if using ratios)
        val_ratio: Fraction for validation (if using ratios)
        test_ratio: Fraction for testing (if using ratios)
        train_samples: Absolute number of training samples (if using counts)
        val_samples: Absolute number of validation samples (if using counts)
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping split name to (image_id_set, annotation_list)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all image IDs
    image_ids = [img['id'] for img in images]
    image_ids = list(np.random.permutation(image_ids))  # Reproducible shuffle
    
    num_images = len(image_ids)
    
    # Calculate split sizes
    if train_samples is not None and val_samples is not None:
        # Use absolute counts
        num_train = min(train_samples, num_images)
        num_val = min(val_samples, num_images - num_train)
        num_test = num_images - num_train - num_val
        
        print(f"\nUsing absolute sample counts:")
        print(f"  Train: {num_train:,} samples")
        print(f"  Val:   {num_val:,} samples")
        print(f"  Test:  {num_test:,} samples (remaining)")
    elif train_ratio is not None and val_ratio is not None:
        # Use ratios
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val
        
        print(f"\nUsing ratio-based splitting:")
        print(f"  Train: {num_train:,} samples ({train_ratio:.1%})")
        print(f"  Val:   {num_val:,} samples ({val_ratio:.1%})")
        print(f"  Test:  {num_test:,} samples (remaining)")
    else:
        raise ValueError("Must provide either (train_ratio, val_ratio, test_ratio) or (train_samples, val_samples)")
    
    train_ids = set(image_ids[:num_train])
    val_ids = set(image_ids[num_train:num_train+num_val])
    test_ids = set(image_ids[num_train+num_val:])
    
    # Verify no overlap
    assert len(train_ids & val_ids) == 0, "Train/Val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train/Test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val/Test overlap detected!"
    
    print(f"\n✅ Split sizes:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Test: {len(test_ids)} images")
    print(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids)} images")
    
    # Split annotations
    def split_annotations(ann_list: List[Dict], image_set: Set[int]) -> List[Dict]:
        return [ann for ann in ann_list if ann.get("image_id") in image_set]
    
    splits = {
        "train": (train_ids, split_annotations(annotations, train_ids)),
        "val": (val_ids, split_annotations(annotations, val_ids)),
        "test": (test_ids, split_annotations(annotations, test_ids)),
    }
    
    return splits


def generate_class_report(annotations: List[Dict], categories: List[Dict], 
                         split_name: str) -> Dict:
    """Generate class distribution report for a split."""
    category_map = {cat['id']: cat['name'] for cat in categories}
    category_counter = Counter()
    
    for ann in annotations:
        cat_id = ann.get('category_id')
        if cat_id in category_map:
            category_counter[category_map[cat_id]] += 1
    
    all_categories = set(category_map.values())
    annotated_categories = set(category_counter.keys())
    zero_count = all_categories - annotated_categories
    rare_count = {cat: count for cat, count in category_counter.items() if count < 5}
    very_rare = {cat: count for cat, count in category_counter.items() if count < 2}
    
    report = {
        'split': split_name,
        'total_categories': len(all_categories),
        'annotated_categories': len(annotated_categories),
        'zero_annotation': len(zero_count),
        'rare_<5': len(rare_count),
        'very_rare_<2': len(very_rare),
        'zero_classes': list(zero_count),
        'rare_classes': dict(rare_count),
        'distribution': dict(category_counter)
    }
    
    return report


def save_splits(splits: Dict, images: List[Dict], categories: List[Dict], 
                output_dir: Path):
    """Save cleaned splits to JSON files."""
    for split_name, (img_set, ann_list) in splits.items():
        split_data = {
            "images": [img for img in images if img['id'] in img_set],
            "annotations": ann_list,
            "categories": categories
        }
        
        out_file = output_dir / f"{split_name}_annotations.json"
        with open(out_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"\n✅ {split_name.upper()} split saved:")
        print(f"  File: {out_file}")
        print(f"  Images: {len(split_data['images'])}")
        print(f"  Annotations: {len(ann_list)}")


def main():
    """Main execution."""
    # Step 1: Load and merge existing splits
    print("\n[1/4] Loading and merging existing splits...")
    all_images, all_annotations, all_categories = load_and_merge_splits()
    
    if not all_images:
        print("❌ No images found! Check your dataset structure.")
        return
    
    # Step 2: Fix bounding boxes
    print("\n[2/4] Fixing bounding boxes...")
    clean_annotations, bbox_stats = fix_all_bboxes(all_annotations, all_images)
    
    print(f"\n✅ Bounding box fixes:")
    print(f"  Original: {bbox_stats['original']}")
    print(f"  Fixed: {bbox_stats['fixed']}")
    print(f"  Removed: {bbox_stats['removed']}")
    print(f"  Issues found:")
    print(f"    - Out of bounds: {bbox_stats['out_of_bounds']}")
    print(f"    - Negative coords: {bbox_stats['negative_coords']}")
    print(f"    - Invalid size: {bbox_stats['invalid_size']}")
    
    # Step 3: Re-split dataset
    print("\n[3/4] Re-splitting dataset with zero overlap...")
    splits = split_dataset(
        all_images, clean_annotations,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
        seed=RANDOM_SEED
    )
    
    # Step 4: Generate class distribution reports
    print("\n[4/4] Generating class distribution reports...")
    reports = {}
    for split_name, (_, ann_list) in splits.items():
        reports[split_name] = generate_class_report(
            ann_list, all_categories, split_name
        )
    
    # Save splits
    print("\n💾 Saving cleaned splits...")
    save_splits(splits, all_images, all_categories, OUTPUT_DIR)
    
    # Save class distribution report
    report_file = OUTPUT_DIR / "class_distribution_report.json"
    with open(report_file, 'w') as f:
        json.dump(reports, f, indent=2)
    
    print(f"\n✅ Class distribution report saved: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Fixed {bbox_stats['fixed']} bounding boxes")
    print(f"✅ Removed {bbox_stats['removed']} invalid annotations")
    print(f"✅ Created 3 splits with ZERO overlap")
    
    for split_name, report in reports.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Categories with annotations: {report['annotated_categories']}/{report['total_categories']}")
        print(f"  Zero-annotation classes: {report['zero_annotation']}")
        print(f"  Rare classes (<5 samples): {report['rare_<5']}")
        print(f"  Very rare classes (<2 samples): {report['very_rare_<2']}")
    
    print("\n✅ Dataset is now ready for training!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("\n📊 Next steps:")
    print("   1. Review class_distribution_report.json for weighted loss implementation")
    print("   2. Use cleaned splits in your training script")
    print("   3. Implement class-weighted or Focal Loss for rare classes")


if __name__ == "__main__":
    main()

