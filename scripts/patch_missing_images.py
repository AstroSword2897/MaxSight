#!/usr/bin/env python3
"""Patch missing COCO images during training...."""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COCOImagePatcher:
    """Downloads missing COCO images to complete the dataset."""
    
    # COCO image URLs.
    TRAIN_URL = "http://images.cocodataset.org/train2017/"
    VAL_URL = "http://images.cocodataset.org/val2017/"
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.datasets_dir = root_dir / "datasets"
        self.coco_raw_dir = self.datasets_dir / "coco_raw"
        self.splits_dir = self.datasets_dir / "cleaned_splits"
        
    def find_missing_images(self, split: str) -> Tuple[List[str], List[Path]]:
        """Find missing images for a given split.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            (list of image filenames, list of expected paths)"""
        # Load annotation file.
        ann_file = self.splits_dir / f"maxsight_{split}.json"
        if not ann_file.exists():
            logger.error(f"Annotation file not found: {ann_file}")
            return [], []
        
        with open(ann_file) as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {split} split")
        
        # Check which images are missing.
        missing_files = []
        missing_paths = []
        
        for sample in data:
            img_path = Path(sample['image_path'])
            
            if not img_path.exists():
                missing_files.append(img_path.name)
                missing_paths.append(img_path)
        
        logger.info(f"Found {len(missing_files)} missing images in {split} split")
        return missing_files, missing_paths
    
    def download_image(self, filename: str, split: str, retries: int = 3) -> bool:
        """Download a single image from COCO servers...."""
        # Determine URL and target directory.
        if split == 'train':
            url = self.TRAIN_URL + filename
            target_dir = self.coco_raw_dir / "train2017"
        else:
            url = self.VAL_URL + filename
            target_dir = self.coco_raw_dir / "val2017"
        
        target_path = target_dir / filename
        
        # Skip if already exists.
        if target_path.exists():
            return True
        
        # Create directory if needed.
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download with retries.
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(url, target_path)
                return True
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries} for {filename}: {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to download {filename} after {retries} attempts: {e}")
                    return False
        
        return False
    
    def patch_split(self, split: str, max_workers: int = 4) -> Tuple[int, int]:
        """Download all missing images for a split...."""
        logger.info(f"Starting patch for {split} split...")
        
        # Find missing images.
        missing_files, _ = self.find_missing_images(split)
        
        if not missing_files:
            logger.info(f"No missing images in {split} split!")
            return 0, 0
        
        logger.info(f"Downloading {len(missing_files)} images with {max_workers} workers...")
        
        # Download in parallel.
        success_count = 0
        fail_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks.
            futures = {
                executor.submit(self.download_image, filename, split): filename
                for filename in missing_files
            }
            
            # Process completions.
            for i, future in enumerate(as_completed(futures), 1):
                filename = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Exception downloading {filename}: {e}")
                    fail_count += 1
                
                # Progress update every 50 images.
                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(missing_files)} "
                              f"(ok {success_count}, fail {fail_count})")
        
        logger.info(f"Completed {split} split: {success_count} success, {fail_count} failed")
        return success_count, fail_count
    
    def verify_completion(self, split: str) -> None:
        """Verify that all images are now available."""
        missing_files, _ = self.find_missing_images(split)
        
        if not missing_files:
            logger.info(f"ok All {split} images are now available!")
        else:
            logger.warning(f"WARNING Still missing {len(missing_files)} {split} images")


def main():
    parser = argparse.ArgumentParser(
        description="Download missing COCO images to complete dataset"
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'all'],
        default='all',
        help='Which split to patch (default: all)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel download threads (default: 4)'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/Users/nani/2026-Prototype'),
        help='Root directory of the project'
    )
    
    args = parser.parse_args()
    
    # Initialize patcher.
    patcher = COCOImagePatcher(args.root)
    
    logger.info("=" * 60)
    logger.info("COCO Image Patcher")
    logger.info("=" * 60)
    
    # Patch requested splits.
    if args.split == 'all':
        splits = ['train', 'val']
    else:
        splits = [args.split]
    
    total_success = 0
    total_fail = 0
    
    for split in splits:
        success, fail = patcher.patch_split(split, max_workers=args.workers)
        total_success += success
        total_fail += fail
        
        # Verify.
        patcher.verify_completion(split)
        logger.info("")
    
    # Final summary.
    logger.info("=" * 60)
    logger.info(f"SUMMARY: Downloaded {total_success} images, {total_fail} failed")
    logger.info("=" * 60)
    
    if total_fail > 0:
        logger.warning(f"WARNING {total_fail} images could not be downloaded")
        logger.warning("These may be removed from COCO servers or have connectivity issues")
    
    if total_success > 0:
        logger.info("ok New images will be used automatically in subsequent training batches")


if __name__ == '__main__':
    main()


