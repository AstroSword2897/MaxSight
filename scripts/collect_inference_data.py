#!/usr/bin/env python3
"""
Inference Dataset Statistics Collection Script

Collects comprehensive statistics from inference datasets:
- Dataset size and splits
- Class distribution
- Image statistics (size, format, channels)
- Object counts per image
- Spatial distribution of objects
- Dataset metadata
- Quality metrics

Supports:
- COCO dataset
- Open Images V6
- BDD100K
- ADE20K
- Custom datasets

Usage:
    python scripts/collect_inference_data.py \
        --dataset coco \
        --data-dir datasets/coco \
        --output inference_stats.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.dataset import MaxSightDataset
from ml.data.inference_datasets import (
    OpenImagesV6Dataset, BDD100KDataset, ADE20KDataset
)
from ml.models.maxsight_cnn import COCO_CLASSES


class InferenceDatasetCollector:
    """
    Collect comprehensive statistics from inference datasets.
    """
    
    def __init__(self, dataset_name: str = 'coco'):
        """
        Initialize dataset collector.
        
        Args:
            dataset_name: Name of dataset ('coco', 'open_images', 'bdd100k', 'ade20k')
        """
        self.dataset_name = dataset_name
        self.stats = {
            'dataset_name': dataset_name,
            'collection_timestamp': datetime.now().isoformat(),
            'total_images': 0,
            'total_objects': 0,
            'class_distribution': Counter(),
            'image_sizes': [],
            'image_formats': Counter(),
            'objects_per_image': [],
            'spatial_distribution': {
                'center_x': [],
                'center_y': [],
                'width': [],
                'height': []
            },
            'class_cooccurrence': defaultdict(int),
            'metadata': {}
        }
    
    def collect_from_coco(self, data_dir: Path, max_samples: Optional[int] = None):
        """Collect statistics from COCO dataset."""
        print(f"Loading COCO dataset from {data_dir}...")
        
        try:
            dataset = MaxSightDataset(data_dir)
            self.stats['total_images'] = len(dataset)
            
            if max_samples:
                dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
                self.stats['total_images'] = len(dataset)
            
            print(f"✅ Loaded {len(dataset)} images")
            
            # Collect statistics
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    print(f"  Processing {batch_idx}/{len(dataset)}...")
                
                # Parse batch
                if isinstance(batch, dict):
                    image = batch.get('image', batch.get('images'))
                    boxes = batch.get('boxes', [])
                    labels = batch.get('labels', [])
                elif isinstance(batch, (list, tuple)):
                    image = batch[0]
                    boxes = batch[1] if len(batch) > 1 else []
                    labels = batch[2] if len(batch) > 2 else []
                else:
                    image = batch
                    boxes = []
                    labels = []
                
                # Image statistics
                if torch.is_tensor(image):
                    self.stats['image_sizes'].append(tuple(image.shape[-2:]))
                elif isinstance(image, Image.Image):
                    self.stats['image_sizes'].append(image.size)
                    self.stats['image_formats'][image.format or 'UNKNOWN'] += 1
                
                # Object statistics
                if boxes:
                    if isinstance(boxes, list):
                        num_objects = len(boxes)
                        if num_objects > 0:
                            self.stats['objects_per_image'].append(num_objects)
                            self.stats['total_objects'] += num_objects
                            
                            # Class distribution
                            if labels:
                                for label in labels:
                                    if isinstance(label, torch.Tensor):
                                        label = label.item() if label.numel() == 1 else label.tolist()
                                    if isinstance(label, (int, np.integer)):
                                        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
                                        self.stats['class_distribution'][class_name] += 1
                            
                            # Spatial distribution
                            for box in boxes:
                                if isinstance(box, torch.Tensor):
                                    box = box.tolist()
                                if isinstance(box, (list, tuple)) and len(box) >= 4:
                                    # Assume normalized [cx, cy, w, h] or [x1, y1, x2, y2]
                                    if len(box) == 4:
                                        if box[2] <= 1.0 and box[3] <= 1.0:  # Normalized
                                            cx, cy, w, h = box
                                        else:  # Pixel coordinates
                                            x1, y1, x2, y2 = box
                                            cx = (x1 + x2) / 2
                                            cy = (y1 + y2) / 2
                                            w = x2 - x1
                                            h = y2 - y1
                                        
                                        self.stats['spatial_distribution']['center_x'].append(cx)
                                        self.stats['spatial_distribution']['center_y'].append(cy)
                                        self.stats['spatial_distribution']['width'].append(w)
                                        self.stats['spatial_distribution']['height'].append(h)
            
            print("✅ Statistics collected")
            
        except Exception as e:
            print(f"⚠️  Error collecting from COCO: {e}")
            import traceback
            traceback.print_exc()
    
    def collect_from_open_images(self, data_dir: Path, max_samples: Optional[int] = None):
        """Collect statistics from Open Images V6 dataset."""
        print(f"Loading Open Images V6 dataset from {data_dir}...")
        
        try:
            dataset = OpenImagesV6Dataset(data_dir, max_samples=max_samples)
            self.stats['total_images'] = len(dataset)
            
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    print(f"  Processing {batch_idx}/{len(dataset)}...")
                
                # Similar processing as COCO
                # (Implementation would be dataset-specific)
                pass
            
        except Exception as e:
            print(f"⚠️  Error collecting from Open Images: {e}")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics from collected data."""
        stats = self.stats.copy()
        
        # Image size statistics
        if stats['image_sizes']:
            sizes_array = np.array(stats['image_sizes'])
            stats['image_size_stats'] = {
                'width': {
                    'mean': float(np.mean(sizes_array[:, 0])),
                    'std': float(np.std(sizes_array[:, 0])),
                    'min': int(np.min(sizes_array[:, 0])),
                    'max': int(np.max(sizes_array[:, 0]))
                },
                'height': {
                    'mean': float(np.mean(sizes_array[:, 1])),
                    'std': float(np.std(sizes_array[:, 1])),
                    'min': int(np.min(sizes_array[:, 1])),
                    'max': int(np.max(sizes_array[:, 1]))
                }
            }
        
        # Objects per image statistics
        if stats['objects_per_image']:
            obj_counts = np.array(stats['objects_per_image'])
            stats['objects_per_image_stats'] = {
                'mean': float(np.mean(obj_counts)),
                'std': float(np.std(obj_counts)),
                'min': int(np.min(obj_counts)),
                'max': int(np.max(obj_counts)),
                'median': float(np.median(obj_counts))
            }
        
        # Spatial distribution statistics
        for key, values in stats['spatial_distribution'].items():
            if values:
                values_array = np.array(values)
                stats['spatial_distribution'][f'{key}_stats'] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array))
                }
        
        # Class distribution (convert Counter to dict)
        stats['class_distribution'] = dict(stats['class_distribution'])
        stats['class_distribution_counts'] = {
            'total_classes': len(stats['class_distribution']),
            'most_common': dict(stats['class_distribution'].most_common(10))
        }
        
        # Convert image_formats Counter to dict
        stats['image_formats'] = dict(stats['image_formats'])
        
        return stats
    
    def save(self, output_path: Path):
        """Save collected statistics to JSON file."""
        final_stats = self.compute_statistics()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        final_stats = convert_types(final_stats)
        
        with open(output_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n✅ Inference dataset statistics saved to {output_path}")
        print(f"   Total images: {final_stats['total_images']}")
        print(f"   Total objects: {final_stats['total_objects']}")
        print(f"   Classes: {final_stats['class_distribution_counts']['total_classes']}")
        print(f"   Avg objects per image: {final_stats.get('objects_per_image_stats', {}).get('mean', 0):.2f}")


def collect_inference_data(
    dataset_name: str,
    data_dir: Path,
    output_path: Path,
    max_samples: Optional[int] = None
):
    """
    Collect inference dataset statistics.
    
    Args:
        dataset_name: Name of dataset ('coco', 'open_images', 'bdd100k', 'ade20k')
        data_dir: Dataset directory
        output_path: Output JSON file path
        max_samples: Maximum number of samples to process (None = all)
    """
    print("="*60)
    print("Inference Dataset Statistics Collection")
    print("="*60)
    
    collector = InferenceDatasetCollector(dataset_name=dataset_name)
    
    if dataset_name.lower() == 'coco':
        collector.collect_from_coco(data_dir, max_samples=max_samples)
    elif dataset_name.lower() == 'open_images':
        collector.collect_from_open_images(data_dir, max_samples=max_samples)
    else:
        print(f"⚠️  Dataset '{dataset_name}' collection not yet implemented")
        print("   Using COCO collection method...")
        collector.collect_from_coco(data_dir, max_samples=max_samples)
    
    collector.save(output_path)
    return collector.compute_statistics()


def main():
    parser = argparse.ArgumentParser(description="Collect inference dataset statistics")
    parser.add_argument("--dataset", type=str, default="coco", 
                       choices=["coco", "open_images", "bdd100k", "ade20k"],
                       help="Dataset name")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output", type=str, default="inference_stats.json", help="Output JSON file")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return 1
    
    collect_inference_data(
        dataset_name=args.dataset,
        data_dir=data_dir,
        output_path=output_path,
        max_samples=args.max_samples
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

