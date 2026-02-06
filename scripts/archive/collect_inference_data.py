#!/usr/bin/env python3
"""Inference Dataset Statistics Collection Script..."""

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
    """Strict, COCO-correct inference dataset statistics collector...."""
    
    def __init__(self, dataset_name: str = 'coco'):
        """Initialize dataset collector.
        
        Args:
            dataset_name: Name of dataset (currently only 'coco' supported)"""
        self.dataset_name = dataset_name
        self.stats = {
            'dataset_name': dataset_name,
            'collection_timestamp': datetime.now().isoformat(),
            'total_images': 0,
            'total_objects': 0,
            'class_distribution': Counter(),
            'image_sizes': [],  # (W, H) tuples - consistent format
            'objects_per_image': [],
            'box_areas': [],  # normalized area fractions
            'box_centers': [],  # normalized (cx, cy) tuples
            'class_cooccurrence': Counter(),  # (class1, class2) pairs
            'metadata': {}
        }
    
    def process_sample(
        self, 
        image: torch.Tensor, 
        boxes: torch.Tensor, 
        labels: torch.Tensor
    ):
        """Process a single sample with strict COCO format handling...."""
        _, H, W = image.shape
        self.stats['image_sizes'].append((W, H))  # Consistent: (W, H)
        
        num_objs = boxes.shape[0]
        self.stats['objects_per_image'].append(num_objs)
        self.stats['total_objects'] += num_objs
        
        # Track classes present in this image for co-occurrence
        present_classes = set()
        
        # Process each object
        for i in range(num_objs):
            box = boxes[i]  # [4] tensor
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            
            # COCO format: [x, y, w, h] in pixels (not normalized)
            x, y, w, h = box.tolist()
            
            # Normalized center coordinates
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            self.stats['box_centers'].append((cx, cy))
            
            # Normalized area fraction
            area_fraction = (w * h) / (W * H)
            self.stats['box_areas'].append(area_fraction)
            
            # Class distribution
            if 0 <= label < len(COCO_CLASSES):
                cls_name = COCO_CLASSES[label]
                self.stats['class_distribution'][cls_name] += 1
                present_classes.add(cls_name)
        
        # Class co-occurrence: count pairs of classes in same image
        present_classes_list = list(present_classes)
        for i, c1 in enumerate(present_classes_list):
            for c2 in present_classes_list[i+1:]:
                # Count both (c1, c2) and (c2, c1) for symmetry
                self.stats['class_cooccurrence'][(c1, c2)] += 1
                self.stats['class_cooccurrence'][(c2, c1)] += 1
    
    def collect_from_coco(self, data_dir: Path, max_samples: Optional[int] = None):
        """Collect statistics from COCO dataset with strict format handling."""
        print(f"Loading COCO dataset from {data_dir}...")
        
        try:
            dataset = MaxSightDataset(data_dir)
            print(f"✅ Loaded {len(dataset)} images")
            
            # Collect statistics
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and batch_idx >= max_samples:
                    break
                
                if batch_idx % 100 == 0:
                    print(f"  Processing {batch_idx}/{min(max_samples or len(dataset), len(dataset))}...")
                
                # Handle different batch formats safely
                if isinstance(batch, dict):
                    image = batch.get('image')
                    boxes = batch.get('boxes')
                    labels = batch.get('labels')
                elif isinstance(batch, (list, tuple)):
                    image = batch[0]
                    boxes = batch[1] if len(batch) > 1 else None
                    labels = batch[2] if len(batch) > 2 else None
                else:
                    image = batch
                    boxes = None
                    labels = None
                
                # Ensure image is tensor and remove batch dimension
                if torch.is_tensor(image):
                    if image.dim() == 4:  # [B, C, H, W]
                        image = image.squeeze(0)  # [C, H, W]
                    elif image.dim() == 3:  # [C, H, W]
                        pass  # Already correct
                    else:
                        print(f"⚠️  Skipping sample {batch_idx}: unexpected image shape {image.shape}")
                        continue
                else:
                    print(f"⚠️  Skipping sample {batch_idx}: image is not a tensor")
                    continue
                
                # Handle boxes and labels - must be tensors
                if boxes is not None and torch.is_tensor(boxes):
                    if boxes.dim() == 2:  # [N, 4]
                        boxes = boxes.squeeze(0) if boxes.shape[0] == 1 else boxes
                    elif boxes.dim() == 3:  # [B, N, 4]
                        boxes = boxes.squeeze(0)  # [N, 4]
                    else:
                        print(f"⚠️  Skipping sample {batch_idx}: unexpected boxes shape {boxes.shape}")
                        continue
                else:
                    # No boxes - skip this sample
                    continue
                
                if labels is not None and torch.is_tensor(labels):
                    if labels.dim() == 1:  # [N]
                        labels = labels.squeeze(0) if labels.shape[0] == 1 else labels
                    elif labels.dim() == 2:  # [B, N]
                        labels = labels.squeeze(0)  # [N]
                    else:
                        print(f"⚠️  Skipping sample {batch_idx}: unexpected labels shape {labels.shape}")
                        continue
                else:
                    # No labels - skip this sample
                    continue
                
                # Ensure boxes and labels have matching length
                if boxes.shape[0] != labels.shape[0]:
                    print(f"⚠️  Skipping sample {batch_idx}: boxes/labels length mismatch")
                    continue
                
                # Process sample
                try:
                    self.process_sample(image, boxes, labels)
                    self.stats['total_images'] += 1
                except Exception as e:
                    print(f"⚠️  Error processing sample {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print("✅ Statistics collected")
            
        except Exception as e:
            print(f"⚠️  Error collecting from COCO: {e}")
            import traceback
            traceback.print_exc()
    
    def collect_from_open_images(self, data_dir: Path, max_samples: Optional[int] = None):
        """Collect statistics from Open Images V6 dataset - NOT IMPLEMENTED."""
        raise NotImplementedError("Open Images V6 collection not yet implemented. Use COCO dataset.")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics from collected data with clean schema."""
        if self.stats['total_images'] == 0:
            return {
                'dataset': self.dataset_name,
                'images': 0,
                'objects': 0,
                'error': 'No valid samples processed'
            }
        
        # Image size statistics (W, H) - consistent format
        sizes = np.array(self.stats['image_sizes'])
        
        # Box statistics
        areas = np.array(self.stats['box_areas']) if self.stats['box_areas'] else np.array([])
        centers = np.array(self.stats['box_centers']) if self.stats['box_centers'] else np.array([])
        
        # Objects per image
        obj_counts = np.array(self.stats['objects_per_image'])
        
        return {
            'dataset': self.dataset_name,
            'collection_timestamp': self.stats['collection_timestamp'],
            'images': self.stats['total_images'],
            'objects': self.stats['total_objects'],
            'objects_per_image': {
                'mean': float(np.mean(obj_counts)),
                'median': float(np.median(obj_counts)),
                'std': float(np.std(obj_counts)),
                'min': int(np.min(obj_counts)),
                'max': int(np.max(obj_counts))
            },
            'image_resolution': {
                'mean_width': float(sizes[:, 0].mean()),
                'mean_height': float(sizes[:, 1].mean()),
                'std_width': float(sizes[:, 0].std()),
                'std_height': float(sizes[:, 1].std()),
                'min_width': int(sizes[:, 0].min()),
                'max_width': int(sizes[:, 0].max()),
                'min_height': int(sizes[:, 1].min()),
                'max_height': int(sizes[:, 1].max())
            },
            'box_statistics': {
                'area_fraction': {
                    'mean': float(areas.mean()) if len(areas) > 0 else 0.0,
                    'std': float(areas.std()) if len(areas) > 0 else 0.0,
                    'median': float(np.median(areas)) if len(areas) > 0 else 0.0
                },
                'center_distribution': {
                    'mean_x': float(centers[:, 0].mean()) if len(centers) > 0 else 0.0,
                    'mean_y': float(centers[:, 1].mean()) if len(centers) > 0 else 0.0,
                    'std_x': float(centers[:, 0].std()) if len(centers) > 0 else 0.0,
                    'std_y': float(centers[:, 1].std()) if len(centers) > 0 else 0.0
                }
            },
            'class_distribution': dict(self.stats['class_distribution']),
            'class_distribution_summary': {
                'total_classes': len(self.stats['class_distribution']),
                'most_common': dict(self.stats['class_distribution'].most_common(20)),
                'least_common': dict(list(self.stats['class_distribution'].most_common())[-10:]) if len(self.stats['class_distribution']) >= 10 else {}
            },
            'class_cooccurrence': {
                'total_pairs': len(self.stats['class_cooccurrence']),
                'top_pairs': dict(self.stats['class_cooccurrence'].most_common(20))
            }
        }
    
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
    """Collect inference dataset statistics...."""
    print("="*60)
    print("COCO Inference Dataset Statistics Collection")
    print("="*60)
    
    if dataset_name.lower() != 'coco':
        print(f"⚠️  Dataset '{dataset_name}' not yet implemented. Only 'coco' is supported.")
        print("   Falling back to COCO collection...")
        dataset_name = 'coco'
    
    collector = InferenceDatasetCollector(dataset_name=dataset_name)
    collector.collect_from_coco(data_dir, max_samples=max_samples)
    collector.save(output_path)
    return collector.compute_statistics()



