"""Data pipeline for MaxSight training."""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from collections import defaultdict

from ml.data.dataset import MaxSightDataset
from ml.utils.preprocessing import ImagePreprocessor


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for MaxSight batches. Handles variable-length objects, optional audio, and optional frame sequences."""
    # Sequence mode uses 'frames' as primary image tensor; otherwise use single-frame 'images'.
    has_frames = any('frames' in item for item in batch)
    if has_frames:
        frame_lists = [item['frames'] for item in batch]
        batch_size = len(frame_lists)
        max_t = max(frames.shape[0] for frames in frame_lists)
        c, h, w = frame_lists[0].shape[1:]
        frames_batch = torch.zeros(batch_size, max_t, c, h, w, dtype=frame_lists[0].dtype)
        frame_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, frames in enumerate(frame_lists):
            t = frames.shape[0]
            frames_batch[i, :t] = frames
            frame_lengths[i] = t
        images = frames_batch
    else:
        images = torch.stack([item['images'] for item in batch])
        batch_size = len(batch)
        frame_lengths = None
    
    max_objects = max(item['num_objects'].item() for item in batch) if batch else 10
    max_objects = min(max_objects, batch[0].get('labels', torch.zeros(10)).shape[0])
    
    labels = torch.zeros(batch_size, max_objects, dtype=torch.long)
    boxes = torch.zeros(batch_size, max_objects, 4, dtype=torch.float32)
    distance = torch.zeros(batch_size, max_objects, dtype=torch.long)
    num_objects = torch.zeros(batch_size, dtype=torch.long)
    urgency = torch.zeros(batch_size, dtype=torch.long)
    
    has_audio = any('audio' in item for item in batch)
    audio_tensors: List[torch.Tensor] = []
    audio_lengths: List[int] = []
    
    for i, item in enumerate(batch):
        num_obj = item['num_objects'].item()
        num_objects[i] = num_obj
        
        if num_obj > 0:
            labels[i, :num_obj] = item['labels'][:num_obj]
            item_boxes = item['boxes'][:num_obj].clone()
            item_boxes[:, 2] = torch.clamp(item_boxes[:, 2], min=1e-4)
            item_boxes[:, 3] = torch.clamp(item_boxes[:, 3], min=1e-4)
            boxes[i, :num_obj] = item_boxes
            distance[i, :num_obj] = item['distance'][:num_obj]
        
        urgency[i] = item['urgency']
        
        if has_audio and 'audio' in item:
            audio = item['audio']
            audio_tensors.append(audio.squeeze(0))
            audio_lengths.append(audio.shape[-1])
        elif has_audio:
            audio_tensors.append(torch.zeros(13, 100))
            audio_lengths.append(100)
    
    result: Dict[str, Any] = {
        'images': images,
        'labels': labels,
        'boxes': boxes,
        'distance': distance,
        'num_objects': num_objects,
        'urgency': urgency,
    }
    
    if has_frames:
        result['frame_lengths'] = frame_lengths
    
    if has_audio and audio_tensors:
        max_audio_len = max(audio_lengths) if audio_lengths else 100
        padded_audio = torch.zeros(batch_size, 13, max_audio_len)
        for i, audio in enumerate(audio_tensors):
            padded_audio[i, :, :audio.shape[-1]] = audio
        result['audio'] = padded_audio
        result['audio_lengths'] = torch.tensor(audio_lengths, dtype=torch.long)
    
    if 'condition_mode' in batch[0]:
        result['condition_mode'] = batch[0]['condition_mode']

    if has_frames and all('temporal_consistency' in item for item in batch):
        result['temporal_consistency'] = torch.stack(
            [item['temporal_consistency'].reshape(-1)[0] for item in batch]
        )
    if has_frames and all('flicker' in item for item in batch):
        result['flicker'] = torch.stack(
            [item['flicker'].reshape(-1)[0] for item in batch]
        )
    if all('clip_id' in item for item in batch):
        result['clip_ids'] = [str(item['clip_id']) for item in batch]

    return result


def create_data_loaders(
    train_annotation_file: Path,
    val_annotation_file: Path,
    test_annotation_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    condition_mode: Optional[str] = None,
    apply_lighting_augmentation: bool = True,
    max_objects: int = 10,
    shuffle_train: bool = True,
    drop_last: bool = False,
    use_weighted_sampling: bool = False,
    class_weights: Optional[Dict[int, float]] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train/val/test data loaders for MaxSight training."""
    # Auto-detect image directory if not provided.
    if image_dir is None:
        # Checks common locations.
        possible_dirs = [
            train_annotation_file.parent.parent / 'train2017',
            train_annotation_file.parent.parent / 'val2017',
            train_annotation_file.parent.parent / 'images',
            train_annotation_file.parent.parent.parent / 'coco_raw' / 'train2017',
            train_annotation_file.parent.parent.parent / 'coco_raw' / 'val2017',
        ]
        for dir_path in possible_dirs:
            if dir_path.exists():
                image_dir = dir_path.parent  # Use parent to allow train2017/val2017 subdirs.
                break
        
        if image_dir is None:
            # Default to annotation file parent.
            image_dir = train_annotation_file.parent.parent
    
    # Create datasets.
    train_dataset = MaxSightDataset(
        data_dir=image_dir.parent if 'train2017' in str(image_dir) or 'val2017' in str(image_dir) else image_dir,
        annotation_file=train_annotation_file,
        image_dir=image_dir / 'train2017' if (image_dir / 'train2017').exists() else image_dir,
        audio_dir=audio_dir,
        condition_mode=condition_mode,
        apply_lighting_augmentation=apply_lighting_augmentation,
        max_objects=max_objects
    )
    
    val_dataset = MaxSightDataset(
        data_dir=image_dir.parent if 'train2017' in str(image_dir) or 'val2017' in str(image_dir) else image_dir,
        annotation_file=val_annotation_file,
        image_dir=image_dir / 'val2017' if (image_dir / 'val2017').exists() else image_dir,
        audio_dir=audio_dir,
        condition_mode=None,  # No augmentation for validation.
        apply_lighting_augmentation=False,
        max_objects=max_objects
    )
    
    test_dataset = None
    if test_annotation_file and test_annotation_file.exists():
        test_dataset = MaxSightDataset(
            data_dir=image_dir.parent if 'train2017' in str(image_dir) or 'val2017' in str(image_dir) else image_dir,
            annotation_file=test_annotation_file,
            image_dir=image_dir / 'val2017' if (image_dir / 'val2017').exists() else image_dir,
            audio_dir=audio_dir,
            condition_mode=None,
            apply_lighting_augmentation=False,
            max_objects=max_objects
        )
    
    # Create samplers if using weighted sampling.
    train_sampler = None
    if use_weighted_sampling and class_weights:
        # Compute sample weights based on class distribution.
        sample_weights = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            labels = sample['labels']
            num_obj = sample['num_objects'].item()
            
            # Weight by most frequent class in sample.
            if num_obj > 0:
                class_idx = labels[0].item()
                weight = class_weights.get(class_idx, 1.0)
            else:
                weight = 1.0
            
            sample_weights.append(weight)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling.
    
    # Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train and train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(annotation_file: Path) -> Dict[int, float]:
    """Compute class weights from annotations for handling class imbalance. Returns: Dictionary mapping class_idx -> weight (inverse frequency)"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Count class frequencies.
    class_counts = defaultdict(int)
    total_objects = 0
    
    if 'images' in data and 'annotations' in data:
        # COCO format.
        for ann in data['annotations']:
            category_id = ann.get('category_id', 0)
            class_counts[category_id] += 1
            total_objects += 1
    else:
        # Custom format.
        for ann in data:
            for obj in ann.get('objects', []):
                class_idx = obj.get('class', 0)
                class_counts[class_idx] += 1
                total_objects += 1
    
    # Compute inverse frequency weights.
    if total_objects == 0:
        return {}
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        # Inverse frequency: more frequent = lower weight.
        class_weights[class_idx] = total_objects / (len(class_counts) * count)
    
    return class_weights


def get_data_info(loader: DataLoader) -> Dict[str, Any]:
    """Get information about a data loader (dataset size, batch count, etc.). Returns: Dictionary with dataset statistics."""
    dataset = loader.dataset
    batch_size = loader.batch_size
    
    info = {
        'dataset_size': len(dataset),
        'batch_size': batch_size,
        'num_batches': len(loader),
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
    }
    
    # Sample a batch to get tensor shapes.
    try:
        sample_batch = next(iter(loader))
        info['batch_shapes'] = {
            key: list(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            for key, value in sample_batch.items()
        }
    except Exception:
        info['batch_shapes'] = 'Unable to sample batch'
    
    return info







