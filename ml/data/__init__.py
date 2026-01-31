# MaxSight Data Module - Dataset loading, downloading, and management for training
# Exports: download_datasets (COCO/AudioSet downloaders), Dataset classes (PyTorch implementations)
# Supports: COCO (80 classes, ~200K images), AudioSet (audio), synthetic data (testing)
# Complexity: O(N) dataset size, but lazy loading - only batches in memory (critical for large datasets)
# Relationship: Provides training data pipeline - required for training on real-world environmental scenes
# Usage: from ml.data.dataset import MaxSightDataset

from ml.data.dataset import MaxSightDataset
from ml.data.generate_annotations import generate_annotations_from_coco
from ml.data.data_pipeline import (
    create_data_loaders,
    collate_fn,
    compute_class_weights,
    get_data_info
)

# Production accessibility dataset (therapy-oriented)
# Note: AccessibilityDataset and related functions are available via direct import:
# from ml.data.create_accessibility_dataset import AccessibilityDataset
# They are not re-exported here to avoid unused import warnings

__all__ = [
    'MaxSightDataset',
    'generate_annotations_from_coco',
    'create_data_loaders',
    'collate_fn',
    'compute_class_weights',
    'get_data_info',
]

