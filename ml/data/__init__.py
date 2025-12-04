# MaxSight Data Module - Dataset loading, downloading, and management for training
# Exports: download_datasets (COCO/AudioSet downloaders), Dataset classes (PyTorch implementations)
# Supports: COCO (80 classes, ~200K images), AudioSet (audio), synthetic data (testing)
# Complexity: O(N) dataset size, but lazy loading - only batches in memory (critical for large datasets)
# Relationship: Provides training data pipeline - required for training on real-world environmental scenes
# Usage: from ml.data.dataset import MaxSightDataset

from ml.data.dataset import MaxSightDataset
from ml.data.generate_annotations import generate_annotations_from_coco

# Production accessibility dataset (therapy-oriented)
try:
    from ml.data.create_accessibility_dataset import (
        AccessibilityDataset,
        SyntheticImpairmentEngine,
        generate_synthetic_dataset,
        create_label_template
    )
    __all__ = [
        'MaxSightDataset',
        'generate_annotations_from_coco',
        'AccessibilityDataset',
        'SyntheticImpairmentEngine',
        'generate_synthetic_dataset',
        'create_label_template',
        'combine_datasets'
    ]
except ImportError:
    # Fallback if dependencies not available
    __all__ = ['MaxSightDataset', 'generate_annotations_from_coco']

