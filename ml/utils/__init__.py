# MaxSight Utils Module - Data preprocessing and utility functions
# Exports: preprocessing module (image transforms, impairment simulations, MFCC extraction, distance estimation)
# Critical for adapting data to vision conditions and preparing inputs for MaxSightCNN
# Complexity: O(H*W) for images, O(T) for audio - must be fast as preprocessing step
# Relationship: Converts raw sensor data (camera/audio) to model-ready format for accessibility adaptations
# Usage: from ml.utils.preprocessing import apply_glaucoma_transform, extract_mfcc

from ml.utils.preprocessing import ImagePreprocessor, DistanceEstimator, TextRegionDetector
from ml.utils.output_scheduler import OutputScheduler
from ml.utils.logging_config import setup_logging, get_logger

try:
    from ml.utils.ocr_integration import OCRIntegration, create_text_description
    __all__ = [
        'ImagePreprocessor',
        'DistanceEstimator',
        'TextRegionDetector',
        'OutputScheduler',
        'OCRIntegration',
        'create_text_description',
        'setup_logging',
        'get_logger'
    ]
except ImportError:
    __all__ = [
        'ImagePreprocessor',
        'DistanceEstimator',
        'TextRegionDetector',
        'OutputScheduler',
        'setup_logging',
        'get_logger'
    ]
