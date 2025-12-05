"""
Eye/Face Micro-Model - Phase 1 Stub

This module provides eye tracking and fatigue detection:
- EyeModel: Blink probability, fixation patterns, pupil size
- EyeImagePreprocessor: Preprocesses eye/face images for model input

Status: Phase 1 (Sprint 1) - Stub implementation, not yet integrated
This is a placeholder for future eye tracking features.

Usage:
    from ml.models.eye_model import EyeModel, EyeImagePreprocessor
    
    # Preprocess image
    preprocessor = EyeImagePreprocessor()
    preprocessed = preprocessor(image)  # PIL Image -> [3, 64, 64] tensor
    
    # Run model
    model = EyeModel()
    outputs = model(preprocessed.unsqueeze(0))  # Add batch dimension
"""

from ml.models.eye_model.eye_model import EyeModel, EyeImagePreprocessor

__all__ = ['EyeModel', 'EyeImagePreprocessor']
