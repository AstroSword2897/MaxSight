# MaxSight Models Module - Neural network definitions for accessibility vision system
# Exports: MaxSightCNN (main detection model), create_model (factory function)
# Model features: Anchor-free detection, multi-task learning, condition adaptations, text detection, audio fusion
# Core of MaxSight system - all training/inference depends on these definitions
# Complexity: ~29M params, O(H*W*C) forward pass (H/W=image size, C=channels)
# Usage: from ml.models.maxsight_cnn import create_model

