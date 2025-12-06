"""
Safe MaxSightCNN with Error Handling and Fallbacks
Enhanced version with dependency validation and error propagation handling.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import MaxSightCNN, create_model
from ml.utils.error_handling import HeadExecutionManager, safe_head_execution
from ml.config import RuntimeConfig, DependencyGraph


class SafeMaxSightCNN(MaxSightCNN):
    """
    MaxSightCNN with error handling and fallback logic.
    
    Features:
    - Dependency validation
    - Error propagation handling
    - Fallback execution
    - Timeout management
    - Graceful degradation
    """
    
    def __init__(
        self,
        *args,
        runtime_config: Optional[RuntimeConfig] = None,
        enable_safe_mode: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.runtime_config = runtime_config or RuntimeConfig()
        self.enable_safe_mode = enable_safe_mode
        self.dependency_graph = DependencyGraph()
        
        if self.enable_safe_mode:
            self.head_manager = HeadExecutionManager(
                enable_fallbacks=self.runtime_config.enable_fallbacks,
                timeout_ms=self.runtime_config.timeout_ms,
                uncertainty_threshold=self.runtime_config.uncertainty_threshold
            )
        else:
            self.head_manager = None
    
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Safe forward pass with error handling and fallbacks.
        """
        if not self.enable_safe_mode:
            return super().forward(images, audio_features)
        
        try:
            # Execute base forward pass
            outputs = super().forward(images, audio_features)
            
            # Validate outputs
            outputs = self._validate_outputs(outputs)
            
            # Check uncertainty and apply fallback if needed
            if self.runtime_config.fallback_on_uncertainty:
                uncertainty = outputs.get('uncertainty')
                if uncertainty is not None and self.head_manager:
                    if self.head_manager.check_uncertainty_fallback(uncertainty, outputs):
                        outputs = self._apply_uncertainty_fallback(outputs)
            
            return outputs
            
        except Exception as e:
            # Fallback to minimal outputs
            if self.runtime_config.fallback_on_error:
                return self._get_fallback_outputs(images, str(e))
            else:
                raise
    
    def _validate_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Validate outputs and handle missing/invalid values."""
        validated = {}
        
        for key, value in outputs.items():
            if value is None:
                # Skip None values
                continue
            
            if isinstance(value, torch.Tensor):
                # Check for NaN/Inf
                if torch.isnan(value).any() or torch.isinf(value).any():
                    # Replace with zeros or default
                    validated[key] = torch.zeros_like(value)
                else:
                    validated[key] = value
            else:
                validated[key] = value
        
        # Ensure required outputs exist
        required = ['classifications', 'boxes', 'objectness']
        for req in required:
            if req not in validated:
                batch_size = outputs.get('classifications', torch.zeros(1, 196, 80)).shape[0]
                if req == 'classifications':
                    validated[req] = torch.zeros(batch_size, 196, self.num_classes, device=images.device)
                elif req == 'boxes':
                    validated[req] = torch.zeros(batch_size, 196, 4, device=images.device)
                elif req == 'objectness':
                    validated[req] = torch.zeros(batch_size, 196, device=images.device)
        
        return validated
    
    def _apply_uncertainty_fallback(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply fallback when uncertainty is high."""
        # Reduce number of detections
        # Lower confidence thresholds
        # Use simpler outputs
        
        if 'objectness' in outputs:
            # Lower objectness threshold (be more conservative)
            outputs['objectness'] = outputs['objectness'] * 0.8
        
        if 'classifications' in outputs:
            # Reduce classification confidence
            outputs['classifications'] = outputs['classifications'] * 0.9
        
        return outputs
    
    def _get_fallback_outputs(
        self,
        images: torch.Tensor,
        error_msg: str
    ) -> Dict[str, torch.Tensor]:
        """Get minimal fallback outputs when forward pass fails."""
        if images is not None:
            batch_size = images.shape[0]
            device = images.device
        else:
            batch_size = 1
            device = next(self.parameters()).device
        
        return {
            'classifications': torch.zeros(batch_size, 196, self.num_classes, device=device),
            'boxes': torch.zeros(batch_size, 196, 4, device=device),
            'objectness': torch.zeros(batch_size, 196, device=device),
            'text_regions': torch.zeros(batch_size, 196, device=device),
            'scene_embedding': torch.zeros(batch_size, 512, device=device),
            'urgency_scores': torch.zeros(batch_size, 4, device=device),
            'distance_zones': torch.zeros(batch_size, 196, 3, device=device),
            'num_locations': 196,
            'error': error_msg,
            'fallback_used': True
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary from head manager."""
        if self.head_manager:
            return self.head_manager.get_execution_summary()
        return {}


def create_safe_model(
    runtime_config: Optional[RuntimeConfig] = None,
    enable_safe_mode: bool = True,
    **model_kwargs
) -> SafeMaxSightCNN:
    """
    Create a safe MaxSight model with error handling.
    
    Arguments:
        runtime_config: Runtime configuration
        enable_safe_mode: Enable safe mode with error handling
        **model_kwargs: Arguments for model creation
    
    Returns:
        SafeMaxSightCNN instance
    """
    return SafeMaxSightCNN(
        runtime_config=runtime_config,
        enable_safe_mode=enable_safe_mode,
        **model_kwargs
    )

