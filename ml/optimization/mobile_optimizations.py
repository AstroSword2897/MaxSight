"""Mobile Efficiency Optimizations for Phase 7: Optimization & Mobile Deployment Includes: - Model pruning - Knowledge distillation for mobile - Dynamic head disabling - Memory-efficient inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class MobileOptimizer:
    """Mobile optimization utilities for Phase 7. Provides: - Model pruning - Head disabling for efficiency - Memory-efficient inference - Dynamic tier adjustment."""
    
    @staticmethod
    def prune_model(
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = 'magnitude'
    ) -> nn.Module:
        """Prune model weights for mobile deployment."""
        pruned_model = copy.deepcopy(model)
        pruned_model.eval()
        
        # Collect all weights.
        weights = []
        for module in pruned_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights.append(module.weight.data)
        
        # Calculate threshold.
        all_weights = torch.cat([w.flatten() for w in weights])
        threshold_idx = int(len(all_weights) * pruning_ratio)
        
        if method == 'magnitude':
            threshold = torch.sort(torch.abs(all_weights))[0][threshold_idx]
        else:
            threshold = torch.median(torch.abs(all_weights))
        
        # Prune weights.
        pruned_count = 0
        total_count = 0
        
        for module in pruned_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = torch.abs(module.weight.data) > threshold
                module.weight.data *= mask.float()
                pruned_count += (~mask).sum().item()
                total_count += mask.numel()
        
        print(f"Pruned {pruned_count}/{total_count} weights ({100*pruned_count/total_count:.1f}%)")
        
        return pruned_model
    
    @staticmethod
    def disable_heads(
        model: nn.Module,
        heads_to_disable: List[str]
    ) -> nn.Module:
        """Disable specific heads for efficiency."""
        for name, module in model.named_modules():
            if any(head_name in name for head_name in heads_to_disable):
                # Replace with identity.
                if hasattr(module, 'forward'):
                    module.forward = lambda x: torch.zeros_like(x) if torch.is_tensor(x) else {}
        
        return model
    
    @staticmethod
    def estimate_memory_usage(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Estimate memory usage for mobile deployment. Returns: Dictionary with memory estimates (MB)"""
        # Model parameters.
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        param_size_mb = param_size / (1024 ** 2)
        
        # Model buffers.
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        buffer_size_mb = buffer_size / (1024 ** 2)
        
        # Forward pass memory (approximate)
        dummy_input = torch.randn(*input_size)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                if isinstance(output, dict):
                    output_size = sum(
                        v.numel() * v.element_size() if torch.is_tensor(v) else 0
                        for v in output.values()
                    )
                else:
                    output_size = output.numel() * output.element_size()
            except:
                output_size = 0
        
        forward_memory_mb = (dummy_input.numel() * dummy_input.element_size() + output_size) / (1024 ** 2)
        
        return {
            'parameters_mb': param_size_mb,
            'buffers_mb': buffer_size_mb,
            'forward_pass_mb': forward_memory_mb,
            'total_mb': param_size_mb + buffer_size_mb + forward_memory_mb
        }


class EdgeCloudHybrid:
    """Edge-Cloud Hybrid Architecture for Phase 7."""
    
    def __init__(
        self,
        edge_model: nn.Module,
        cloud_endpoint: Optional[str] = None
    ):
        self.edge_model = edge_model
        self.cloud_endpoint = cloud_endpoint
        self.use_cloud = cloud_endpoint is not None
    
    def forward(
        self,
        images: torch.Tensor,
        use_cloud: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """Hybrid forward pass."""
        use_cloud = use_cloud if use_cloud is not None else self.use_cloud
        
        # Edge inference (always runs)
        edge_outputs = self.edge_model(images)
        
        if not use_cloud:
            return edge_outputs
        
        # Cloud inference (if enabled)
        # In production, send HTTP request to cloud endpoint.
        # For now, return edge outputs with cloud flag.
        cloud_outputs = {
            'cloud_processed': False,
            'cloud_latency_ms': 0.0
        }
        
        # Merge outputs.
        combined = {**edge_outputs, **cloud_outputs}
        
        return combined
    
    def should_use_cloud(
        self,
        urgency: float,
        battery_level: float,
        network_available: bool = True
    ) -> bool:
        """Determine if cloud processing should be used."""
        if not network_available:
            return False
        
        # Use cloud for high urgency and good battery.
        return urgency > 0.7 and battery_level > 0.3






