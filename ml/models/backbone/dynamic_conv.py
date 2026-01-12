"""Dynamic Convolution Module for MaxSight 3.0

Per-sample adaptive kernels based on lighting, occlusion, and motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DynamicConv2d(nn.Module):
    """
    Dynamic convolution where each sample uses a weighted combination
    of multiple base kernels based on input conditions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_kernels: int = 4,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups

        # Base kernels
        self.base_kernels = nn.ParameterList([
            nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
            for _ in range(num_kernels)
        ])
        for k in self.base_kernels:
            nn.init.kaiming_normal_(k, mode='fan_out', nonlinearity='relu')

        # Condition MLP: maps [lighting, occlusion, motion] -> kernel weights
        condition_dim = max(in_channels // 4, 16)
        self.condition_mlp = nn.Sequential(
            nn.Linear(4, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, num_kernels),
            nn.Softmax(dim=-1)  # per-sample kernel weights
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, attention: Optional[torch.Tensor] = None, motion: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with per-sample dynamic kernel.

        Args:
            x: Input features [B, C, H, W]
            attention: Optional attention map [B, H, W] for occlusion scoring
            motion: Optional motion magnitude [B, 1]

        Returns:
            Output features [B, out_channels, H', W']
        """
        B, C, H, W = x.shape

        # Compute per-sample conditions
        lighting = self.compute_lighting_condition(x)  # [B, 2]
        occlusion = self.compute_occlusion_score(attention) if attention is not None else torch.zeros(B, 1, device=x.device)
        motion_cond = motion if motion is not None else torch.zeros(B, 1, device=x.device)

        condition_vec = torch.cat([lighting, occlusion, motion_cond], dim=1)  # [B, 4]
        kernel_weights = self.condition_mlp(condition_vec)  # [B, num_kernels]

        # Combine base kernels per sample
        # Result: dynamic_kernels [B, out_channels, in_channels/groups, K, K]
        dynamic_kernels = torch.stack(self.base_kernels, dim=0)  # [num_kernels, out_ch, in_ch, K, K]
        dynamic_kernels = torch.einsum('bk,kocwh->bocwh', kernel_weights, dynamic_kernels)

        outputs = []
        for i in range(B):
            out = F.conv2d(
                x[i:i+1],
                dynamic_kernels[i],
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups
            )
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    @property
    def output_size(self):
        return self.kernel_size  # note: H' = H if stride=1 and padding=kernel_size//2

    @staticmethod
    def compute_lighting_condition(x: torch.Tensor) -> torch.Tensor:
        """
        Compute brightness and contrast for each sample.
        Returns [B, 2]
        """
        brightness = x.mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
        contrast = x.std(dim=(1, 2, 3), keepdim=True)     # [B,1,1,1]
        return torch.cat([brightness, contrast], dim=1)   # [B,2]

    @staticmethod
    def compute_occlusion_score(attention: torch.Tensor) -> torch.Tensor:
        """
        Compute occlusion score from attention map [B,H,W]
        Returns [B,1]
        """
        return 1 - attention.mean(dim=(1, 2), keepdim=True)

    @staticmethod
    def compute_motion_condition(motion: torch.Tensor) -> torch.Tensor:
        """
        Return motion condition [B,1]
        """
        return motion
