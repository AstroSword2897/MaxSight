"""
Production-grade training loop for MaxSight CNN - IMPROVED VERSION

This version fixes all identified issues:
- Safe mixed precision handling
- Fixed gradient accumulation edge cases
- EMA with bias correction
- Official PyTorch schedulers
- Safe backbone freezing
- Loss dict with .get() defaults
- Integrated DetectionMetrics for mAP
- Resume capability
- Batch validation
- Comprehensive logging (production-grade)
- Proper error handling and exception management

Author: Production-grade improvements based on detailed analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LinearLR,
    ConstantLR
)
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import json
import time
import logging
from copy import deepcopy
import numpy as np

try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    class DummyAutocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = DummyAutocast
    GradScaler = None
    AMP_AVAILABLE = False

from ml.training.metrics import DetectionMetrics

# Setup logging
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seed set to {seed}")


def move_targets_to_device(targets: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move all tensor targets to device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}


def parse_batch(batch: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Parse batch from dataloader with validation.
    Supports tuple (images, targets) or dict format.
    
        Arguments:
        batch: Batch from DataLoader (tuple or dict)
    
    Returns:
        Tuple of (images tensor, targets dict)
    
    Raises:
        ValueError: If batch format is invalid or images are malformed
    """
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else {}
    elif isinstance(batch, dict):
        images = batch.get('images')
        if images is None:
            images = batch.get('image')
        if images is None:
            raise ValueError("Batch must contain 'images' or 'image' key")
        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    # Validate images
    if not torch.is_tensor(images):
        raise ValueError(f"Images must be a tensor, got {type(images)}")
    if images.dim() != 4:
        raise ValueError(f"Images must be 4D [B, C, H, W], got shape {images.shape}")
    
    return images, targets


class EMA:
    """
    Exponential Moving Average with bias correction.
    
    Maintains shadow copies of model parameters with exponential moving average.
    Provides bias correction for early training steps.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, total_steps: int = 10000):
        """
        Initialize EMA.
        
        Arguments:
            model: Model to track
            decay: EMA decay factor
            total_steps: Total training steps for bias correction
        """
        self.decay = decay
        self.total_steps = total_steps
        self.global_step = 0
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with bias correction."""
        self.global_step += 1
        
        # Bias correction: adjust decay for early steps
        bias_correction = 1 - (self.decay ** self.global_step)
        effective_decay = self.decay / bias_correction if bias_correction > 0 else self.decay
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = effective_decay * self.shadow[name] + (1 - effective_decay) * param.data
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Apply shadow parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        # Clear backup after restore
        self.backup.clear()


class ProductionTrainLoop:
    """
    Production-grade training loop with all improvements.
    
    Features:
    - Safe mixed precision (proper fallback handling)
    - Fixed gradient accumulation (no double-update)
    - EMA with bias correction
    - Official PyTorch schedulers
    - Safe backbone freezing (isinstance checks)
    - Loss dict with .get() defaults
    - Integrated DetectionMetrics for mAP
    - Resume capability
    - Batch validation
    - Comprehensive logging (production-grade)
    - Proper error handling and exception management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        use_mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_interval: int = 50,
        checkpoint_dir: str = './checkpoints',
        save_best_only: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int = 0,
        freeze_bn_stats: bool = True,
        ema_decay: float = 0.9999,
        scheduler_type: str = 'cosine',  # 'cosine', 'onecycle', 'cosine_restarts'
        warmup_epochs: int = 5,
        num_classes: int = 80,  # For DetectionMetrics
        resume_from: Optional[str] = None,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        early_stopping_metric: str = 'val_loss'  # 'val_loss' or 'val_map'
    ):
        """
        Initialize production training loop.
        
        Arguments:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function (optional, uses default if None)
            device: Device to train on ('cuda', 'cpu', 'mps')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            use_mixed_precision: Use mixed precision training
            gradient_clip_norm: Gradient clipping norm
            gradient_accumulation_steps: Steps to accumulate gradients
            log_interval: Logging interval (batches)
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best model
            freeze_backbone: Freeze backbone parameters
            freeze_backbone_epochs: Epochs to freeze backbone
            freeze_bn_stats: Freeze BatchNorm stats when freezing backbone
            ema_decay: EMA decay factor
            scheduler_type: LR scheduler type
            warmup_epochs: Warmup epochs
            num_classes: Number of classes for metrics
            resume_from: Path to checkpoint to resume from
            seed: Random seed
            logger: Optional logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.freeze_bn_stats = freeze_bn_stats
        self.ema_decay = ema_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_counter = 0
        self.early_stopping_best_metric = float('inf') if early_stopping_metric == 'val_loss' else 0.0
        
        # Setup logger
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Set seed
        set_seed(seed)
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and (
            device == 'cuda' or str(device).startswith('cuda') or device == 'mps'
        )
        if self.use_mixed_precision and GradScaler is not None:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
            if use_mixed_precision:
                self.logger.warning("Mixed precision requested but not available, disabling")
        
        # Optimizer setup with discriminative learning rates
        self.backbone_params = []
        self.head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Identify backbone vs head parameters
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        if self.freeze_backbone and self.freeze_backbone_epochs > 0:
            # Freeze backbone initially
            self._freeze_backbone()
            param_groups = [
                {'params': self.head_params, 'lr': learning_rate}
            ]
        else:
            param_groups = [
                {'params': self.backbone_params, 'lr': learning_rate * 0.1},
                {'params': self.head_params, 'lr': learning_rate}
            ]
        
        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        
        # Scheduler - Use official PyTorch schedulers
        total_steps = len(train_loader) * num_epochs
        warmup_steps = warmup_epochs * len(train_loader) if warmup_epochs > 0 else 0
        
        # Ensure warmup doesn't exceed total steps
        if warmup_steps >= total_steps:
            warmup_epochs = 0
            warmup_steps = 0
            self.logger.warning(f"Warmup steps ({warmup_steps}) >= total steps, disabling warmup")
        
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'cosine_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(train_loader) * 10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type == 'cosine':
            if warmup_steps > 0:
                # Warmup + Cosine
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                cosine_steps = max(1, total_steps - warmup_steps)
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_steps,
                    eta_min=learning_rate * 0.01
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, total_steps),
                    eta_min=learning_rate * 0.01
                )
        else:
            # Default: constant LR
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        
        # EMA with bias correction
        self.ema = EMA(model, decay=ema_decay, total_steps=total_steps) if ema_decay > 0 else None
        
        # DetectionMetrics for validation
        self.detection_metrics = DetectionMetrics(
            num_classes=num_classes,
            iou_thresholds=[0.5, 0.75],
            device=torch.device(device)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_map = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'val_map_50': [],
            'val_map_75': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Resume from checkpoint if provided
        if resume_from:
            try:
                self._load_checkpoint(resume_from)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint {resume_from}: {e}")
                raise
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters safely using isinstance checks."""
        frozen_count = 0
        for name, module in self.model.named_modules():
            # Check if this is a backbone module
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_count += 1
                
                # Freeze BatchNorm stats if requested
                if self.freeze_bn_stats and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        
        if frozen_count > 0:
            self.logger.info(f"Backbone frozen ({frozen_count} parameters, BN stats frozen: {self.freeze_bn_stats})")
    
    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        unfrozen_count = 0
        for name, module in self.model.named_modules():
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
                
                # Unfreeze BatchNorm stats
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()
        
        if unfrozen_count > 0:
            self.logger.info(f"Backbone unfrozen ({unfrozen_count} parameters)")
    
