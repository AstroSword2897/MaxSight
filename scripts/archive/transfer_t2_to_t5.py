#!/usr/bin/env python3
"""Transfer T2_HYBRID_VIT → T5_TEMPORAL

Implements the tier transfer plan with:
- Selective weight copying
- Gradual unfreezing
- Parameter-grouped learning rates
- Loss unlock schedule"""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.maxsight_cnn import create_model, CapabilityTier, TierConfig
from ml.training.transfer_learning import TierTransferManager, create_transfer_optimizer
from ml.data.data_pipeline import create_data_loaders
from ml.training.train_loop import ProductionTrainLoop
from ml.training.losses import MultiHeadLoss, ObjectnessLoss, ClassificationLoss, BoxRegressionLoss, DistanceZoneLoss, UrgencyLoss
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.utils.logging_config import setup_logging

import logging

setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_loss_fn(config: dict, phase: int = 1):
    """Create loss function for current phase."""
    # Get weights for current phase (4 phases now)
    if phase == 1:
        weights = config['loss']['phase_1_weights']
    elif phase == 2:
        weights = config['loss']['phase_2_weights']
    elif phase == 3:
        weights = config['loss']['phase_3_weights']
    else:  # Phase == 4.
        weights = config['loss']['phase_4_weights']
    
    # Create loss functions for enabled tasks.
    loss_functions = {}
    
    if weights.get('detection', False):
        loss_functions['objectness'] = ObjectnessLoss()
    if weights.get('classification', False):
        loss_functions['classification'] = ClassificationLoss(num_classes=80)
    if weights.get('box_regression', False):
        loss_functions['box'] = BoxRegressionLoss()
    if weights.get('distance', False):
        loss_functions['distance'] = DistanceZoneLoss()
    if weights.get('urgency', False):
        loss_functions['urgency'] = UrgencyLoss()
    # Add more as needed...
    
    if config['loss']['use_gradnorm']:
        return GradNormMultiHeadLoss(loss_functions)
    else:
        return MultiHeadLoss(loss_functions)


def main():
    parser = argparse.ArgumentParser(description='Transfer T2 → T5')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to transfer config YAML'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate source checkpoint, do not transfer'
    )
    
    args = parser.parse_args()
    
    # Load config.
    config = load_config(args.config)
    
    logger.info("="*70)
    logger.info("T2 → T5 Tier Transfer")
    logger.info("="*70)
    
    # Create T5 model.
    logger.info("Creating T5 model...")
    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    t5_model = create_model(
        num_classes=80,
        tier_config=tier_config
    )
    
    # Initialize transfer manager.
    transfer_mgr = TierTransferManager(
        source_checkpoint=Path(config['source']['checkpoint']),
        target_model=t5_model,
        transfer_config=config['transfer']
    )
    
    # Validate source checkpoint.
    if not transfer_mgr.validate_source_checkpoint():
        logger.error("Source checkpoint validation failed!")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("OK Source checkpoint validated successfully")
        return
    
    # Transfer weights.
    logger.info("Transferring weights...")
    stats = transfer_mgr.transfer_weights(strict=config['transfer']['strict_transfer'])
    logger.info(f"Transfer stats: {stats}")
    
    # Create data loaders.
    logger.info("Creating data loaders...")
    train_loader, val_loader, _ = create_data_loaders(
        train_annotation_file=Path("datasets/cleaned_splits/train.json"),
        val_annotation_file=Path("datasets/cleaned_splits/val.json"),
        test_annotation_file=None,
        image_dir=Path("datasets/coco_raw"),
        batch_size=4,
        num_workers=8,
        pin_memory=True,
        condition_mode=None,
        apply_lighting_augmentation=False
    )
    
    # Create optimizer with parameter groups.
    logger.info("Creating optimizer with parameter groups...")
    optimizer = create_transfer_optimizer(
        model=t5_model,
        base_lr=config['training']['base_learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function (phase 1)
    loss_fn = create_loss_fn(config, phase=1)
    
    # Create training loop.
    train_loop = ProductionTrainLoop(
        model=t5_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=config['training']['mixed_precision'],
        gradient_clip_norm=config['training']['gradient_clip_norm']
    )
    
    logger.info("Starting transfer training...")
    logger.info("Phase 1 (epochs 0-10): Detection only")
    logger.info("Phase 2 (epochs 10-25): + Navigation")
    logger.info("Phase 3 (epochs 25-40): + Therapy/urgency")
    logger.info("Phase 4 (epochs 40+): All tasks enabled")
    
    # Training loop with freeze/unfreeze and loss unlock.
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(num_epochs):
        # Apply freeze schedule.
        transfer_mgr.apply_freeze_schedule(epoch)
        
        # Get loss unlock schedule.
        loss_unlock = transfer_mgr.get_loss_unlock_schedule(epoch)
        
        # Determine phase (4 phases now)
        if epoch < config['transfer']['phase_1_epochs']:
            phase = 1
        elif epoch < config['transfer']['phase_2_epochs']:
            phase = 2
        elif epoch < config['transfer']['phase_3_epochs']:
            phase = 3
        else:
            phase = 4
        
        # Update loss function if phase changed.
        phase_boundaries = [
            0,
            config['transfer']['phase_1_epochs'],
            config['transfer']['phase_2_epochs'],
            config['transfer']['phase_3_epochs']
        ]
        if epoch in phase_boundaries:
            loss_fn = create_loss_fn(config, phase=phase)
            train_loop.loss_fn = loss_fn
            logger.info(f"Switched to Phase {phase} loss configuration")
        
        # Train epoch.
        train_metrics = train_loop.train_epoch()
        
        # Validate.
        if epoch % int(config['validation']['val_check_interval'] * len(train_loader)) == 0:
            val_metrics = train_loop.validate()
            logger.info(f"Epoch {epoch}: train_loss={train_metrics.get('loss', 0):.4f}, "
                       f"val_loss={val_metrics.get('loss', 0):.4f}")
        
        # Save checkpoint.
        if epoch % config['checkpoint']['save_every_n_epochs'] == 0:
            checkpoint_path = Path(config['checkpoint']['save_dir']) / f"epoch_{epoch}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': t5_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics.get('loss', float('inf')),
                'transfer_stats': stats,
                'phase': phase
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("OK Transfer training complete!")


if __name__ == "__main__":
    main()


