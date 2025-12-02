#!/usr/bin/env python3
"""
Production training script for MaxSight CNN.

Usage:
    python scripts/train_maxsight.py \
        --data-dir datasets/train \
        --epochs 100 \
        --batch-size 32 \
        --device cuda \
        --checkpoint-dir checkpoints
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from ml.models.maxsight_cnn import create_model
from ml.training.losses import MaxSightLoss
from ml.training.train_loop import ProductionTrainLoop
from ml.data.dataset import MaxSightDataset


def main():
    parser = argparse.ArgumentParser(description='Train MaxSight CNN')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Data directory')
    parser.add_argument('--annotation-file', type=str, default=None,
                       help='Annotation file (JSON)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--num-classes', type=int, default=48,
                       help='Number of classes')
    parser.add_argument('--use-audio', action='store_true',
                       help='Use audio features')
    parser.add_argument('--condition-mode', type=str, default=None,
                       choices=[None, 'glaucoma', 'amd', 'cataracts', 'color_blindness'],
                       help='Visual condition mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=args.num_classes,
        condition_mode=args.condition_mode,
        use_audio=args.use_audio
    )
    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create datasets
    print("Loading datasets...")
    data_dir = Path(args.data_dir)
    
    train_dataset = MaxSightDataset(
        data_dir=data_dir / 'train',
        annotation_file=args.annotation_file or (data_dir / 'train' / 'annotations.json'),
        condition_mode=args.condition_mode
    )
    
    val_dataset = MaxSightDataset(
        data_dir=data_dir / 'val',
        annotation_file=args.annotation_file or (data_dir / 'val' / 'annotations.json'),
        condition_mode=args.condition_mode
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create loss function
    loss_fn = MaxSightLoss(num_classes=args.num_classes)
    
    # Create trainer
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed
    )
    
    # Train
    results = trainer.train()
    
    print("\n" + "=" * 70)
    print("Training Results:")
    print(f"  Best model: {results['best_model_path']}")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

