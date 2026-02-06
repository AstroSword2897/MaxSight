#!/usr/bin/env python3
"""Loss Data Collection Script for MaxSight Training..."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, COCO_CLASSES
from ml.training.losses import (
    ObjectnessLoss, ClassificationLoss, BoxRegressionLoss,
    DistanceZoneLoss, UrgencyLoss, UncertaintyLoss, DepthLoss,
    MotionLoss, SceneDescriptionLoss, OCRLoss, FatigueLoss,
    MultiHeadLoss
)
from ml.training.task_balancing import GradNormMultiHeadLoss, PerHeadLossMonitor
from ml.data.dataset import MaxSightDataset
from ml.training.train_loop import ProductionTrainLoop


class LossDataCollector:
    """Collect comprehensive loss function data during training."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cpu',
        collect_gradients: bool = False,
        collect_task_weights: bool = True
    ):
        """Initialize loss data collector...."""
        self.model = model.to(device)
        self.device = device
        self.collect_gradients = collect_gradients
        self.collect_task_weights = collect_task_weights
        
        # Create loss functions if not provided
        if loss_fn is None:
            loss_functions = {
                'objectness': ObjectnessLoss(),
                'classification': ClassificationLoss(num_classes=len(COCO_CLASSES)),
                'box': BoxRegressionLoss(),
                'distance': DistanceZoneLoss(),
                'urgency': UrgencyLoss(),
                'uncertainty': UncertaintyLoss(),
                'depth': DepthLoss(),
                'motion': MotionLoss(),
                'scene_description': SceneDescriptionLoss(),
                'ocr': OCRLoss(),
                'fatigue': FatigueLoss()
            }
            self.loss_fn = MultiHeadLoss(loss_functions)
        else:
            self.loss_fn = loss_fn
        
        # Data storage
        self.loss_history: Dict[str, List[float]] = defaultdict(list)
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.task_weights: Dict[str, List[float]] = defaultdict(list)
        self.loss_statistics: Dict[str, Dict[str, float]] = {}
        self.iteration = 0
        
        # Loss monitor for trend detection
        self.loss_monitor = PerHeadLossMonitor(window_size=100)
    
    def collect_step(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_gradients: bool = False
    ) -> Dict[str, Any]:
        """Collect loss data for a single training step...."""
        self.iteration += 1
        
        # Compute losses
        if isinstance(self.loss_fn, MultiHeadLoss):
            loss_dict = self.loss_fn(outputs, targets)
        elif isinstance(self.loss_fn, GradNormMultiHeadLoss):
            total_loss, metrics = self.loss_fn(outputs, targets, model=self.model)
            loss_dict = {
                'total_loss': total_loss,
                **metrics.get('head_losses', {})
            }
            if self.collect_task_weights and 'task_weights' in metrics:
                for i, head_name in enumerate(self.loss_fn.head_names):
                    if i < len(metrics['task_weights']):
                        self.task_weights[head_name].append(metrics['task_weights'][i])
        else:
            total_loss = self.loss_fn(outputs, targets)
            loss_dict = {'total_loss': total_loss}
        
        # Extract per-head losses
        head_losses = {}
        for key, value in loss_dict.items():
            if key != 'total_loss' and torch.is_tensor(value):
                loss_val = value.item() if value.numel() == 1 else value.mean().item()
                head_losses[key] = loss_val
                self.loss_history[key].append(loss_val)
        
        total_loss_val = loss_dict.get('total_loss', torch.tensor(0.0))
        if torch.is_tensor(total_loss_val):
            total_loss_val = total_loss_val.item() if total_loss_val.numel() == 1 else total_loss_val.mean().item()
        self.loss_history['total_loss'].append(total_loss_val)
        
        # Collect gradient norms if requested
        gradient_data = {}
        if self.collect_gradients and compute_gradients:
            gradient_data = self._collect_gradient_norms()
        
        # Update loss monitor
        monitor_losses = {k: torch.tensor(v) for k, v in head_losses.items()}
        self.loss_monitor.update(monitor_losses)
        
        return {
            'iteration': self.iteration,
            'total_loss': total_loss_val,
            'head_losses': head_losses,
            'gradient_norms': gradient_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def _collect_gradient_norms(self) -> Dict[str, float]:
        """Collect gradient norms for each parameter group."""
        gradient_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                gradient_norms[name] = norm
                
                # Group by component (head, backbone, etc.)
                component = name.split('.')[0]
                if component not in self.gradient_norms:
                    self.gradient_norms[component] = []
                self.gradient_norms[component].append(norm)
        
        return gradient_norms
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for collected loss data.
        
        Returns:
            Dictionary mapping head names to statistics (mean, std, min, max, etc.)"""
        stats = {}
        
        for head_name, losses in self.loss_history.items():
            if len(losses) == 0:
                continue
            
            losses_array = np.array(losses)
            stats[head_name] = {
                'mean': float(np.mean(losses_array)),
                'std': float(np.std(losses_array)),
                'min': float(np.min(losses_array)),
                'max': float(np.max(losses_array)),
                'median': float(np.median(losses_array)),
                'q25': float(np.percentile(losses_array, 25)),
                'q75': float(np.percentile(losses_array, 75)),
                'count': len(losses),
                'trend': self._compute_trend(losses_array)
            }
        
        # Gradient norm statistics
        if self.gradient_norms:
            for component, norms in self.gradient_norms.items():
                if len(norms) > 0:
                    norms_array = np.array(norms)
                    stats[f'gradient_{component}'] = {
                        'mean': float(np.mean(norms_array)),
                        'std': float(np.std(norms_array)),
                        'min': float(np.min(norms_array)),
                        'max': float(np.max(norms_array)),
                        'count': len(norms)
                    }
        
        # Task weight statistics (GradNorm)
        if self.task_weights:
            for head_name, weights in self.task_weights.items():
                if len(weights) > 0:
                    weights_array = np.array(weights)
                    stats[f'task_weight_{head_name}'] = {
                        'mean': float(np.mean(weights_array)),
                        'std': float(np.std(weights_array)),
                        'min': float(np.min(weights_array)),
                        'max': float(np.max(weights_array)),
                        'final': float(weights[-1]),
                        'count': len(weights)
                    }
        
        self.loss_statistics = stats
        return stats
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend (increasing, decreasing, stable) from loss values."""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Linear regression to detect trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Normalize by mean to get relative change
        relative_slope = slope / (np.mean(values) + 1e-8)
        
        if relative_slope > 0.01:
            return 'increasing'
        elif relative_slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of collected data."""
        stats = self.compute_statistics()
        issues = self.loss_monitor.detect_issues()
        
        return {
            'metadata': {
                'total_iterations': self.iteration,
                'heads_tracked': list(self.loss_history.keys()),
                'collection_timestamp': datetime.now().isoformat()
            },
            'loss_statistics': stats,
            'detected_issues': issues,
            'loss_history': {k: v[-100:] for k, v in self.loss_history.items()},  # Last 100 iterations
            'task_weights': {k: v[-50:] for k, v in self.task_weights.items()} if self.task_weights else {}
        }
    
    def save(self, output_path: Path):
        """Save collected data to JSON file."""
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Loss data saved to {output_path}")
        print(f"   Iterations: {self.iteration}")
        print(f"   Heads tracked: {len(self.loss_history)}")
        print(f"   Statistics computed: {len(self.loss_statistics)}")


def collect_loss_data(
    checkpoint_path: Optional[Path] = None,
    data_dir: Path = Path('datasets/coco'),
    num_samples: int = 1000,
    batch_size: int = 4,
    device: str = 'cpu',
    output_path: Path = Path('loss_data.json'),
    collect_gradients: bool = False,
    collect_task_weights: bool = True
):
    """Collect loss data from model training/inference...."""
    print("="*60)
    print("Loss Data Collection")
    print("="*60)
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(num_classes=len(COCO_CLASSES))
    
    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded")
    
    model = model.to(device)
    model.eval()  # Set to eval for inference
    
    # Create loss collector
    print(f"\nInitializing loss collector...")
    collector = LossDataCollector(
        model=model,
        device=device,
        collect_gradients=collect_gradients,
        collect_task_weights=collect_task_weights
    )
    
    # Load dataset
    print(f"\nLoading dataset: {data_dir}")
    try:
        dataset = MaxSightDataset(data_dir)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"⚠️  Dataset loading failed: {e}")
        print("   Using synthetic data for collection...")
        # Create synthetic dataset
        from torch.utils.data import TensorDataset
        images = torch.randn(min(num_samples, 100), 3, 224, 224)
        dataset = TensorDataset(images)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Collect data
    print(f"\nCollecting loss data from {num_samples} samples...")
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            
            # Parse batch
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                targets = batch[1] if len(batch) > 1 else {}
            elif isinstance(batch, dict):
                images = batch.get('image', batch.get('images'))
                targets = {k: v for k, v in batch.items() if k not in ['image', 'images']}
            else:
                images = batch
                targets = {}
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Create synthetic targets if missing
            if not targets:
                B = images.shape[0]
                H, W = outputs.get('classifications', torch.zeros(B, 100, len(COCO_CLASSES))).shape[1:3]
                targets = {
                    'objectness': torch.randint(0, 2, (B, H*W)).float().to(device),
                    'labels': torch.randint(0, len(COCO_CLASSES), (B, H*W)).long().to(device),
                    'boxes': torch.rand(B, H*W, 4).to(device),
                    'distance': torch.randint(0, 3, (B, H*W)).long().to(device),
                    'urgency': torch.randint(0, 4, (B,)).long().to(device)
                }
            
            # Collect loss data
            collector.collect_step(outputs, targets, compute_gradients=False)
            
            samples_processed += images.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {samples_processed}/{num_samples} samples...")
    
    # Compute statistics and save
    print(f"\nComputing statistics...")
    summary = collector.get_summary()
    
    print(f"\nLoss Statistics Summary:")
    print("-" * 60)
    for head_name, stats in summary['loss_statistics'].items():
        if 'gradient' not in head_name and 'task_weight' not in head_name:
            print(f"{head_name:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, trend={stats.get('trend', 'N/A')}")
    
    # Save data
    collector.save(output_path)
    
    print(f"\n✅ Loss data collection complete!")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect loss function data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--data-dir", type=str, default="datasets/coco", help="Dataset directory")
    parser.add_argument("--output", type=str, default="loss_data.json", help="Output JSON file")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--collect-gradients", action="store_true", help="Collect gradient norms")
    parser.add_argument("--collect-task-weights", action="store_true", default=True, help="Collect task weights")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    collect_loss_data(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_path=output_path,
        collect_gradients=args.collect_gradients,
        collect_task_weights=args.collect_task_weights
    )


if __name__ == "__main__":
    main()

