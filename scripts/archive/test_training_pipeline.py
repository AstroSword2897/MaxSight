#!/usr/bin/env python3
"""
Test training pipeline with small sample.

Tests data loaders, model creation, and a few training steps.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.data_pipeline import create_data_loaders, get_data_info
from ml.models.maxsight_cnn import create_model, CapabilityTier, TierConfig
from ml.training.losses import MultiHeadLoss, ObjectnessLoss, ClassificationLoss, BoxRegressionLoss, DistanceZoneLoss, UrgencyLoss
from ml.training.task_balancing import GradNormMultiHeadLoss


def load_config(config_path: Path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_training_pipeline(config_path: Path = None, num_test_batches: int = 3):
    """Test the training pipeline with a small sample."""
    print("="*70)
    print("Testing MaxSight Training Pipeline")
    print("="*70)
    
    # Load config if provided
    if config_path and config_path.exists():
        print(f"\nLoading config: {config_path}")
        config = load_config(config_path)
        tier_name = config['model']['tier']
        batch_size = config['data']['batch_size']
        print(f"  Tier: {tier_name}")
        print(f"  Batch size: {batch_size}")
    else:
        # Use defaults for T0
        print("\nUsing default T0 configuration")
        tier_name = "T0_BASELINE_CNN"
        batch_size = 4
        config = None
    
    # Check if splits exist
    train_file = Path("datasets/cleaned_splits/train.json")
    val_file = Path("datasets/cleaned_splits/val.json")
    
    if not train_file.exists() or not val_file.exists():
        print("\n❌ Training splits not found!")
        print("Please run: python scripts/setup_training_data.py")
        return False
    
    print("\n" + "="*70)
    print("Step 1: Creating Data Loaders")
    print("="*70)
    
    try:
        train_loader, val_loader, _ = create_data_loaders(
            train_annotation_file=train_file,
            val_annotation_file=val_file,
            test_annotation_file=None,
            image_dir=Path("datasets/coco_raw"),
            batch_size=batch_size,
            num_workers=0,  # Use 0 for testing
            pin_memory=False,
            condition_mode=None,
            apply_lighting_augmentation=False
        )
        
        train_info = get_data_info(train_loader)
        val_info = get_data_info(val_loader)
        
        print(f"✅ Train loader: {train_info['dataset_size']} samples, {train_info['num_batches']} batches")
        print(f"✅ Val loader:   {val_info['dataset_size']} samples, {val_info['num_batches']} batches")
        
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("Step 2: Creating Model")
    print("="*70)
    
    try:
        # Map tier name to CapabilityTier
        tier_map = {
            "T0_BASELINE_CNN": CapabilityTier.T0_BASELINE_CNN,
            "T1_ATTENTION": CapabilityTier.T1_ATTENTION,
            "T2_HYBRID_VIT": CapabilityTier.T2_HYBRID_VIT,
            "T3_CROSS_TASK": CapabilityTier.T3_CROSS_TASK,
            "T4_CROSS_MODAL": CapabilityTier.T4_CROSS_MODAL,
            "T5_TEMPORAL": CapabilityTier.T5_TEMPORAL,
        }
        
        tier = tier_map.get(tier_name, CapabilityTier.T0_BASELINE_CNN)
        tier_config = TierConfig.for_tier(tier)
        
        model = create_model(
            num_classes=80,
            tier_config=tier_config
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params/1e6:.2f}M parameters")
        print(f"   Tier: {tier.name}")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("Step 3: Testing Forward Pass")
    print("="*70)
    
    try:
        # Get a sample batch
        batch = next(iter(train_loader))
        images = batch['images']
        
        print(f"  Input shape: {images.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())[:5]}...")  # Show first 5 keys
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("Step 4: Testing Loss Computation")
    print("="*70)
    
    try:
        # Create loss function
        loss_functions = {
            'objectness': ObjectnessLoss(),
            'classification': ClassificationLoss(num_classes=80),
            'box': BoxRegressionLoss(),
            'distance': DistanceZoneLoss(),
            'urgency': UrgencyLoss(),
        }
        
        use_gradnorm = config and config.get('loss', {}).get('use_gradnorm', False)
        if use_gradnorm:
            loss_fn = GradNormMultiHeadLoss(loss_functions)
        else:
            loss_fn = MultiHeadLoss(loss_functions)
        
        # Compute loss
        model.train()
        outputs = model(images)
        
        targets = {
            'labels': batch['labels'],
            'boxes': batch['boxes'],
            'num_objects': batch['num_objects'],
            'distance': batch['distance'],
            'urgency': batch['urgency'],
        }
        
        loss_dict = loss_fn(outputs, targets)
        total_loss = loss_dict.get('loss', sum(loss_dict.values()))
        
        print(f"✅ Loss computation successful")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   Loss components: {list(loss_dict.keys())}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("Step 5: Testing Training Steps")
    print("="*70)
    
    try:
        # Create optimizer
        lr = config['training']['learning_rate'] if config else 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        model.train()
        total_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            if i >= num_test_batches:
                break
            
            images = batch['images']
            targets = {
                'labels': batch['labels'],
                'boxes': batch['boxes'],
                'num_objects': batch['num_objects'],
                'distance': batch['distance'],
                'urgency': batch['urgency'],
            }
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = loss_fn(outputs, targets)
            loss = loss_dict.get('loss', sum(loss_dict.values()))
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            print(f"  Batch {i+1}/{num_test_batches}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_test_batches
        print(f"✅ Training steps successful")
        print(f"   Average loss: {avg_loss:.4f}")
        
    except Exception as e:
        print(f"❌ Training steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ All Tests Passed!")
    print("="*70)
    print("\nThe training pipeline is ready for full training.")
    print("Next steps:")
    print("1. Ensure COCO train images are downloaded (train2017.zip)")
    print("2. Run: python scripts/setup_training_data.py")
    print("3. Run: python scripts/train_maxsight.py --data-dir <data_dir>")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to YAML config file (optional)'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=3,
        help='Number of batches to test (default: 3)'
    )
    
    args = parser.parse_args()
    
    success = test_training_pipeline(args.config, args.num_batches)
    sys.exit(0 if success else 1)

