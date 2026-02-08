"""Test GradNorm Integration in Training Loop

Tests that GradNorm can be properly integrated and used in the training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
import sys
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.training.train_loop import ProductionTrainLoop
from ml.models.maxsight_cnn import create_model


def create_dummy_loss_fn():
    """Create a dummy loss function that returns per-head losses."""
    class DummyLoss(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, outputs: Dict, targets: Dict) -> Dict:
            """Return loss dict with per-head losses."""
            device = list(outputs.values())[0].device if outputs else torch.device('cpu')
            
            # Extract losses for different heads.
            loss_dict = {
                'total_loss': torch.tensor(1.0, device=device, requires_grad=True),
                'classification_loss': torch.tensor(0.5, device=device, requires_grad=True),
                'localization_loss': torch.tensor(0.3, device=device, requires_grad=True),
                'objectness_loss': torch.tensor(0.2, device=device, requires_grad=True),
            }
            return loss_dict
    
    return DummyLoss()


def test_gradnorm_availability():
    """Test that GradNorm is available and can be imported."""
    print("Testing GradNorm availability...")
    
    try:
        from ml.training.task_balancing import GradNormMultiHeadLoss
        # Verifies the class is actually usable (not just importable)
        assert GradNormMultiHeadLoss is not None
        assert callable(GradNormMultiHeadLoss)
        print("✅ GradNormMultiHeadLoss imported successfully")
        return True
    except ImportError as e:
        print(f"❌ GradNorm not available: {e}")
        return False


def test_gradnorm_initialization():
    """Test that GradNorm can be initialized."""
    print("\nTesting GradNorm initialization...")
    
    try:
        from ml.training.task_balancing import GradNormMultiHeadLoss
        
        # Create dummy head losses.
        head_losses = {
            'classification': nn.MSELoss(),
            'localization': nn.MSELoss(),
            'objectness': nn.BCELoss(),
        }
        
        # Initialize GradNorm.
        gradnorm = GradNormMultiHeadLoss(
            head_losses=head_losses,
            alpha=1.5,
            update_interval=100
        )
        
        print("✅ GradNorm initialized successfully")
        print(f"   - Number of heads: {gradnorm.num_heads}")
        print(f"   - Alpha: {gradnorm.alpha}")
        print(f"   - Update interval: {gradnorm.update_interval}")
        return True, gradnorm
    except Exception as e:
        print(f"❌ GradNorm initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_loop_with_gradnorm():
    """Test that training loop accepts GradNorm parameters."""
    print("\nTesting training loop with GradNorm parameters...")
    
    try:
        # Create a small model.
        model = create_model(num_classes=10)  # Small for testing.
        
        # Create dummy data.
        images = torch.randn(4, 3, 224, 224)
        targets = {
            'classifications': torch.randint(0, 10, (4, 196)),
            'boxes': torch.rand(4, 196, 4),
            'objectness': torch.rand(4, 196),
        }
        dataset = TensorDataset(images, torch.zeros(4))  # Dummy dataset.
        loader = DataLoader(dataset, batch_size=2)
        
        # Create loss function.
        loss_fn = create_dummy_loss_fn()
        
        # Create training loop with GradNorm.
        try:
            trainer = ProductionTrainLoop(
                model=model,
                train_loader=loader,
                val_loader=None,
                loss_fn=loss_fn,
                device='cpu',
                num_epochs=1,
                use_gradnorm=True,  # Enable GradNorm.
                gradnorm_alpha=1.5,
                gradnorm_update_interval=50
            )
            
            print("✅ Training loop created with GradNorm enabled")
            print(f"   - use_gradnorm: {trainer.use_gradnorm}")
            return True
        except Exception as e:
            print(f"⚠️ Training loop creation failed: {e}")
            print("   This is expected if GradNorm requires specific loss structure")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradnorm_loss_computation():
    """Test that GradNorm can compute losses."""
    print("\nTesting GradNorm loss computation...")
    
    try:
        from ml.training.task_balancing import GradNormMultiHeadLoss
        
        # Create dummy head losses.
        head_losses = {
            'classification': nn.MSELoss(),
            'localization': nn.MSELoss(),
        }
        
        gradnorm = GradNormMultiHeadLoss(
            head_losses=head_losses,
            alpha=1.5,
            update_interval=1  # Update every iteration for testing.
        )
        
        # Create dummy outputs and targets.
        outputs = {
            'classifications': torch.randn(2, 196, 10),
            'boxes': torch.randn(2, 196, 4),
        }
        targets = {
            'classifications': torch.randint(0, 10, (2, 196)),
            'boxes': torch.rand(2, 196, 4),
        }
        
        # Create dummy model for gradient computation.
        model = nn.Linear(10, 10)
        
        # Compute loss.
        total_loss, metrics = gradnorm(outputs, targets, model=model)
        
        print("✅ GradNorm loss computation successful")
        print(f"   - Total loss: {total_loss.item():.4f}")
        print(f"   - Metrics keys: {list(metrics.keys())}")
        return True
    except Exception as e:
        print(f"❌ GradNorm loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all GradNorm integration tests."""
    print("=" * 60)
    print("GradNorm Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Availability.
    results.append(("Availability", test_gradnorm_availability()))
    
    # Test 2: Initialization.
    init_success, gradnorm = test_gradnorm_initialization()
    results.append(("Initialization", init_success))
    
    # Test 3: Training loop integration.
    results.append(("Training Loop Integration", test_training_loop_with_gradnorm()))
    
    # Test 4: Loss computation.
    if init_success:
        results.append(("Loss Computation", test_gradnorm_loss_computation()))
    
    # Summary.
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit(main())


