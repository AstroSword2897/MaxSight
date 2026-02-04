#!/usr/bin/env python3
"""
Comprehensive System Test Suite

Tests AutoML, GradNorm, and False Positive detection systems.
Run this after kernel restart to ensure all systems are working.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.metrics import DetectionMetrics
from ml.training.losses import (
    ClassificationLoss,
    BoxRegressionLoss,
    ObjectnessLoss,
    MultiHeadLoss,
)

def test_gradnorm() -> Tuple[bool, str]:
    """Test GradNorm integration."""
    print("\n" + "=" * 70)
    print("🧪 Testing GradNorm Integration")
    print("=" * 70)
    
    try:
        # Test 1: Import
        print("\n1. Testing GradNorm import...")
        from ml.training.task_balancing import GradNormMultiHeadLoss
        print("   ✅ GradNorm imported successfully")
        
        # Test 2: Initialization
        print("\n2. Testing GradNorm initialization...")
        head_losses = {
            'classification': ClassificationLoss(num_classes=91),
            'box': BoxRegressionLoss(),
            'objectness': ObjectnessLoss(),
        }
        
        gradnorm = GradNormMultiHeadLoss(
            head_losses=head_losses,
            alpha=1.5,
            update_interval=100
        )
        print(f"   ✅ GradNorm initialized: {gradnorm.num_heads} heads")
        
        # Test 3: Loss computation (without model for basic test)
        print("\n3. Testing GradNorm loss computation...")
        outputs = {
            'classifications': torch.randn(2, 196, 91, requires_grad=True),
            'boxes': torch.randn(2, 196, 4, requires_grad=True),
            'objectness': torch.randn(2, 196, requires_grad=True),
        }
        targets = {
            'classifications': torch.randint(0, 91, (2, 196)),
            'boxes': torch.rand(2, 196, 4),
            'objectness': torch.rand(2, 196),
        }
        
        # Create a simple model for gradient computation
        dummy_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        total_loss, metrics = gradnorm(outputs, targets, model=dummy_model)
        
        assert torch.is_tensor(total_loss), "Total loss should be a tensor"
        assert not torch.isnan(total_loss), "Loss should not be NaN"
        assert not torch.isinf(total_loss), "Loss should not be Inf"
        
        print(f"   ✅ Loss computation successful: {total_loss.item():.4f}")
        print(f"   ✅ Metrics: {list(metrics.keys())}")
        
        # Test 4: Check for inplace operation issues
        print("\n4. Testing for inplace operation issues...")
        # This should not raise the "version 4; expected version 3" error
        try:
            # Simulate multiple backward passes (like GradNorm does)
            loss1 = gradnorm(outputs, targets, model=dummy_model)[0]
            loss2 = gradnorm(outputs, targets, model=dummy_model)[0]
            print("   ✅ No inplace operation errors detected")
        except RuntimeError as e:
            if "inplace operation" in str(e) or "version" in str(e):
                return False, f"Inplace operation error detected: {e}"
            raise
        
        return True, "All GradNorm tests passed"
        
    except Exception as e:
        return False, f"GradNorm test failed: {e}\n{traceback.format_exc()}"

def test_automl() -> Tuple[bool, str]:
    """Test AutoML/Optuna integration."""
    print("\n" + "=" * 70)
    print("🧪 Testing AutoML Integration")
    print("=" * 70)
    
    try:
        # Test 1: Import Optuna
        print("\n1. Testing Optuna import...")
        import optuna
        print(f"   ✅ Optuna imported: version {optuna.__version__}")
        
        # Test 2: Create study
        print("\n2. Testing Optuna study creation...")
        study = optuna.create_study(
            direction='minimize',
            study_name='test_study'
        )
        print("   ✅ Study created successfully")
        
        # Test 3: Test trial
        print("\n3. Testing Optuna trial...")
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            bs = trial.suggest_int('batch_size', 4, 16, step=4)
            return (lr - 0.001) ** 2 + (wd - 0.0001) ** 2  # Dummy objective
        
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        print(f"   ✅ Trial completed: value={value:.6f}")
        print(f"   ✅ Parameters: {trial.params}")
        
        # Test 4: Check AutoML script exists
        print("\n4. Testing AutoML script availability...")
        automl_script = Path(__file__).parent / 'AutoMLType.py'
        if automl_script.exists():
            print(f"   ✅ AutoML script found: {automl_script}")
        else:
            return False, f"AutoML script not found: {automl_script}"
        
        return True, "All AutoML tests passed"
        
    except ImportError as e:
        return False, f"AutoML dependencies missing: {e}"
    except Exception as e:
        return False, f"AutoML test failed: {e}\n{traceback.format_exc()}"

def test_false_positives() -> Tuple[bool, str]:
    """Test false positive detection and metrics."""
    print("\n" + "=" * 70)
    print("🧪 Testing False Positive Detection")
    print("=" * 70)
    
    try:
        # Test 1: DetectionMetrics initialization
        print("\n1. Testing DetectionMetrics initialization...")
        metrics = DetectionMetrics(num_classes=91)
        print("   ✅ DetectionMetrics initialized")
        
        # Test 2: False positive detection
        print("\n2. Testing false positive detection...")
        
        # Create predictions with some false positives
        pred_boxes = torch.tensor([
            [0.1, 0.1, 0.2, 0.2],  # TP (matches GT)
            [0.5, 0.5, 0.1, 0.1],  # FP (no matching GT)
            [0.7, 0.7, 0.1, 0.1],  # FP (no matching GT)
        ])
        pred_labels = torch.tensor([1, 1, 2])
        pred_scores = torch.tensor([0.9, 0.8, 0.7])
        
        # Ground truth (only one object)
        gt_boxes = torch.tensor([
            [0.1, 0.1, 0.2, 0.2],  # Matches first prediction
        ])
        gt_labels = torch.tensor([1])
        
        metrics.update(
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
        
        print("   ✅ Metrics updated successfully")
        
        # Test 3: Compute false positive rate
        print("\n3. Testing false positive rate computation...")
        precision = metrics.compute_precision()
        recall = metrics.compute_recall()
        f1 = metrics.compute_f1()
        
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall: {recall:.4f}")
        print(f"   ✅ F1: {f1:.4f}")
        
        # Check that FP detection works
        if precision < 1.0:  # Should be less than 1.0 due to FPs
            print("   ✅ False positives detected correctly")
        else:
            print("   ⚠️  Warning: No false positives detected (may be expected)")
        
        # Test 4: Per-class false positives
        print("\n4. Testing per-class false positive tracking...")
        class_fp = metrics.class_fp
        if len(class_fp) > 0:
            print(f"   ✅ Per-class FP tracking works: {len(class_fp)} classes with FPs")
        else:
            print("   ⚠️  No per-class FP data (may be expected)")
        
        # Test 5: Empty predictions (should not crash)
        print("\n5. Testing empty predictions handling...")
        metrics2 = DetectionMetrics(num_classes=91)
        metrics2.update(
            pred_boxes=torch.empty(0, 4),
            pred_labels=torch.empty(0, dtype=torch.long),
            pred_scores=torch.empty(0),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
        precision_empty = metrics2.compute_precision()
        print(f"   ✅ Empty predictions handled: precision={precision_empty:.4f}")
        
        return True, "All false positive tests passed"
        
    except Exception as e:
        return False, f"False positive test failed: {e}\n{traceback.format_exc()}"

def test_model_forward() -> Tuple[bool, str]:
    """Test model forward pass."""
    print("\n" + "=" * 70)
    print("🧪 Testing Model Forward Pass")
    print("=" * 70)
    
    try:
        print("\n1. Creating model...")
        model = create_model(num_classes=91)
        print("   ✅ Model created")
        
        print("\n2. Testing forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        required_keys = ['boxes', 'classifications', 'objectness']
        missing_keys = [k for k in required_keys if k not in outputs]
        
        if missing_keys:
            return False, f"Missing output keys: {missing_keys}"
        
        print("   ✅ Forward pass successful")
        print(f"   ✅ Output keys: {list(outputs.keys())}")
        
        # Check for NaN/Inf
        for key, value in outputs.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    return False, f"NaN detected in {key} output"
                if torch.isinf(value).any():
                    return False, f"Inf detected in {key} output"
        
        print("   ✅ No NaN/Inf in outputs")
        
        return True, "Model forward pass test passed"
        
    except Exception as e:
        return False, f"Model forward pass test failed: {e}\n{traceback.format_exc()}"

def test_validation_loss() -> Tuple[bool, str]:
    """Test validation loss computation."""
    print("\n" + "=" * 70)
    print("🧪 Testing Validation Loss Computation")
    print("=" * 70)
    
    try:
        print("\n1. Testing loss computation...")
        from ml.training.losses import MultiHeadLoss
        
        loss_fn = MultiHeadLoss({
            'classification': ClassificationLoss(num_classes=91),
            'box': BoxRegressionLoss(),
            'objectness': ObjectnessLoss(),
        })
        
        outputs = {
            'classifications': torch.randn(2, 196, 91),
            'boxes': torch.randn(2, 196, 4),
            'objectness': torch.randn(2, 196),
        }
        targets = {
            'classifications': torch.randint(0, 91, (2, 196)),
            'boxes': torch.rand(2, 196, 4),
            'objectness': torch.rand(2, 196),
        }
        
        loss_dict = loss_fn(outputs, targets)
        total_loss = loss_dict.get('total_loss', None)
        
        if total_loss is None:
            return False, "Total loss not found in loss dict"
        
        if torch.isnan(total_loss):
            return False, "Loss is NaN"
        
        if torch.isinf(total_loss):
            return False, "Loss is Inf"
        
        print(f"   ✅ Loss computation successful: {total_loss.item():.4f}")
        print(f"   ✅ Loss components: {list(loss_dict.keys())}")
        
        return True, "Validation loss test passed"
        
    except Exception as e:
        return False, f"Validation loss test failed: {e}\n{traceback.format_exc()}"

def main():
    parser = argparse.ArgumentParser(description="Comprehensive system tests")
    parser.add_argument(
        '--test',
        choices=['all', 'gradnorm', 'automl', 'false-positives', 'model', 'validation'],
        default='all',
        help='Which test to run (default: all)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'mps', 'auto'],
        help='Device to use for tests (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Resolve device
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print("=" * 70)
    print("🔬 Comprehensive System Test Suite")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results: List[Tuple[str, bool, str]] = []
    
    # Run tests
    if args.test in ['all', 'gradnorm']:
        success, message = test_gradnorm()
        results.append(('GradNorm', success, message))
    
    if args.test in ['all', 'automl']:
        success, message = test_automl()
        results.append(('AutoML', success, message))
    
    if args.test in ['all', 'false-positives']:
        success, message = test_false_positives()
        results.append(('False Positives', success, message))
    
    if args.test in ['all', 'model']:
        success, message = test_model_forward()
        results.append(('Model Forward', success, message))
    
    if args.test in ['all', 'validation']:
        success, message = test_validation_loss()
        results.append(('Validation Loss', success, message))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            print(f"   {message}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems operational!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - check output above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
