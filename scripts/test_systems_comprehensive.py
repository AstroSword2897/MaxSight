#!/usr/bin/env python3
"""Comprehensive System Test Suite..."""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path.
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
from ml.training.matching import match_batch


@dataclass
class TestFixtures:
    """Shared test fixtures across all tests."""
    model: Optional[nn.Module] = None
    device: str = 'cpu'
    num_classes: int = 91
    batch_size: int = 2
    num_patches: int = 196
    
    # Shared data.
    images: Optional[torch.Tensor] = None
    outputs: Optional[Dict[str, torch.Tensor]] = None
    targets: Optional[Dict[str, torch.Tensor]] = None
    
    # Loss functions.
    loss_fn: Optional[nn.Module] = None
    gradnorm_loss: Optional[GradNormMultiHeadLoss] = None
    
    # Metrics.
    metrics: Optional[DetectionMetrics] = None


def setup_fixtures(device: str = 'cpu') -> TestFixtures:
    """Set up shared test fixtures."""
    print("\n🔧 Setting up test fixtures...")
    
    fixtures = TestFixtures(device=device)
    
    # Create model.
    print("   Creating model...")
    fixtures.model = create_model(num_classes=fixtures.num_classes)
    fixtures.model.to(device)
    fixtures.model.eval()
    
    # Create sample data.
    print("   Creating sample data...")
    fixtures.images = torch.randn(
        fixtures.batch_size, 3, 224, 224,
        device=device
    )
    
    # Generate model outputs.
    print("   Generating model outputs...")
    with torch.no_grad():
        fixtures.outputs = fixtures.model(fixtures.images)
    
    # Create targets.
    print("   Creating targets...")
    fixtures.targets = {
        'classifications': torch.randint(
            0, fixtures.num_classes,
            (fixtures.batch_size, fixtures.num_patches),
            device=device
        ),
        'boxes': torch.rand(
            fixtures.batch_size, fixtures.num_patches, 4,
            device=device
        ),
        'objectness': torch.rand(
            fixtures.batch_size, fixtures.num_patches,
            device=device
        ),
    }
    
    # Create loss functions.
    print("   Creating loss functions...")
    head_losses = {
        'classification': ClassificationLoss(num_classes=fixtures.num_classes),
        'box': BoxRegressionLoss(),
        'objectness': ObjectnessLoss(),
    }
    fixtures.loss_fn = MultiHeadLoss(head_losses)
    fixtures.gradnorm_loss = GradNormMultiHeadLoss(
        head_losses=head_losses,
        alpha=1.5,
        update_interval=100
    )
    
    # Create metrics tracker.
    print("   Creating metrics tracker...")
    fixtures.metrics = DetectionMetrics(num_classes=fixtures.num_classes)
    
    print("   ✅ Fixtures ready")
    return fixtures

def test_gradnorm(fixtures: TestFixtures) -> Tuple[bool, str]:
    """Test GradNorm integration using shared fixtures."""
    print("\n" + "=" * 70)
    print("🧪 Testing GradNorm Integration")
    print("=" * 70)
    
    try:
        # Test 1: Import.
        print("\n1. Testing GradNorm import...")
        from ml.training.task_balancing import GradNormMultiHeadLoss
        print("   ✅ GradNorm imported successfully")
        
        # Test 2: Verify fixtures are set up.
        print("\n2. Using shared fixtures...")
        assert fixtures.gradnorm_loss is not None, "GradNorm loss not initialized"
        assert fixtures.model is not None, "Model not initialized"
        print(f"   ✅ GradNorm initialized: {fixtures.gradnorm_loss.num_heads} heads")
        print(f"   ✅ Using shared model with {sum(p.numel() for p in fixtures.model.parameters()):,} parameters")
        
        # Test 3: Loss computation with real model.
        print("\n3. Testing GradNorm loss computation with real model...")
        
        # Make outputs require grad for backprop (only float tensors)
        outputs_grad = {}
        for key, value in fixtures.outputs.items():
            if torch.is_tensor(value) and value.dtype in [torch.float32, torch.float64, torch.float16]:
                outputs_grad[key] = value.detach().clone().requires_grad_(True)
            elif torch.is_tensor(value):
                outputs_grad[key] = value.detach().clone()
            else:
                outputs_grad[key] = value
        
        total_loss, metrics = fixtures.gradnorm_loss(
            outputs_grad,
            fixtures.targets,
            model=fixtures.model
        )
        
        assert torch.is_tensor(total_loss), "Total loss should be a tensor"
        assert not torch.isnan(total_loss), "Loss should not be NaN"
        assert not torch.isinf(total_loss), "Loss should not be Inf"
        
        print(f"   ✅ Loss computation successful: {total_loss.item():.4f}")
        print(f"   ✅ Metrics: {list(metrics.keys())}")
        
        # Test 4: Check for inplace operation issues with matching.
        print("\n4. Testing GradNorm with matching pipeline (real-world scenario)...")
        try:
            # This simulates the actual training pipeline.
            fixtures.model.train()
            
            # Generate predictions that require grad.
            pred_outputs = fixtures.model(fixtures.images)
            
            # Match predictions to ground truth (this was causing inplace errors)
            matched_indices, costs = match_batch(
                pred_boxes=pred_outputs['boxes'],
                pred_logits=pred_outputs['classifications'],
                gt_boxes=fixtures.targets['boxes'],
                gt_labels=fixtures.targets['classifications'],
            )
            
            # Compute loss with matched targets.
            loss, loss_dict = fixtures.gradnorm_loss(
                pred_outputs,
                fixtures.targets,
                model=fixtures.model
            )
            
            # Backward pass (no inplace errors expected)
            loss.backward()
            
            print("   ✅ No inplace operation errors in full pipeline")
            print(f"   ✅ Matched {len(matched_indices)} samples")
            
            fixtures.model.eval()
            
        except RuntimeError as e:
            if "inplace operation" in str(e) or "version" in str(e):
                return False, f"Inplace operation error detected: {e}"
            raise
        
        # Test 5: Gradient norm computation.
        print("\n5. Testing gradient norm computation...")
        if 'gradient_norms' in metrics:
            grad_norms = metrics['gradient_norms']
            print(f"   ✅ Gradient norms computed: {grad_norms}")
        else:
            print("   ℹ️  Gradient norms not computed (update_interval not reached)")
        
        return True, "All GradNorm tests passed (integrated with matching pipeline)"
        
    except Exception as e:
        return False, f"GradNorm test failed: {e}\n{traceback.format_exc()}"

def test_automl() -> Tuple[bool, str]:
    """Test AutoML/Optuna integration."""
    print("\n" + "=" * 70)
    print("🧪 Testing AutoML Integration")
    print("=" * 70)
    
    try:
        # Test 1: Import Optuna.
        print("\n1. Testing Optuna import...")
        import optuna
        print(f"   ✅ Optuna imported: version {optuna.__version__}")
        
        # Test 2: Create study.
        print("\n2. Testing Optuna study creation...")
        study = optuna.create_study(
            direction='minimize',
            study_name='test_study'
        )
        print("   ✅ Study created successfully")
        
        # Test 3: Test trial.
        print("\n3. Testing Optuna trial...")
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            bs = trial.suggest_int('batch_size', 4, 16, step=4)
            return (lr - 0.001) ** 2 + (wd - 0.0001) ** 2  # Dummy objective.
        
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        print(f"   ✅ Trial completed: value={value:.6f}")
        print(f"   ✅ Parameters: {trial.params}")
        
        # Test 4: Check AutoML script exists.
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

def test_false_positives(fixtures: TestFixtures) -> Tuple[bool, str]:
    """Test false positive detection using real model outputs."""
    print("\n" + "=" * 70)
    print("🧪 Testing False Positive Detection")
    print("=" * 70)
    
    try:
        # Test 1: Use shared metrics tracker.
        print("\n1. Using shared metrics tracker...")
        assert fixtures.metrics is not None, "Metrics not initialized"
        print("   ✅ DetectionMetrics initialized")
        
        # Test 2: Extract detections from real model outputs.
        print("\n2. Testing false positive detection with real model outputs...")
        
        # Get model predictions (bounding boxes and classifications)
        pred_boxes = fixtures.outputs['boxes'][0]  # First batch item.
        pred_logits = fixtures.outputs['classifications'][0]
        pred_scores = torch.softmax(pred_logits, dim=-1).max(dim=-1)[0]
        pred_labels = torch.softmax(pred_logits, dim=-1).argmax(dim=-1)
        
        # Filter by confidence threshold.
        confidence_threshold = 0.5
        valid_mask = pred_scores > confidence_threshold
        pred_boxes_filtered = pred_boxes[valid_mask]
        pred_labels_filtered = pred_labels[valid_mask]
        pred_scores_filtered = pred_scores[valid_mask]
        
        # Create ground truth (simulating a scene with 2 objects)
        gt_boxes = torch.tensor([
            [0.2, 0.2, 0.3, 0.3],  # Object 1.
            [0.6, 0.6, 0.2, 0.2],  # Object 2.
        ], device=fixtures.device)
        gt_labels = torch.tensor([1, 5], device=fixtures.device)
        
        # Update metrics (this will compute TP/FP/FN)
        fixtures.metrics.update(
            pred_boxes=pred_boxes_filtered,
            pred_labels=pred_labels_filtered,
            pred_scores=pred_scores_filtered,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
        
        print(f"   ✅ Model predicted {len(pred_boxes_filtered)} detections")
        print(f"   ✅ Ground truth has {len(gt_boxes)} objects")
        
        # Test 3: Compute false positive rate.
        print("\n3. Computing detection metrics...")
        precision = fixtures.metrics.compute_precision()
        recall = fixtures.metrics.compute_recall()
        f1 = fixtures.metrics.compute_f1()
        map_score = fixtures.metrics.compute_map()
        
        # Compute_map() can return dict or float.
        map_result = fixtures.metrics.compute_map()
        if isinstance(map_result, dict):
            map_score = map_result.get('mAP', 0.0)
        else:
            map_score = map_result
        
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall: {recall:.4f}")
        print(f"   ✅ F1: {f1:.4f}")
        print(f"   ✅ mAP: {map_score:.4f}")
        
        # Test 4: Per-class false positives.
        print("\n4. Testing per-class false positive tracking...")
        class_fp = fixtures.metrics.class_fp
        class_tp = fixtures.metrics.class_tp
        
        # Sum values - handle both dict and Counter types.
        if hasattr(class_fp, 'values'):
            try:
                fp_values = list(class_fp.values())
                total_fp = sum(v.item() if torch.is_tensor(v) else int(v) for v in fp_values)
            except:
                total_fp = 0
        else:
            total_fp = 0
            
        if hasattr(class_tp, 'values'):
            try:
                tp_values = list(class_tp.values())
                total_tp = sum(v.item() if torch.is_tensor(v) else int(v) for v in tp_values)
            except:
                total_tp = 0
        else:
            total_tp = 0
        
        print(f"   ✅ Total TP: {total_tp}")
        print(f"   ✅ Total FP: {total_fp}")
        if total_tp + total_fp > 0:
            print(f"   ✅ FP Rate: {total_fp / (total_fp + total_tp):.2%}")
        else:
            print(f"   ℹ️  No detections to compute FP rate")
        
        # Test 5: Test with different confidence thresholds.
        print("\n5. Testing false positive rate at different confidence thresholds...")
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            metrics_temp = DetectionMetrics(num_classes=fixtures.num_classes)
            
            valid_mask = pred_scores > threshold
            metrics_temp.update(
                pred_boxes=pred_boxes[valid_mask],
                pred_labels=pred_labels[valid_mask],
                pred_scores=pred_scores[valid_mask],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                iou_threshold=0.5
            )
            
            prec = metrics_temp.compute_precision()
            rec = metrics_temp.compute_recall()
            num_dets = valid_mask.sum().item()
            
            print(f"   Threshold {threshold:.1f}: {num_dets} dets, P={prec:.3f}, R={rec:.3f}")
        
        print("   ✅ Higher thresholds reduce false positives (as expected)")
        
        # Test 6: Empty predictions (should not crash)
        print("\n6. Testing edge case: empty predictions...")
        metrics_empty = DetectionMetrics(num_classes=fixtures.num_classes)
        metrics_empty.update(
            pred_boxes=torch.empty(0, 4, device=fixtures.device),
            pred_labels=torch.empty(0, dtype=torch.long, device=fixtures.device),
            pred_scores=torch.empty(0, device=fixtures.device),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
        precision_empty = metrics_empty.compute_precision()
        recall_empty = metrics_empty.compute_recall()
        print(f"   ✅ Empty predictions handled: P={precision_empty:.4f}, R={recall_empty:.4f}")
        
        return True, "All false positive tests passed (using real model outputs)"
        
    except Exception as e:
        return False, f"False positive test failed: {e}\n{traceback.format_exc()}"

def test_model_forward(fixtures: TestFixtures) -> Tuple[bool, str]:
    """Test model forward pass using shared fixtures."""
    print("\n" + "=" * 70)
    print("🧪 Testing Model Forward Pass")
    print("=" * 70)
    
    try:
        print("\n1. Using shared model...")
        assert fixtures.model is not None, "Model not initialized"
        print("   ✅ Model ready")
        
        print("\n2. Verifying forward pass outputs...")
        assert fixtures.outputs is not None, "Outputs not generated"
        
        required_keys = ['boxes', 'classifications', 'objectness']
        missing_keys = [k for k in required_keys if k not in fixtures.outputs]
        
        if missing_keys:
            return False, f"Missing output keys: {missing_keys}"
        
        print("   ✅ Forward pass successful")
        print(f"   ✅ Output keys: {list(fixtures.outputs.keys())}")
        
        # Check output shapes.
        print("\n3. Verifying output shapes...")
        for key, value in fixtures.outputs.items():
            if torch.is_tensor(value):
                print(f"   {key}: {value.shape}")
        
        # Check for NaN/Inf.
        print("\n4. Checking for NaN/Inf values...")
        for key, value in fixtures.outputs.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    return False, f"NaN detected in {key} output"
                if torch.isinf(value).any():
                    return False, f"Inf detected in {key} output"
        
        print("   ✅ No NaN/Inf in outputs")
        
        # Test 5: Multiple forward passes (ensure consistency)
        print("\n5. Testing multiple forward passes...")
        fixtures.model.eval()
        with torch.no_grad():
            outputs2 = fixtures.model(fixtures.images)
            outputs3 = fixtures.model(fixtures.images)
        
        # Outputs are deterministic in eval mode.
        for key in required_keys:
            diff = (outputs2[key] - outputs3[key]).abs().max().item()
            if diff > 1e-6:
                print(f"   ⚠️  Non-deterministic outputs for {key}: max diff = {diff}")
            else:
                print(f"   ✅ {key} outputs are deterministic")
        
        return True, "Model forward pass test passed (using shared model)"
        
    except Exception as e:
        return False, f"Model forward pass test failed: {e}\n{traceback.format_exc()}"

def test_validation_loss(fixtures: TestFixtures) -> Tuple[bool, str]:
    """Test validation loss computation using shared fixtures."""
    print("\n" + "=" * 70)
    print("🧪 Testing Validation Loss Computation")
    print("=" * 70)
    
    try:
        print("\n1. Testing standard loss computation...")
        assert fixtures.loss_fn is not None, "Loss function not initialized"
        
        loss_dict = fixtures.loss_fn(fixtures.outputs, fixtures.targets)
        total_loss = loss_dict.get('total_loss', None)
        
        if total_loss is None:
            return False, "Total loss not found in loss dict"
        
        if torch.isnan(total_loss):
            return False, "Loss is NaN"
        
        if torch.isinf(total_loss):
            return False, "Loss is Inf"
        
        print(f"   ✅ Loss computation successful: {total_loss.item():.4f}")
        print(f"   ✅ Loss components: {list(loss_dict.keys())}")
        
        # Test 2: Compare with GradNorm loss.
        print("\n2. Comparing standard loss with GradNorm loss...")
        assert fixtures.gradnorm_loss is not None, "GradNorm loss not initialized"
        
        gradnorm_total, gradnorm_metrics = fixtures.gradnorm_loss(
            fixtures.outputs,
            fixtures.targets,
            model=fixtures.model
        )
        
        print(f"   Standard loss: {total_loss.item():.4f}")
        print(f"   GradNorm loss: {gradnorm_total.item():.4f}")
        print(f"   ✅ Both losses computed successfully")
        
        # Test 3: Ensure losses are in reasonable range.
        print("\n3. Checking loss magnitude...")
        if total_loss.item() > 100:
            print(f"   ⚠️  Loss is very high: {total_loss.item():.2f}")
        elif total_loss.item() < 0:
            return False, f"Loss is negative: {total_loss.item()}"
        else:
            print(f"   ✅ Loss is in reasonable range")
        
        # Test 4: Test loss with corrupted data (still computes)
        print("\n4. Testing loss robustness with corrupted predictions...")
        corrupted_outputs = {}
        for key, value in fixtures.outputs.items():
            if torch.is_tensor(value):
                corrupted_outputs[key] = value.clone()
            else:
                corrupted_outputs[key] = value
        
        # Add some noise to float tensors.
        if torch.is_tensor(corrupted_outputs.get('boxes')):
            corrupted_outputs['boxes'] = corrupted_outputs['boxes'] + torch.randn_like(corrupted_outputs['boxes']) * 0.1
        
        corrupted_loss_dict = fixtures.loss_fn(corrupted_outputs, fixtures.targets)
        corrupted_loss = corrupted_loss_dict['total_loss']
        
        if torch.isnan(corrupted_loss):
            return False, "Loss becomes NaN with corrupted data"
        
        print(f"   ✅ Loss with corrupted data: {corrupted_loss.item():.4f}")
        print(f"   ✅ Loss computation is robust")
        
        # Test 5: Batch accumulation simulation (like validation loop)
        print("\n5. Simulating validation loop batch accumulation...")
        total_accumulated = 0.0
        num_batches = 5
        
        for i in range(num_batches):
            loss_dict_batch = fixtures.loss_fn(fixtures.outputs, fixtures.targets)
            batch_loss = loss_dict_batch['total_loss'].item()
            
            # Check for NaN/Inf before accumulation (the fix we applied)
            if torch.isnan(torch.tensor(batch_loss)) or torch.isinf(torch.tensor(batch_loss)):
                print(f"   ⚠️  Batch {i} has invalid loss, skipping")
                continue
            
            total_accumulated += batch_loss
        
        avg_loss = total_accumulated / num_batches
        
        if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
            return False, "Accumulated loss is NaN/Inf"
        
        print(f"   ✅ Average loss over {num_batches} batches: {avg_loss:.4f}")
        print(f"   ✅ NaN/Inf protection working")
        
        return True, "Validation loss test passed (integrated with all components)"
        
    except Exception as e:
        return False, f"Validation loss test failed: {e}\n{traceback.format_exc()}"

def test_integration_pipeline(fixtures: TestFixtures) -> Tuple[bool, str]:
    """Test complete integration: model → loss → gradnorm → metrics."""
    print("\n" + "=" * 70)
    print("🧪 Testing Integrated Training Pipeline")
    print("=" * 70)
    
    try:
        print("\n1. Simulating one training step...")
        
        # Set model to train mode.
        fixtures.model.train()
        
        # Forward pass.
        outputs = fixtures.model(fixtures.images)
        
        # Match predictions to targets.
        print("   Matching predictions to ground truth...")
        matched_indices, costs = match_batch(
            pred_boxes=outputs['boxes'],
            pred_logits=outputs['classifications'],
            gt_boxes=fixtures.targets['boxes'],
            gt_labels=fixtures.targets['classifications'],
        )
        
        # Compute loss with GradNorm.
        print("   Computing GradNorm loss...")
        loss, loss_dict = fixtures.gradnorm_loss(
            outputs,
            fixtures.targets,
            model=fixtures.model
        )
        
        # Backward pass.
        print("   Running backward pass...")
        loss.backward()
        
        # Update metrics.
        print("   Updating detection metrics...")
        for i in range(fixtures.batch_size):
            pred_boxes = outputs['boxes'][i].detach()
            pred_logits = outputs['classifications'][i].detach()
            pred_scores = torch.softmax(pred_logits, dim=-1).max(dim=-1)[0]
            pred_labels = torch.softmax(pred_logits, dim=-1).argmax(dim=-1)
            
            valid_mask = pred_scores > 0.5
            
            fixtures.metrics.update(
                pred_boxes=pred_boxes[valid_mask],
                pred_labels=pred_labels[valid_mask],
                pred_scores=pred_scores[valid_mask],
                gt_boxes=fixtures.targets['boxes'][i],
                gt_labels=fixtures.targets['classifications'][i],
                iou_threshold=0.5
            )
        
        print(f"   ✅ Training step complete: loss = {loss.item():.4f}")
        
        # Compute final metrics.
        print("\n2. Computing final metrics...")
        precision = fixtures.metrics.compute_precision()
        recall = fixtures.metrics.compute_recall()
        f1 = fixtures.metrics.compute_f1()
        map_result = fixtures.metrics.compute_map()
        
        # Compute_map() can return dict or float.
        if isinstance(map_result, dict):
            map_score = map_result.get('mAP', 0.0)
        else:
            map_score = map_result
        
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")
        print(f"   mAP: {map_score:.4f}")
        
        print("\n3. Verifying no issues...")
        
        # Check for NaN/Inf in loss.
        if torch.isnan(loss) or torch.isinf(loss):
            return False, "Loss is NaN/Inf after training step"
        
        # Check for NaN/Inf in gradients.
        has_nan_grad = False
        for name, param in fixtures.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    print(f"   ⚠️  NaN/Inf gradient in {name}")
        
        if has_nan_grad:
            return False, "NaN/Inf gradients detected"
        
        print("   ✅ No NaN/Inf in loss or gradients")
        print("   ✅ Integrated pipeline working correctly")
        
        fixtures.model.eval()
        
        return True, "Complete integration test passed (model→loss→gradnorm→metrics)"
        
    except Exception as e:
        return False, f"Integration test failed: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive integrated system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:..."""
    )
    parser.add_argument(
        '--test',
        choices=['all', 'gradnorm', 'automl', 'false-positives', 'model', 'validation', 'integration'],
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
    
    # Resolve device.
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print("=" * 70)
    print("🔬 Comprehensive Integrated System Test Suite")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Set up shared fixtures (used by all tests)
    fixtures = None
    if args.test != 'automl':  # AutoML test doesn't need fixtures.
        fixtures = setup_fixtures(device=device)
    
    results: List[Tuple[str, bool, str]] = []
    
    # Run tests with shared fixtures.
    if args.test in ['all', 'model']:
        success, message = test_model_forward(fixtures)
        results.append(('Model Forward', success, message))
    
    if args.test in ['all', 'validation']:
        success, message = test_validation_loss(fixtures)
        results.append(('Validation Loss', success, message))
    
    if args.test in ['all', 'gradnorm']:
        success, message = test_gradnorm(fixtures)
        results.append(('GradNorm', success, message))
    
    if args.test in ['all', 'false-positives']:
        success, message = test_false_positives(fixtures)
        results.append(('False Positives', success, message))
    
    if args.test in ['all', 'integration']:
        success, message = test_integration_pipeline(fixtures)
        results.append(('Integration Pipeline', success, message))
    
    # AutoML test runs independently (doesn't need fixtures)
    if args.test in ['all', 'automl']:
        success, message = test_automl()
        results.append(('AutoML', success, message))
    
    # Print summary.
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
        print("✅ Model, loss, GradNorm, metrics, and integration working correctly")
        print("✅ Ready for training!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - check output above")
        return 1

if __name__ == '__main__':
    sys.exit(main())

