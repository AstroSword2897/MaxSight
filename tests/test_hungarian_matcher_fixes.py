"""Test Hungarian Matcher Robustness

Verifies that the matcher handles edge cases without crashing:
- NaN/Inf in boxes
- Zero-width/height boxes
- Empty ground truth
- Invalid cost matrices"""

import torch
import pytest
from ml.training.matching import (
    compute_matching_cost,
    match_predictions_to_gt,
    match_batch,
    build_matched_pred_targets
)
from ml.utils.batch_validation import (
    validate_boxes,
    validate_labels,
    sanitize_boxes,
    validate_and_sanitize_batch
)


def test_validate_boxes_valid():
    """Test validation of valid boxes"""
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    is_valid, msg = validate_boxes(boxes)
    assert is_valid, f"Valid boxes should pass: {msg}"


def test_validate_boxes_nan():
    """Test validation catches NaN"""
    boxes = torch.tensor([[float('nan'), 0.5, 0.2, 0.3]])
    is_valid, msg = validate_boxes(boxes)
    assert not is_valid
    assert "NaN" in msg


def test_validate_boxes_inf():
    """Test validation catches Inf"""
    boxes = torch.tensor([[0.5, float('inf'), 0.2, 0.3]])
    is_valid, msg = validate_boxes(boxes)
    assert not is_valid
    assert "Inf" in msg


def test_validate_boxes_zero_width():
    """Test validation catches zero width"""
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.3]])
    is_valid, msg = validate_boxes(boxes)
    assert not is_valid
    assert "width" in msg.lower()


def test_sanitize_boxes_nan():
    """Test sanitization replaces NaN"""
    boxes = torch.tensor([[float('nan'), 0.5, 0.2, 0.3]])
    clean_boxes = sanitize_boxes(boxes)
    assert not torch.isnan(clean_boxes).any()
    assert (clean_boxes[:, 2] > 0).all()  # Width positive.
    assert (clean_boxes[:, 3] > 0).all()  # Height positive.


def test_sanitize_boxes_zero_size():
    """Test sanitization fixes zero size"""
    boxes = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    clean_boxes = sanitize_boxes(boxes, min_size=1e-4)
    assert (clean_boxes[:, 2] >= 1e-4).all()
    assert (clean_boxes[:, 3] >= 1e-4).all()


def test_matching_empty_gt():
    """Test matcher handles empty ground truth"""
    pred_boxes = torch.rand(10, 4)
    pred_logits = torch.rand(10, 91)
    gt_boxes = torch.empty(0, 4)
    gt_labels = torch.empty(0, dtype=torch.long)
    
    indices, costs = match_predictions_to_gt(
        pred_boxes, pred_logits, gt_boxes, gt_labels
    )
    
    assert indices.shape == (2, 0)
    assert costs.shape == (0,)


def test_matching_valid_inputs():
    """Test matcher works with valid inputs"""
    pred_boxes = torch.tensor([
        [0.3, 0.3, 0.2, 0.2],
        [0.7, 0.7, 0.15, 0.15]
    ])
    pred_logits = torch.randn(2, 91)
    gt_boxes = torch.tensor([
        [0.3, 0.3, 0.2, 0.2],
        [0.7, 0.7, 0.15, 0.15]
    ])
    gt_labels = torch.tensor([1, 2])
    
    indices, costs = match_predictions_to_gt(
        pred_boxes, pred_logits, gt_boxes, gt_labels
    )
    
    assert indices.shape[0] == 2  # (pred_idx, gt_idx)
    assert indices.shape[1] <= 2  # At most 2 matches.
    assert costs.shape[0] == indices.shape[1]


def test_matching_batch_mixed():
    """Test batch matching with some valid, some empty samples"""
    batch_size = 4
    num_pred = 10
    num_gt = 5
    
    pred_boxes = torch.rand(batch_size, num_pred, 4)
    pred_logits = torch.randn(batch_size, num_pred, 91)
    gt_boxes = torch.zeros(batch_size, num_gt, 4)
    gt_labels = torch.zeros(batch_size, num_gt, dtype=torch.long)
    
    # Make samples 0 and 2 valid.
    gt_boxes[0, :3] = torch.rand(3, 4)
    gt_boxes[0, :3, 2:] = gt_boxes[0, :3, 2:].clamp(min=0.1)  # Valid sizes.
    gt_labels[0, :3] = torch.randint(1, 91, (3,))
    
    gt_boxes[2, :2] = torch.rand(2, 4)
    gt_boxes[2, :2, 2:] = gt_boxes[2, :2, 2:].clamp(min=0.1)
    gt_labels[2, :2] = torch.randint(1, 91, (2,))
    
    indices_list, costs_list = match_batch(
        pred_boxes, pred_logits, gt_boxes, gt_labels
    )
    
    assert len(indices_list) == batch_size
    assert len(costs_list) == batch_size
    
    # Samples 1 and 3 have no matches (empty GT)
    assert indices_list[1].shape[1] == 0
    assert indices_list[3].shape[1] == 0


def test_validate_and_sanitize_batch():
    """Test end-to-end batch validation and sanitization"""
    batch = {
        'images': torch.rand(2, 3, 224, 224),
        'boxes': torch.tensor([
            [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.0, 0.15]],  # Second box has zero width.
            [[0.5, 0.5, 0.1, 0.1], [float('nan'), 0.3, 0.1, 0.1]]  # Second box has NaN.
        ]),
        'labels': torch.tensor([[1, 2], [3, 4]]),
        'urgency': torch.tensor([0, 1])
    }
    
    clean_batch, is_valid, msg = validate_and_sanitize_batch(
        batch, num_classes=91, auto_fix=True
    )
    
    assert is_valid, f"Batch should be sanitized: {msg}"
    assert not torch.isnan(clean_batch['boxes']).any()
    assert (clean_batch['boxes'][:, :, 2] > 0).all()  # All widths positive.
    assert (clean_batch['boxes'][:, :, 3] > 0).all()  # All heights positive.


if __name__ == '__main__':
    # Run tests.
    test_validate_boxes_valid()
    test_validate_boxes_nan()
    test_validate_boxes_inf()
    test_validate_boxes_zero_width()
    test_sanitize_boxes_nan()
    test_sanitize_boxes_zero_size()
    test_matching_empty_gt()
    test_matching_valid_inputs()
    test_matching_batch_mixed()
    test_validate_and_sanitize_batch()
    
    print("OK All Hungarian matcher robustness tests passed!")


