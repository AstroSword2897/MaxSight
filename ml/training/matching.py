"""
Hungarian Matching for Multi-Object Detection

Matches predicted boxes to ground truth using optimal bipartite matching.

Based on DETR's approach with combined classification + bbox + GIoU costs.

"""


import torch
import numpy as np
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


def compute_giou_cost(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute GIoU-based cost (1 - GIoU) between prediction and ground truth boxes.
    Lower cost = better match.
    
    Boxes are in normalized (x, y, w, h) format.
    """
    # Convert center format to corners for easier computation
    p_x1 = pred_boxes[:, 0]
    p_y1 = pred_boxes[:, 1]
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]
    
    g_x1 = gt_boxes[:, 0]
    g_y1 = gt_boxes[:, 1]
    g_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]
    g_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]
    
    # Expand for pairwise comparison: [N, 1] and [1, M] -> broadcasting to [N, M]
    p_x1 = p_x1.unsqueeze(1)
    p_y1 = p_y1.unsqueeze(1)
    p_x2 = p_x2.unsqueeze(1)
    p_y2 = p_y2.unsqueeze(1)
    
    g_x1 = g_x1.unsqueeze(0)
    g_y1 = g_y1.unsqueeze(0)
    g_x2 = g_x2.unsqueeze(0)
    g_y2 = g_y2.unsqueeze(0)
    
    # Intersection area
    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Union area
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]  # w * h
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    
    pred_area = pred_area.unsqueeze(1)
    gt_area = gt_area.unsqueeze(0)
    union_area = pred_area + gt_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-8)
    
    # Enclosing box (for GIoU)
    enclose_x1 = torch.min(p_x1, g_x1)
    enclose_y1 = torch.min(p_y1, g_y1)
    enclose_x2 = torch.max(p_x2, g_x2)
    enclose_y2 = torch.max(p_y2, g_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU = IoU - (enclosing_area - union_area) / enclosing_area
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)
    
    return 1.0 - giou


def compute_matching_cost(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0
) -> torch.Tensor:
    """
    Build cost matrix for Hungarian algorithm.
    
    Each cell [i,j] represents cost of assigning prediction i to ground truth j.
    Combines three components: classification, bounding box, and GIoU.
    """
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    
    # CRITICAL: Clone and convert to float32 to avoid precision issues and inplace operation errors
    # Cloning ensures we don't modify the original tensors which may be part of the computation graph
    pred_boxes = pred_boxes.clone().float()
    pred_logits = pred_logits.clone().float()
    gt_boxes = gt_boxes.clone().float()
    
    # Validate inputs - check for NaN/Inf
    if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
        raise ValueError(f"Invalid pred_boxes: NaN={torch.isnan(pred_boxes).sum()}, Inf={torch.isinf(pred_boxes).sum()}")
    if torch.isnan(gt_boxes).any() or torch.isinf(gt_boxes).any():
        raise ValueError(f"Invalid gt_boxes: NaN={torch.isnan(gt_boxes).sum()}, Inf={torch.isinf(gt_boxes).sum()}")
    if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
        raise ValueError(f"Invalid pred_logits: NaN={torch.isnan(pred_logits).sum()}, Inf={torch.isinf(pred_logits).sum()}")
    
    # Validate box dimensions (width, height > 0)
    # Use a small epsilon to account for floating point precision issues
    # CRITICAL: Use non-inplace operations to avoid breaking computation graph
    min_dim = 1e-5
    if (gt_boxes[:, 2] < min_dim).any() or (gt_boxes[:, 3] < min_dim).any():
        # Auto-fix: clamp to minimum using non-inplace operations
        gt_boxes = gt_boxes.clone()
        gt_boxes[:, 2] = torch.clamp(gt_boxes[:, 2], min=min_dim)
        gt_boxes[:, 3] = torch.clamp(gt_boxes[:, 3], min=min_dim)
    if (pred_boxes[:, 2] < min_dim).any() or (pred_boxes[:, 3] < min_dim).any():
        # Auto-fix: clamp to minimum using non-inplace operations
        pred_boxes = pred_boxes.clone()
        pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=min_dim)
        pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=min_dim)
    
    # Classification cost: negative log-likelihood
    # We want high confidence on the correct class, so we use -log(p)
    probs = torch.softmax(pred_logits, dim=-1)
    # Add small epsilon to prevent log(0)
    class_cost = -torch.log(probs[:, gt_labels] + 1e-8)  # [num_pred, num_gt]
    
    # L1 distance between box centers and sizes
    bbox_cost = torch.cdist(
        pred_boxes.unsqueeze(1),
        gt_boxes.unsqueeze(0),
        p=1
    ).squeeze()  # [num_pred, num_gt]
    
    # GIoU cost
    giou_cost = compute_giou_cost(pred_boxes, gt_boxes)
    
    # Weighted combination
    total_cost = (
        lambda_class * class_cost +
        lambda_bbox * bbox_cost +
        lambda_giou * giou_cost
    )
    
    # Final validation - ensure cost matrix is finite
    if torch.isnan(total_cost).any() or torch.isinf(total_cost).any():
        raise ValueError(f"Cost matrix contains invalid values: NaN={torch.isnan(total_cost).sum()}, Inf={torch.isinf(total_cost).sum()}")
    
    return total_cost


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    use_hungarian: bool = True  # Default to Hungarian for optimal matching
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find best assignment between predictions and ground truth.
    
        Arguments:
        use_hungarian: If True, use proper Hungarian algorithm (globally optimal but slower).
                      If False, use greedy matching (faster, usually sufficient).
    
    Returns:
        indices: [2, num_matched] tensor with (pred_idx, gt_idx) pairs
        costs: [num_matched] tensor with cost for each match
    """
    # Handle empty predictions or ground truth
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device),
            torch.empty((0,), device=pred_boxes.device)
        )
    
    # Sanitize boxes before cost computation to prevent failures
    # Clamp invalid dimensions and replace NaN/Inf
    pred_boxes = pred_boxes.float()
    gt_boxes = gt_boxes.float()
    pred_logits = pred_logits.float()
    
    # Replace NaN/Inf with small valid values first
    pred_boxes = torch.where(torch.isfinite(pred_boxes), pred_boxes, torch.zeros_like(pred_boxes))
    gt_boxes = torch.where(torch.isfinite(gt_boxes), gt_boxes, torch.zeros_like(gt_boxes))
    pred_logits = torch.where(torch.isfinite(pred_logits), pred_logits, torch.zeros_like(pred_logits))
    
    # Fix invalid box dimensions (width/height must be > 0)
    pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=1e-4)  # width
    pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=1e-4)  # height
    gt_boxes[:, 2] = torch.clamp(gt_boxes[:, 2], min=1e-4)  # width
    gt_boxes[:, 3] = torch.clamp(gt_boxes[:, 3], min=1e-4)  # height
    
    # Ensure boxes are within valid range [0, 1] for normalized coordinates
    pred_boxes = torch.clamp(pred_boxes, min=0.0, max=1.0)
    gt_boxes = torch.clamp(gt_boxes, min=0.0, max=1.0)
    
    # Compute cost matrix [num_pred, num_gt] - this validates inputs
    try:
        cost = compute_matching_cost(
            pred_boxes, pred_logits, gt_boxes, gt_labels,
            lambda_class, lambda_bbox, lambda_giou
        )
    except ValueError as e:
        # If cost computation still fails after sanitization, return empty matches
        # Only log at DEBUG level to reduce log spam during early training
        logger.debug(f"Cost computation failed after sanitization: {e}. Returning empty matches.")
        return (
            torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device),
            torch.empty((0,), device=pred_boxes.device)
        )
    
    if use_hungarian:
        # Use proper Hungarian algorithm for globally optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            cost_np = cost.detach().cpu().numpy()
            
            # Final check: scipy will fail if cost matrix has NaN/Inf
            if not np.isfinite(cost_np).all():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Cost matrix has non-finite values, falling back to greedy matching")
                use_hungarian = False
            else:
                pred_indices, gt_indices = linear_sum_assignment(cost_np)
                
                pred_idx = torch.tensor(pred_indices, dtype=torch.long, device=pred_boxes.device)
                gt_idx = torch.tensor(gt_indices, dtype=torch.long, device=pred_boxes.device)
                matched_costs = cost[pred_idx, gt_idx]
                
                indices = torch.stack([pred_idx, gt_idx])
                return indices, matched_costs
        except (ImportError, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Hungarian matching failed ({e}), falling back to greedy matching")
            use_hungarian = False
    
    # Greedy matching (faster, usually sufficient)
    num_gt = cost.shape[1]
    pred_idx = []
    gt_idx = []
    matched_costs = []
    
    # For each GT, assign the lowest-cost available prediction
    used_preds = set()
    for gt_i in range(num_gt):
        gt_costs = cost[:, gt_i]
        
        # Find lowest cost pred that hasn't been assigned yet
        for rank in gt_costs.argsort():
            pred_i = int(rank.item())
            if pred_i not in used_preds:
                pred_idx.append(pred_i)
                gt_idx.append(gt_i)
                matched_costs.append(gt_costs[pred_i])
                used_preds.add(pred_i)
                break
    
    if len(pred_idx) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device),
            torch.empty((0,), device=pred_boxes.device)
        )
    
    pred_idx = torch.tensor(pred_idx, dtype=torch.long, device=pred_boxes.device)
    gt_idx = torch.tensor(gt_idx, dtype=torch.long, device=pred_boxes.device)
    matched_costs = torch.stack(matched_costs)
    
    indices = torch.stack([pred_idx, gt_idx])
    
    return indices, matched_costs


def match_batch(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Match predictions to ground truth for a full batch.
    
        Arguments:
        pred_boxes: [batch, num_pred, 4] in (x, y, w, h) format
        pred_logits: [batch, num_pred, num_classes]
        gt_boxes: [batch, num_gt, 4] in (x, y, w, h) format
        gt_labels: [batch, num_gt]
    
    Returns:
        indices_list: List of [2, num_matched_i] tensors per sample
        costs_list: List of [num_matched_i] tensors per sample
    """
    batch_size = pred_boxes.shape[0]
    indices_list = []
    costs_list = []
    
    for i in range(batch_size):
        # Skip samples with no ground truth or invalid boxes
        valid_gt = (gt_boxes[i, :, 2] > 0) & (gt_boxes[i, :, 3] > 0)
        
        # Additional validation: check for NaN/Inf in this sample
        if torch.isnan(gt_boxes[i]).any() or torch.isinf(gt_boxes[i]).any():
            logger.warning(f"Sample {i} has NaN/Inf in gt_boxes, skipping")
            indices_list.append(
                torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device)
            )
            costs_list.append(
                torch.empty((0,), device=pred_boxes.device)
            )
            continue
        
        # CRITICAL: Clone tensors before sanitizing to avoid breaking computation graph
        # Inplace operations (masked_fill_) break GradNorm gradient computation
        pred_boxes_i = pred_boxes[i].clone()
        pred_logits_i = pred_logits[i].clone()
        
        # Sanitize NaN/Inf predictions instead of skipping - allows training to continue
        if torch.isnan(pred_boxes_i).any() or torch.isinf(pred_boxes_i).any():
            logger.warning(f"Sample {i} has NaN/Inf in pred_boxes, sanitizing")
            # Replace NaN/Inf with small valid boxes (use non-inplace operations)
            nan_mask = torch.isnan(pred_boxes_i) | torch.isinf(pred_boxes_i)
            default_box = torch.tensor([0.5, 0.5, 0.1, 0.1], device=pred_boxes.device, dtype=pred_boxes.dtype)
            # Use non-inplace masked_fill to avoid breaking computation graph
            for j in range(4):
                pred_boxes_i[:, j] = pred_boxes_i[:, j].masked_fill(nan_mask[:, j], default_box[j])
            # Also sanitize logits if needed
            if torch.isnan(pred_logits_i).any() or torch.isinf(pred_logits_i).any():
                logit_nan_mask = torch.isnan(pred_logits_i) | torch.isinf(pred_logits_i)
                pred_logits_i = pred_logits_i.masked_fill(logit_nan_mask, 0.0)
        
        if valid_gt.sum() == 0:
            indices_list.append(
                torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device)
            )
            costs_list.append(
                torch.empty((0,), device=pred_boxes.device)
            )
            continue
        
        # Get valid ground truth for this sample
        gt_boxes_valid = gt_boxes[i][valid_gt]
        gt_labels_valid = gt_labels[i][valid_gt]
        
        # Find matches (with built-in error handling)
        # Use sanitized cloned tensors instead of original tensors
        indices, costs = match_predictions_to_gt(
            pred_boxes_i,
            pred_logits_i,
            gt_boxes_valid,
            gt_labels_valid,
            lambda_class, lambda_bbox, lambda_giou
        )
        
        indices_list.append(indices)
        costs_list.append(costs)
    
    return indices_list, costs_list


def build_matched_pred_targets(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Use Hungarian matching to align per-location predictions to per-object targets.
    Returns (aligned_pred, aligned_target) dicts with keys expected by loss heads:
    objectness, classification, box, distance, urgency (urgency left as-is from batch).

    - objectness: pred [B, N], target [B, N] with 1 at matched locations, 0 elsewhere
    - classification: pred [total_matched, C], target [total_matched]
    - box: pred [total_matched, 4], target [total_matched, 4]
    - distance: pred [total_matched, 3], target [total_matched]
    - urgency: pred [B, 4], target [B] (unchanged)
    """
    import torch
    B = outputs["boxes"].size(0)
    N = outputs["boxes"].size(1)
    device = outputs["boxes"].device
    pred_boxes = outputs["boxes"]
    pred_logits = outputs["classifications"]
    gt_boxes = targets["boxes"]
    gt_labels = targets["labels"].float().clamp(min=0).long()

    indices_list, _ = match_batch(pred_boxes, pred_logits, gt_boxes, gt_labels)

    # Objectness target: [B, N] with 1 at matched pred indices
    target_objectness = torch.zeros(B, N, device=device, dtype=pred_boxes.dtype)
    matched_pred_cls = []
    matched_gt_cls = []
    matched_pred_box = []
    matched_gt_box = []
    matched_pred_dist = []
    matched_gt_dist = []

    for i in range(B):
        idx = indices_list[i]
        if idx.size(1) == 0:
            continue
        valid_gt = (gt_boxes[i, :, 2] > 0)
        gt_boxes_valid = gt_boxes[i][valid_gt]
        gt_labels_valid = gt_labels[i][valid_gt]
        num_gt_valid = gt_labels_valid.size(0)
        N_i = pred_boxes.size(1)
        # linear_sum_assignment(cost [num_pred, num_gt]) returns (row_ind=pred_idx, col_ind=gt_idx)
        if idx[0].max().item() < N_i and idx[1].max().item() < num_gt_valid:
            pred_idx, gt_idx = idx[0], idx[1]
        elif idx[1].max().item() < N_i and idx[0].max().item() < num_gt_valid:
            pred_idx, gt_idx = idx[1], idx[0]
        else:
            continue
        target_objectness[i, pred_idx] = 1.0

        gt_dist_valid = targets["distance"][i][valid_gt].long().clamp(0, 2)

        matched_pred_cls.append(pred_logits[i][pred_idx])
        matched_gt_cls.append(gt_labels_valid[gt_idx])
        matched_pred_box.append(pred_boxes[i][pred_idx])
        matched_gt_box.append(gt_boxes_valid[gt_idx])
        if "distance_zones" in outputs and outputs["distance_zones"] is not None:
            matched_pred_dist.append(outputs["distance_zones"][i][pred_idx])
            matched_gt_dist.append(gt_dist_valid[gt_idx])

    aligned_pred = {
        "objectness": outputs["objectness"],
        "classification": torch.cat(matched_pred_cls, dim=0) if matched_pred_cls else torch.empty(0, pred_logits.size(-1), device=device),
        "box": torch.cat(matched_pred_box, dim=0) if matched_pred_box else torch.empty(0, 4, device=device),
        "distance": torch.cat(matched_pred_dist, dim=0) if matched_pred_dist else torch.empty(0, 3, device=device),
        "urgency_scores": outputs.get("urgency_scores"),
    }
    aligned_target = {
        "objectness": target_objectness,
        "labels": torch.cat(matched_gt_cls, dim=0) if matched_gt_cls else torch.empty(0, dtype=torch.long, device=device),
        "boxes": torch.cat(matched_gt_box, dim=0) if matched_gt_box else torch.empty(0, 4, device=device),
        "distance": torch.cat(matched_gt_dist, dim=0) if matched_gt_dist else torch.empty(0, dtype=torch.long, device=device),
        "urgency": targets.get("urgency"),
    }
    return aligned_pred, aligned_target
