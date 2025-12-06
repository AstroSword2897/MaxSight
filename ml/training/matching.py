"""
Hungarian Matching for Multi-Object Detection

Matches predicted boxes to ground truth using optimal bipartite matching.

Based on DETR's approach with combined classification + bbox + GIoU costs.

"""


import torch
from typing import Tuple, List


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
    
    # Classification cost: negative log-likelihood
    # We want high confidence on the correct class, so we use -log(p)
    probs = torch.softmax(pred_logits, dim=-1)
    class_cost = -probs[:, gt_labels].log()  # [num_pred, num_gt]
    
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
    # Compute cost matrix [num_pred, num_gt]
    cost = compute_matching_cost(
        pred_boxes, pred_logits, gt_boxes, gt_labels,
        lambda_class, lambda_bbox, lambda_giou
    )
    
    if use_hungarian:
        # Use proper Hungarian algorithm for globally optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            cost_np = cost.cpu().numpy()
            pred_indices, gt_indices = linear_sum_assignment(cost_np)
            
            pred_idx = torch.tensor(pred_indices, dtype=torch.long, device=pred_boxes.device)
            gt_idx = torch.tensor(gt_indices, dtype=torch.long, device=pred_boxes.device)
            matched_costs = cost[pred_idx, gt_idx]
            
            indices = torch.stack([pred_idx, gt_idx])
            return indices, matched_costs
        except ImportError:
            print("Warning: scipy not available, falling back to greedy matching")
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
        # Skip samples with no ground truth
        valid_gt = (gt_boxes[i, :, 2] > 0)
        
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
        
        # Find matches
        indices, costs = match_predictions_to_gt(
            pred_boxes[i],
            pred_logits[i],
            gt_boxes_valid,
            gt_labels_valid,
            lambda_class, lambda_bbox, lambda_giou
        )
        
        indices_list.append(indices)
        costs_list.append(costs)
    
    return indices_list, costs_list
