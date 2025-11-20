"""Loss functions for object detection: focal loss, IoU loss, target assignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FocalLoss(nn.Module):
    """Focal loss for class imbalance - downweights easy examples, focuses on hard ones."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss: alpha * (1-pt)^gamma * ce_loss. Downweights high-confidence (easy) examples."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between box pairs. Assumes center format (cx, cy, w, h) normalized [0,1]."""
    # Convert to corner format for intersection calc
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    inter_x1 = torch.max(boxes1_x1, boxes2_x1)
    inter_y1 = torch.max(boxes1_y1, boxes2_y1)
    inter_x2 = torch.min(boxes1_x2, boxes2_x2)
    inter_y2 = torch.min(boxes1_y2, boxes2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = boxes1[:, 2] * boxes1[:, 3] + boxes2[:, 2] * boxes2[:, 3] - inter_area
    iou = inter_area / (union_area + 1e-8)
    return iou


def assign_targets_to_anchors(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    anchor_points: torch.Tensor,
    num_classes: int,
    pos_iou_thresh: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign GT to anchors using center-in-box matching. TODO: Switch to IoU-based for better accuracy."""
    N = anchor_points.shape[0]
    M = gt_boxes.shape[0]
    
    assigned_labels = torch.zeros(N, dtype=torch.long, device=gt_boxes.device)
    assigned_boxes = torch.zeros(N, 4, device=gt_boxes.device)
    assigned_mask = torch.zeros(N, dtype=torch.float32, device=gt_boxes.device)
    
    if M == 0:
        return assigned_labels, assigned_boxes, assigned_mask
    
    for i in range(M):
        gt_box = gt_boxes[i]
        gt_label = gt_labels[i]
        
        x1 = gt_box[0] - gt_box[2] / 2
        y1 = gt_box[1] - gt_box[3] / 2
        x2 = gt_box[0] + gt_box[2] / 2
        y2 = gt_box[1] + gt_box[3] / 2
        
        inside_mask = (anchor_points[:, 0] >= x1) & (anchor_points[:, 0] <= x2) & \
                      (anchor_points[:, 1] >= y1) & (anchor_points[:, 1] <= y2)
        
        if inside_mask.sum() > 0:
            assigned_labels[inside_mask] = gt_label + 1  # +1: 0 is background
            assigned_boxes[inside_mask] = gt_box
            assigned_mask[inside_mask] = 1.0
    
    return assigned_labels, assigned_boxes, assigned_mask


class MaxSightLoss(nn.Module):
    """Multi-task loss: focal (cls) + IoU (box) + BCE (obj) + optional urgency/distance."""
    
    def __init__(
        self,
        num_classes: int = 48,
        classification_weight: float = 1.0,
        localization_weight: float = 5.0,
        objectness_weight: float = 1.0,
        urgency_weight: float = 0.5,
        distance_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.objectness_weight = objectness_weight
        self.urgency_weight = urgency_weight
        self.distance_weight = distance_weight
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Multi-task loss: cls + box + obj + urgency + distance. Weights tuned for balance."""
        batch_size = outputs['classifications'].size(0)
        num_locations = outputs['num_locations']
        device = outputs['classifications'].device
        
        # Generate anchor grid (assumes square - fragile if multi-scale)
        grid_size = int(num_locations ** 0.5)
        y_coords = torch.linspace(0, 1, grid_size, device=device)
        x_coords = torch.linspace(0, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        anchor_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        num_positives = 0
        for b in range(batch_size):
            cls_logits = outputs['classifications'][b]
            box_preds = outputs['boxes'][b]
            obj_preds = outputs['objectness'][b]
            
            gt_labels = targets['labels'][b]
            gt_boxes = targets['boxes'][b]
            num_objs = targets.get('num_objects', torch.tensor([len(gt_labels)]))[b]
            
            gt_labels = gt_labels[:num_objs]
            gt_boxes = gt_boxes[:num_objs]
            
            assigned_labels, assigned_boxes, assigned_mask = assign_targets_to_anchors(
                gt_boxes, gt_labels, anchor_points, self.num_classes
            )
            
            if assigned_mask.sum() > 0:
                pos_mask = assigned_mask > 0
                cls_targets = torch.clamp(assigned_labels[pos_mask] - 1, 0, self.num_classes - 1)
                cls_loss = self.focal_loss(cls_logits[pos_mask], cls_targets)
                total_cls_loss += cls_loss
                
                iou = compute_iou(box_preds[pos_mask], assigned_boxes[pos_mask])
                box_loss = (1 - iou).mean()
                total_box_loss += box_loss
                num_positives += pos_mask.sum().item()
            
            obj_loss = self.bce_loss(obj_preds, assigned_mask)
            total_obj_loss += obj_loss
        
        # Average over batch
        if num_positives > 0:
            cls_loss = total_cls_loss / batch_size
            box_loss = total_box_loss / batch_size
        else:
            cls_loss = torch.tensor(0.0, device=device)
            box_loss = torch.tensor(0.0, device=device)
        
        obj_loss = total_obj_loss / batch_size
        
        # Urgency loss (scene-level)
        if 'urgency' in targets:
            urgency_loss = self.ce_loss(outputs['urgency_scores'], targets['urgency'])
        else:
            urgency_loss = torch.tensor(0.0, device=device)
        
        # Distance loss (disabled - no labels in dataset)
        distance_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.localization_weight * box_loss +
            self.objectness_weight * obj_loss +
            self.urgency_weight * urgency_loss +
            self.distance_weight * distance_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'localization_loss': box_loss,
            'objectness_loss': obj_loss,
            'urgency_loss': urgency_loss,
            'distance_loss': distance_loss,
            'num_positives': torch.tensor(num_positives, device=device, dtype=torch.long)
        }


if __name__ == "__main__":
    print("Testing MaxSight Loss...")
    
    # Dummy data
    batch_size = 2
    num_locations = 196  # 14x14
    num_classes = 48
    
    outputs = {
        'classifications': torch.randn(batch_size, num_locations, num_classes),
        'boxes': torch.rand(batch_size, num_locations, 4) * 0.5 + 0.25,
        'objectness': torch.rand(batch_size, num_locations),
        'urgency_scores': torch.randn(batch_size, 4),
        'distance_zones': torch.randn(batch_size, num_locations, 3),
        'num_locations': num_locations
    }
    
    targets = {
        'labels': torch.randint(0, num_classes, (batch_size, 5)),
        'boxes': torch.rand(batch_size, 5, 4) * 0.5 + 0.25,
        'urgency': torch.randint(0, 4, (batch_size,)),
        'distance': torch.randint(0, 3, (batch_size, 5)),
        'num_objects': torch.tensor([3, 2])
    }
    
    criterion = MaxSightLoss(num_classes=num_classes)
    losses = criterion(outputs, targets)
    
    print("\nLoss values:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nLoss computation successful!")
