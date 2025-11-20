"""COCO-style mAP calculator for object detection. Tracks per-class/size/lighting metrics."""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict
import numpy as np


def compute_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Batch IoU calc using broadcasting (faster than loops). Assumes center format."""
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return torch.zeros(pred_boxes.shape[0], gt_boxes.shape[0], device=pred_boxes.device)
    
    pred_boxes = pred_boxes.unsqueeze(1)
    gt_boxes = gt_boxes.unsqueeze(0)
    
    # Convert to corners for intersection calc
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_y1 = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
    gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
    gt_y2 = gt_boxes[..., 1] + gt_boxes[..., 3] / 2
    
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - inter_area
    return (inter_area / (union_area + 1e-9)).squeeze()


class DetectionMetrics:
    """COCO-style mAP calculator. Tracks per-class/size/lighting metrics."""
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = [0.5], device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.device = device or torch.device('cpu')
        
        self.reset()
    
    def reset(self, device: Optional[torch.device] = None):
        """Reset accumulators. Pass device to ensure tensors on correct device (CPU/GPU)."""
        if device is not None:
            self.device = device
        
        self.class_tp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_fp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_fn = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_predictions = defaultdict(list)  # Store (score, is_tp, area) for AP calc
        self.class_gt_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.lighting_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.size_metrics = {'small': {'tp': 0, 'fp': 0, 'fn': 0}, 'medium': {'tp': 0, 'fp': 0, 'fn': 0}, 'large': {'tp': 0, 'fp': 0, 'fn': 0}}
    
    def _get_size_category(self, box: torch.Tensor) -> str:
        """COCO size categories. Assumes 224x224 images (normalized thresholds)."""
        area = box[2] * box[3]
        if area < 0.02:
            return 'small'
        elif area < 0.18:
            return 'medium'
        return 'large'
    
    def update(self, pred_boxes: torch.Tensor, pred_labels: torch.Tensor, pred_scores: torch.Tensor,
               gt_boxes: torch.Tensor, gt_labels: torch.Tensor, lighting: Optional[str] = None,
               iou_threshold: float = 0.5) -> None:
        """Update metrics with predictions/GT. Uses greedy matching by confidence."""
        # Move to device (avoids mismatch errors)
        pred_boxes = pred_boxes.to(self.device)
        pred_labels = pred_labels.to(self.device)
        pred_scores = pred_scores.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        gt_labels = gt_labels.to(self.device)
        
        # Count GT objects per class (for recall calc)
        for gt_label in gt_labels:
            class_idx = int(gt_label.item())
            if 0 <= class_idx < self.num_classes:
                self.class_gt_counts[class_idx] += 1
        
        if len(pred_boxes) == 0:
            for gt_label, gt_box in zip(gt_labels, gt_boxes):
                class_idx = int(gt_label.item())
                if 0 <= class_idx < self.num_classes:
                    self.class_fn[class_idx] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fn'] += 1
                    
                    # Update size metrics
                    size_cat = self._get_size_category(gt_box)
                    self.size_metrics[size_cat]['fn'] += 1
            return
        
        if len(gt_boxes) == 0:
            for pred_label, pred_score, pred_box in zip(pred_labels, pred_scores, pred_boxes):
                class_idx = int(pred_label.item())
                if 0 <= class_idx < self.num_classes:
                    self.class_fp[class_idx] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fp'] += 1
                    
                    # Store as FP for AP calculation
                    box_area = (pred_box[2] * pred_box[3]).item()
                    self.class_predictions[class_idx].append(
                        (pred_score.item(), False, box_area)
                    )
                    
                    # Update size metrics
                    size_cat = self._get_size_category(pred_box)
                    self.size_metrics[size_cat]['fp'] += 1
            return
        
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        # Greedy matching: sort by confidence, match to best available GT of same class
        matched_gt = set()
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for sorted_idx in sorted_indices:
            pred_idx = int(sorted_idx.item())
            pred_class = int(pred_labels[pred_idx].item())
            pred_score = pred_scores[pred_idx].item()
            pred_box = pred_boxes[pred_idx]
            box_area = (pred_box[2] * pred_box[3]).item()
            
            if not (0 <= pred_class < self.num_classes):
                continue
            
            # Find best matching GT of same class
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt or int(gt_labels[gt_idx].item()) != pred_class:
                    continue
                iou = iou_matrix[pred_idx, gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            is_tp = best_iou >= iou_threshold and best_gt_idx is not None
            
            if is_tp:
                self.class_tp[pred_class] += 1
                matched_gt.add(best_gt_idx)
                if lighting:
                    self.lighting_metrics[lighting]['tp'] += 1
                self.size_metrics[self._get_size_category(pred_box)]['tp'] += 1
            else:
                self.class_fp[pred_class] += 1
                if lighting:
                    self.lighting_metrics[lighting]['fp'] += 1
                self.size_metrics[self._get_size_category(pred_box)]['fp'] += 1
            
            # Store for AP calc (sorted by confidence)
            self.class_predictions[pred_class].append((pred_score, is_tp, box_area))
        
        # False negatives: unmatched GT boxes
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                gt_class = int(gt_labels[gt_idx].item())
                gt_box = gt_boxes[gt_idx]
                if 0 <= gt_class < self.num_classes:
                    self.class_fn[gt_class] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fn'] += 1
                    self.size_metrics[self._get_size_category(gt_box)]['fn'] += 1
    
    def compute_precision(self, class_idx: Optional[int] = None) -> float:
        """Precision: TP / (TP + FP). None = overall, int = per-class."""
        if class_idx is None:
            total_tp = self.class_tp.sum().item()
            total_fp = self.class_fp.sum().item()
        else:
            if not (0 <= class_idx < self.num_classes):
                return 0.0
            total_tp = self.class_tp[class_idx].item()
            total_fp = self.class_fp[class_idx].item()
        denominator = total_tp + total_fp
        return total_tp / denominator if denominator > 0 else 0.0
    
    def compute_recall(self, class_idx: Optional[int] = None) -> float:
        """Compute recall: TP / (TP + FN)."""
        if class_idx is None:
            total_tp = self.class_tp.sum().item()
            total_fn = self.class_fn.sum().item()
        else:
            if not (0 <= class_idx < self.num_classes):
                return 0.0
            total_tp = self.class_tp[class_idx].item()
            total_fn = self.class_fn[class_idx].item()
        
        denominator = total_tp + total_fn
        return total_tp / denominator if denominator > 0 else 0.0
    
    def compute_f1(self, class_idx: Optional[int] = None) -> float:
        """Compute F1 score: 2 * (P * R) / (P + R)."""
        precision = self.compute_precision(class_idx)
        recall = self.compute_recall(class_idx)
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def compute_ap(self, class_idx: int, iou_threshold: float = 0.5) -> float:
        """AP using 11-point interpolation (COCO standard). Area under PR curve."""
        predictions = self.class_predictions.get(class_idx, [])
        total_gt = self.class_gt_counts[class_idx].item()
        
        if len(predictions) == 0 or total_gt == 0:
            return 0.0
        
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for confidence, is_tp, box_area in predictions:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt
            precisions.append(precision)
            recalls.append(recall)
        
        # 11-point interpolation
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            max_precision = 0.0
            for r, p in zip(recalls, precisions):
                if r >= float(recall_threshold):
                    max_precision = max(max_precision, p)
            ap += max_precision
        
        return ap / 11.0
    
    def compute_map(self, iou_threshold: float = 0.5) -> float:
        """Mean AP across all classes."""
        ap_scores = []
        
        for class_idx in range(self.num_classes):
            ap = self.compute_ap(class_idx, iou_threshold)
            ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    
    def compute_map_coco(self) -> Dict[str, float]:
        """COCO-style mAP@[0.5:0.95] - averages mAP across IoU thresholds 0.5 to 0.95."""
        # Generate IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95 (step 0.05)
        # Also include 0.75 separately because it's commonly reported
        thresholds = [0.5, 0.75] + [float(t) for t in np.arange(0.5, 1.0, 0.05)]
        
        results = {}
        aps_all = []  # Store all APs for averaging
        
        # Compute mAP at each threshold
        for threshold in thresholds:
            threshold_float = float(threshold)  # Ensure it's a Python float (numpy types cause issues)
            ap = self.compute_map(threshold_float)
            aps_all.append(ap)
            
            # Store specific thresholds that are commonly reported
            if abs(threshold_float - 0.5) < 1e-6:  # Floating point comparison
                results['mAP@0.5'] = ap
            elif abs(threshold_float - 0.75) < 1e-6:
                results['mAP@0.75'] = ap
        
        # Average AP across all thresholds - this is the COCO standard
        results['mAP@[0.5:0.95]'] = sum(aps_all) / len(aps_all)
        
        return results
    
    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get precision, recall, F1, AP for each class."""
        results = {}
        
        for class_idx in range(self.num_classes):
            results[class_idx] = {
                'precision': self.compute_precision(class_idx),
                'recall': self.compute_recall(class_idx),
                'f1': self.compute_f1(class_idx),
                'ap': self.compute_ap(class_idx)
            }
        
        return results
    
    def get_lighting_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get precision, recall, F1 for each lighting condition."""
        results = {}
        
        for lighting, metrics in self.lighting_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            results[lighting] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
    
    def get_size_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics by object size (COCO-style)."""
        results = {}
        
        for size_cat, metrics in self.size_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            results[size_cat] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
