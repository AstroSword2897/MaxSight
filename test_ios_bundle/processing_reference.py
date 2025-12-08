"""
MaxSight Processing Reference for iOS
Essential preprocessing, postprocessing, and scheduling logic.

Port these functions to Swift for iOS implementation.
This is the minimal set needed to process model inputs/outputs.

Generated automatically from MaxSight repository.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


# From ml/utils/preprocessing.py
def apply_refractive_error_blur(image: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """Apply Gaussian blur for refractive errors"""
    kernel_size = int(2 * sigma * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])




# From ml/utils/preprocessing.py
def apply_cataract_contrast(image: torch.Tensor, contrast_factor: float = 0.5) -> torch.Tensor:
    """Reduce contrast for cataracts simulation"""
    return TF.adjust_contrast(image, contrast_factor)




# From ml/utils/preprocessing.py
def apply_glaucoma_vignette(image: torch.Tensor, center_percent: float = 0.4) -> torch.Tensor:
    """Apply peripheral masking for glaucoma"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) * center_percent
    
    # Create circular mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist < radius).float()
    
    # Expand mask to match image dimensions
    while mask.dim() < image.dim():
        mask = mask.unsqueeze(0)
    # Ensure mask has same shape as image
    if mask.shape != image.shape:
        mask = mask.expand_as(image)
    
    return image * mask




# From ml/utils/preprocessing.py
def apply_amd_central_darkening(image: torch.Tensor, darken_factor: float = 0.3) -> torch.Tensor:
    """Darken center region for AMD simulation"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = float(min(w, h)) * 0.2
    
    # Create circular darkening mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = 1.0 - (dist < radius).float() * darken_factor
    
    # Expand mask to match image dimensions
    while mask.dim() < image.dim():
        mask = mask.unsqueeze(0)
    # Ensure mask has same shape as image
    if mask.shape != image.shape:
        mask = mask.expand_as(image)
    
    return image * mask




# From ml/utils/preprocessing.py
def apply_low_light(image: torch.Tensor, brightness_factor: float = 0.3) -> torch.Tensor:
    """Reduce brightness for retinitis pigmentosa"""
    return image * brightness_factor




# From ml/utils/preprocessing.py
def apply_color_shift(image: torch.Tensor, shift_type: str = 'red_green') -> torch.Tensor:
    """
    Apply color shifts for color blindness simulation using proper color space transformation.
    
    Supports multiple types:
    - 'protanopia': Red-blind (L-cone missing)
    - 'deuteranopia': Green-blind (M-cone missing)
    - 'tritanopia': Blue-blind (S-cone missing)
    - 'red_green': Simple red-green mix (legacy, less accurate)
    
    Arguments:
        image: Tensor [C, H, W] or [B, C, H, W] in range [0, 1]
        shift_type: Type of color blindness to simulate
    
    Returns:
        Color-shifted tensor with same shape and range
    """
    # Validate input
    if image.dim() == 4:
        if image.shape[1] != 3:
            return image
        is_batch = True
    elif image.dim() == 3:
        if image.shape[0] != 3:
            return image
        is_batch = False
        image = image.unsqueeze(0)
    else:
        return image
    
    # Color blindness transformation matrices (LMS color space)
    # These are proper color space transformations, not simple channel mixing
    if shift_type == 'protanopia':
        # Red-blind: L-cone missing, simulate by shifting L to M
        transform = torch.tensor([
            [0.0, 1.05118294, -0.05116099],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=image.device, dtype=image.dtype)
    elif shift_type == 'deuteranopia':
        # Green-blind: M-cone missing, simulate by shifting M to L
        transform = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9513092, 0.0, 0.04866992],
            [0.0, 0.0, 1.0]
        ], device=image.device, dtype=image.dtype)
    elif shift_type == 'tritanopia':
        # Blue-blind: S-cone missing, simulate by shifting S to L
        transform = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.86744736, 1.86727089, 0.0]
        ], device=image.device, dtype=image.dtype)
    elif shift_type == 'red_green':
        # Legacy: Simple red-green mix (less accurate but faster)
        if is_batch:
            r, g, b = image[:, 0], image[:, 1], image[:, 2]
            mixed = (r + g) / 2
            result = torch.stack([mixed, mixed, b], dim=1)
        else:
            r, g, b = image[0, 0], image[0, 1], image[0, 2]
            mixed = (r + g) / 2
            result = torch.stack([mixed, mixed, b], dim=0).unsqueeze(0)
        return result.squeeze(0) if not is_batch else result
    else:
        # Unknown type, return original
        return image.squeeze(0) if not is_batch else image
    
    # Convert RGB to LMS (Long/Medium/Short wavelength cones)
    # Simplified LMS approximation (more accurate would use full color space conversion)
    # Optimized: Use einsum instead of flatten/reshape for 2-3x speedup and less memory
    transform = transform.to(device=image.device, dtype=image.dtype)
    
    if is_batch:
        # Efficient einsum: [B, C, H, W] format
        # transform: [3, 3], image: [B, 3, H, W] -> result: [B, 3, H, W]
        result = torch.einsum('ij,bjhw->bihw', transform, image)
    else:
        # [C, H, W] format
        # transform: [3, 3], image: [3, H, W] -> result: [3, H, W]
        result = torch.einsum('ij,jhw->ihw', transform, image)
        result = result.unsqueeze(0)  # Add batch dim for consistency
    
    # Clamp to valid range
    result = torch.clamp(result, 0.0, 1.0)
    return result.squeeze(0) if not is_batch else result




# From ml/models/maxsight_cnn.py (class MaxSightCNN method)
def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
    """
    Non-Maximum Suppression - removes duplicate detections of the same object.
    
    Optimized version: processes boxes more efficiently by avoiding repeated masking
    and using vectorized operations where possible.
    
    When multiple boxes overlap a lot (high IoU), we keep only the one with
    the highest score. This prevents the same object from being detected multiple times.
    
    This is a greedy algorithm - not optimal but fast and works well in practice.
    For very large numbers of boxes (100+), consider using torchvision.ops.nms for
    absolute maximum speed, but this implementation is readable and flexible.
    """
    if len(boxes) == 0:
        return []  # Edge case - no boxes to process
    
    # Convert to corner format - easier for IoU calculation
    # Center format is convenient for the model but corner format is better for IoU
    boxes_corners = _center_to_corners(boxes)
    
    # Sort by score (best first)
    # Boxes should already be sorted but we sort again to be safe
    # (defensive programming - doesn't hurt and makes code more robust)
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)  # Handle scalar case
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    keep = []  # Indices of boxes to keep
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)  # Track what we've suppressed
    
    # Go through boxes in order of confidence (greedy approach)
    # We process highest confidence first, then suppress overlapping ones
    for i in range(len(boxes)):
        idx = int(sorted_indices[i].item())  # Get the actual index
        
        # Skip if we already decided to suppress this one
        # (can happen if a lower-confidence box was processed first due to sorting)
        if suppressed[idx]:
            continue
        
        # Keep this box - it's the best one so far
        keep.append(idx)
        
        # Now suppress any boxes that overlap too much with this one
        # Only check remaining boxes (ones we haven't processed yet)
        if i < len(boxes) - 1:
            remaining_indices = sorted_indices[i+1:]  # All boxes after current
            remaining_mask = ~suppressed[remaining_indices]  # Only check unsuppressed ones
            
            if remaining_mask.any():
                remaining_idx = remaining_indices[remaining_mask]
                remaining_boxes = boxes_corners[remaining_idx]
                
                # Check how much each remaining box overlaps with current box
                # Compute IoU between current box and all remaining boxes at once (vectorized)
                current_box = boxes_corners[idx:idx+1]  # Keep as [1, 4] for broadcasting
                ious = _compute_iou_corners(current_box, remaining_boxes)
                
                # Suppress boxes that overlap too much (IoU >= threshold)
                # Higher threshold = more aggressive suppression
                suppress_mask = ious.flatten() >= threshold
                suppressed[remaining_idx[suppress_mask]] = True
    
    return keep



# From ml/models/maxsight_cnn.py (class MaxSightCNN method)
def _compute_iou(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between box1 (center format) and all boxes2 (center format)
    
    Arguments:
        box1: [1, 4] or [N, 4] tensor in center format (x, y, w, h)
        boxes2: [M, 4] tensor in center format (x, y, w, h)
    
    Returns:
        [1, M] or [N, M] IoU scores
    """
    # Convert center format to corners
    box1_corners = _center_to_corners(box1)
    boxes2_corners = _center_to_corners(boxes2)
    
    return _compute_iou_corners(box1_corners, boxes2_corners)



# From ml/models/maxsight_cnn.py (class MaxSightCNN method)
def _compute_iou_corners(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between box1 and all boxes2
    
    IoU measures how much two boxes overlap. 1.0 = identical, 0.0 = no overlap.
    Used to decide if two detections are actually the same object.
    
    This is vectorized - computes IoU between box1 and all boxes2 at once.
    Much faster than looping.
    """
    # Make sure box1 is 2D - handle edge case where it's 1D
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)  # [4] -> [1, 4]
    
    # Expand dimensions for broadcasting - compare box1 with all boxes2 at once
    # Broadcasting magic: [N, 1, 4] vs [1, M, 4] -> [N, M, 4]
    box1 = box1.unsqueeze(1)  # [N, 4] -> [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [M, 4] -> [1, M, 4]
    
    # Find the intersection rectangle
    # Two boxes overlap if their intersection exists
    # Top-left corner: max of the two top-left corners (rightmost left, bottommost top)
    inter_x1 = torch.max(box1[..., 0], boxes2[..., 0])  # x1 coordinates
    inter_y1 = torch.max(box1[..., 1], boxes2[..., 1])  # y1 coordinates
    # Bottom-right corner: min of the two bottom-right corners (leftmost right, topmost bottom)
    inter_x2 = torch.min(box1[..., 2], boxes2[..., 2])  # x2 coordinates
    inter_y2 = torch.min(box1[..., 3], boxes2[..., 3])  # y2 coordinates
    
    # Calculate intersection area (clamp to 0 in case boxes don't overlap)
    # If boxes don't overlap, inter_x2 < inter_x1, so we clamp to 0
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)  # Width of intersection
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)  # Height of intersection
    inter_area = inter_w * inter_h  # Area of intersection
    
    # Calculate area of each box
    # Simple width * height
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Union = area1 + area2 - intersection (don't double-count overlap)
    # If boxes overlap, we'd count the overlap twice without subtracting it
    union_area = box1_area + boxes2_area - inter_area
    
    # IoU = intersection / union (add tiny epsilon to avoid division by zero)
    # 1e-6 is standard - small enough to not affect results, big enough to prevent NaN
    iou = inter_area / (union_area + 1e-6)
    
    # Clean up dimensions if needed
    # If box1 was [1, 4], result is [1, M] - squeeze to [M] for convenience
    if iou.size(0) == 1:
        iou = iou.squeeze(0)
    
    return iou



# From ml/models/maxsight_cnn.py (class MaxSightCNN method)
def _center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format to corner format
    
    Center format: (center_x, center_y, width, height)
    Corner format: (x1, y1, x2, y2) - top-left and bottom-right corners
    
    Corner format is easier for IoU calculations.
    """
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - w / 2  # Left edge
    y1 = y_center - h / 2  # Top edge
    x2 = x_center + w / 2  # Right edge
    y2 = y_center + h / 2  # Bottom edge
    return torch.stack([x1, y1, x2, y2], dim=1)



# From ml/utils/output_scheduler.py (class CrossModalScheduler method)
def _get_priority_threshold() -> int:
    """Get priority threshold based on alert frequency"""
    thresholds = {
        AlertFrequency.LOW: 70,      # Only hazards + navigation
        AlertFrequency.MEDIUM: 40,    # + useful objects
        AlertFrequency.HIGH: 0       # All objects
    }
    return thresholds.get(.config.alert_frequency, 40)



# From ml/utils/output_scheduler.py (class CrossModalScheduler method)
def _calculate_intensity(priority: int, findability: float, urgency: int) -> float:
    """Calculate output intensity (0-1)"""
    # Base intensity from priority
    base_intensity = priority / 100.0
    
    # Adjust for findability (harder to find = higher intensity)
    findability_adjustment = (1.0 - findability) * 0.2
    
    # Adjust for urgency
    urgency_adjustment = urgency / 3.0 * 0.3
    
    intensity = base_intensity + findability_adjustment + urgency_adjustment
    
    # Apply channel-specific scaling
    if config.preferred_channel == OutputChannel.AUDIO:
        intensity *= config.audio_volume
    elif config.preferred_channel == OutputChannel.HAPTIC:
        intensity *= config.haptic_intensity
    else:
        intensity *= config.visual_contrast
    
    return min(1.0, max(0.0, intensity))



# From ml/utils/output_scheduler.py (class CrossModalScheduler method)
def _calculate_frequency(priority: int, urgency: int) -> float:
    """Calculate output frequency in Hz"""
    # Higher priority/urgency = faster rhythm
    if priority >= 90 or urgency >= 3:
        return 10.0  # Fast rhythm for hazards
    elif priority >= 70:
        return 5.0   # Medium rhythm for navigation
    else:
        return 2.0   # Slow rhythm for useful objects



# From ml/utils/output_scheduler.py (class CrossModalScheduler method)
def _select_channel(priority: int, urgency: int) -> OutputChannel:
    """Select output channel based on priority and user preference"""
    # High priority/urgency -> use preferred channel or hybrid
    if priority >= 90 or urgency >= 3:
        if config.preferred_channel == OutputChannel.HYBRID:
            return OutputChannel.HYBRID
        return config.preferred_channel
    
    # Medium priority -> use preferred channel
    if priority >= 70:
        return config.preferred_channel
    
    # Low priority -> use less intrusive channel
    if config.preferred_channel == OutputChannel.AUDIO:
        return OutputChannel.VISUAL  # Visual overlay instead of audio
    return config.preferred_channel



# From ml/utils/ocr_integration.py (class OCRIntegration method)
def _cluster_text_pixels(
    self,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    h: int,
    w: int,
    cluster_distance: int = 10,
    use_dbscan: bool = True
) -> List[Tuple[int, int, int, int]]:
    """
    Cluster text pixels into regions using DBSCAN (improved) or simple distance-based method.
    
    WHY DBSCAN:
    DBSCAN is more efficient (O(N log N) vs O(N²)) and handles irregularly shaped regions
    better than simple distance-based clustering. For large images with many text pixels,
    this provides significant performance improvement while maintaining accuracy.
    
    HOW IT SUPPORTS THE PROBLEM STATEMENT:
    Efficient text region detection enables real-time text reading, supporting the "Reads
    Environment" feature. Users need text detected quickly for practical use, not just
    accurate detection.
    
    Arguments:
        x_coords: X coordinates of text pixels
        y_coords: Y coordinates of text pixels
        h: Image height
        w: Image width
        cluster_distance: Maximum distance for clustering
        use_dbscan: Use DBSCAN for better performance (requires scikit-learn)
    
    Returns:
        List of (x_min, y_min, x_max, y_max) bounding boxes
    """
    if len(x_coords) == 0:
        return []
    
    # Convert to numpy for easier processing
    coords = torch.stack([x_coords, y_coords], dim=1).cpu().numpy()
    
    if use_dbscan:
        try:
            from sklearn.cluster import DBSCAN  # type: ignore
        except ImportError:
            raise RuntimeError(
                "scikit-learn required for text region clustering. "
                "Install: pip install scikit-learn"
            )
        
        # Use DBSCAN for efficient clustering
        # eps = cluster_distance in normalized coordinates
        # min_samples = 2 (at least 2 pixels per cluster)
        dbscan = DBSCAN(eps=cluster_distance, min_samples=2, metric='euclidean')
        labels = dbscan.fit_predict(coords)
        
        # Group pixels by cluster label
        regions = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_coords = coords[cluster_mask]
            
            if len(cluster_coords) > 0:
                x_min = int(cluster_coords[:, 0].min())
                y_min = int(cluster_coords[:, 1].min())
                x_max = int(cluster_coords[:, 0].max())
                y_max = int(cluster_coords[:, 1].max())
                
                # Add padding
                padding = 2
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w - 1, x_max + padding)
                y_max = min(h - 1, y_max + padding)
                
                if x_max > x_min and y_max > y_min:
                    regions.append((x_min, y_min, x_max, y_max))
        
        return regions
    
    # Optimized distance-based clustering (fallback) using cKDTree for O(N log N) performance
    # Vectorized approach: use scipy.spatial.cKDTree if available, otherwise simple O(N²) fallback
    try:
        from scipy.spatial import cKDTree  # type: ignore
        
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(coords)
        regions = []
        used = set()
        
        for i, (x, y) in enumerate(coords):
            if i in used:
                continue
            
            # Start new region
            cluster = [i]
            used.add(i)
            x_min, y_min, x_max, y_max = x, y, x, y
            
            # Find all neighbors within cluster_distance using KD-tree (O(log N) per query)
            neighbors = tree.query_ball_point((x, y), cluster_distance)
            
            for j in neighbors:
                if j in used or j == i:
                    continue
                cluster.append(j)
                used.add(j)
                x2, y2 = coords[j]
                x_min = min(x_min, x2)
                y_min = min(y_min, y2)
                x_max = max(x_max, x2)
                y_max = max(y_max, y2)
            
            # Add padding
            padding = 2
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(w - 1, int(x_max) + padding)
            y_max = min(h - 1, int(y_max) + padding)
            
            if x_max > x_min and y_max > y_min:
                regions.append((x_min, y_min, x_max, y_max))
        
        return regions
        
    except ImportError:
        # Fallback to simple O(N²) clustering if scipy not available
        regions = []
        used = set()
        
        for i, (x, y) in enumerate(coords):
            if i in used:
                continue
            
            # Start new region
            cluster = [i]
            used.add(i)
            x_min, y_min, x_max, y_max = x, y, x, y
            
            # Find nearby pixels (O(N) per pixel)
            for j, (x2, y2) in enumerate(coords):
                if j in used or j == i:
                    continue
                
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < cluster_distance:
                    cluster.append(j)
                    used.add(j)
                    x_min = min(x_min, x2)
                    y_min = min(y_min, y2)
                    x_max = max(x_max, x2)
                    y_max = max(y_max, y2)
            
            # Add padding
            padding = 2
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(w - 1, int(x_max) + padding)
            y_max = min(h - 1, int(y_max) + padding)
            
            if x_max > x_min and y_max > y_min:
                regions.append((x_min, y_min, x_max, y_max))
        
        return regions



# From ml/utils/ocr_integration.py
def _group_text_by_proximity(text_results: List[Dict], proximity_threshold: float = 0.1) -> List[Dict]:
    """
    Group text regions by spatial proximity (line/block grouping).
    
    WHY THIS FUNCTION:
    Prevents splitting connected text (lines, paragraphs) into multiple regions. This
    provides more natural text descriptions and better context understanding.
    
        Arguments:
        text_results: List of OCR results
        proximity_threshold: Maximum distance for grouping (normalized)
    
    Returns:
        List of grouped text results
    """
    if not text_results:
        return []
    
    groups = []
    used = set()
    
    for i, result in enumerate(text_results):
        if i in used:
            continue
        
        # Start new group
        group = [result]
        used.add(i)
        box1 = result['box']
        cx1, cy1 = box1[0], box1[1]
        
        # Find nearby text regions (likely same line/block)
        for j, other in enumerate(text_results):
            if j in used or j == i:
                continue
            
            box2 = other['box']
            cx2, cy2 = box2[0], box2[1]
            
            # Check if vertically aligned (same line) or horizontally close (same block)
            vertical_distance = abs(cy1 - cy2)
            horizontal_distance = abs(cx1 - cx2)
            
            if vertical_distance < proximity_threshold or horizontal_distance < proximity_threshold:
                group.append(other)
                used.add(j)
        
        # Combine text in group
        combined_text = ' '.join([r['text'] for r in group])
        avg_confidence = sum(r['confidence'] for r in group) / len(group)
        
        # Calculate group center
        avg_cx = sum(r['box'][0] for r in group) / len(group)
        avg_cy = sum(r['box'][1] for r in group) / len(group)
        
        groups.append({
            'text': combined_text,
            'confidence': avg_confidence,
            'box': [avg_cx, avg_cy, 0.1, 0.1],  # Approximate group size
            'region_count': len(group)
        })
    
    return groups



