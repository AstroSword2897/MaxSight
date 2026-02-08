"""Region Extractor for Multi-Vector Retrieval

Extracts object-level region embeddings using MaxSightCNN/DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torchvision.transforms as T


class RegionExtractor(nn.Module):
    """Region extractor for object-level embeddings.
    
    Uses detection model to extract bounding boxes, then encodes regions."""
    
    def __init__(
        self,
        detection_model: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        max_regions: int = 8,
        region_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        
        self.detection_model = detection_model
        self.encoder = encoder
        self.max_regions = max_regions
        self.region_size = region_size
        
        # Region encoder (if not provided, use simple CNN)
        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 256)
            )
        
        # Region normalization.
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def extract_regions(
        self,
        images: torch.Tensor,  # [B, 3, H, W].
        boxes: Optional[torch.Tensor] = None  # [B, N, 4] (x1, y1, x2, y2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract region embeddings...."""
        B, C, H, W = images.shape
        
        # Get boxes from detection model if not provided.
        if boxes is None and self.detection_model is not None:
            with torch.no_grad():
                outputs = self.detection_model(images)
                # Extract boxes from outputs (format depends on model)
                # Placeholder: would need to adapt to actual model output format.
                boxes = torch.zeros(B, self.max_regions, 4, device=images.device)
        
        if boxes is None:
            # Fallback: use grid regions.
            boxes = self._create_grid_regions(B, H, W, device=images.device)
        
        # Crop and encode regions.
        region_embeddings = []
        valid_regions = []
        
        for b in range(B):
            region_embs = []
            valid = []
            
            for i in range(min(self.max_regions, boxes.shape[1])):
                box = boxes[b, i]
                
                # Skip invalid boxes.
                if (box[2] <= box[0]) or (box[3] <= box[1]):
                    region_embs.append(torch.zeros(256, device=images.device))
                    valid.append(False)
                    continue
                
                # Crop region.
                x1, y1, x2, y2 = box.int()
                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))
                x2 = max(x1, min(x2, W))
                y2 = max(y1, min(y2, H))
                
                if x2 <= x1 or y2 <= y1:
                    region_embs.append(torch.zeros(256, device=images.device))
                    valid.append(False)
                    continue
                
                region = images[b:b+1, :, y1:y2, x1:x2]  # [1, 3, h, w].
                
                # Resize to fixed size.
                region = F.interpolate(region, size=self.region_size, mode='bilinear', align_corners=False)
                
                # Normalize.
                region = self.normalize(region)
                
                # Encode.
                region_emb = self.encoder(region).squeeze(0)  # [embed_dim].
                region_embs.append(region_emb)
                valid.append(True)
            
            # Pad to max_regions.
            while len(region_embs) < self.max_regions:
                region_embs.append(torch.zeros(256, device=images.device))
                valid.append(False)
            
            region_embeddings.append(torch.stack(region_embs[:self.max_regions]))
            valid_regions.append(valid)
        
        region_embeddings = torch.stack(region_embeddings)  # [B, max_regions, embed_dim].
        
        # L2 normalize.
        region_embeddings = F.normalize(region_embeddings, p=2, dim=2)
        
        return region_embeddings, boxes
    
    def _create_grid_regions(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create grid-based regions as fallback."""
        grid_size = int(self.max_regions ** 0.5)
        boxes = []
        
        for b in range(B):
            batch_boxes = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * W // grid_size
                    y1 = i * H // grid_size
                    x2 = (j + 1) * W // grid_size
                    y2 = (i + 1) * H // grid_size
                    batch_boxes.append([x1, y1, x2, y2])
            
            # Pad to max_regions.
            while len(batch_boxes) < self.max_regions:
                batch_boxes.append([0, 0, W, H])
            
            boxes.append(batch_boxes[:self.max_regions])
        
        return torch.tensor(boxes, device=device, dtype=torch.float32)



