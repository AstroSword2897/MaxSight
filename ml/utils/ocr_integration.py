"""
OCR Integration Module for MaxSight
Sprint 1 Day 5: OCR Integration & Text Reading

This module provides:
- Text region detection (using model's text_head)
- OCR text extraction (iOS Vision framework integration)
- Text-to-speech pipeline for reading aloud
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image


class OCRIntegration:
    """
    OCR integration for MaxSight - reads text from detected regions.
    
    For iOS: Uses Vision framework VNRecognizeTextRequest
    For Python: Uses fallback text extraction methods
    """
    
    def __init__(self, text_threshold: float = 0.5, confidence_threshold: float = 0.3):
        """
        Initialize OCR integration.
        
        Args:
            text_threshold: Threshold for text region detection from model
            confidence_threshold: Minimum confidence for OCR text recognition
        """
        self.text_threshold = text_threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_text_regions_from_model(
        self,
        text_scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[Dict]:
        """
        Detect text regions from model's text_head output.
        
        Args:
            text_scores: Text probability scores [N] or [H, W]
            boxes: Bounding boxes [N, 4] in center format (cx, cy, w, h)
            image_size: Image dimensions (height, width)
        
        Returns:
            List of text region dicts with 'box', 'confidence', 'region_id'
        """
        text_regions = []
        
        # Handle different input shapes
        if text_scores.dim() == 2:  # [H, W] - spatial map
            # Find regions above threshold
            h, w = text_scores.shape
            y_coords, x_coords = torch.where(text_scores > self.text_threshold)
            
            if len(y_coords) > 0:
                # Group nearby pixels into regions (simple clustering)
                regions = self._cluster_text_pixels(x_coords, y_coords, h, w)
                
                for region_id, (x_min, y_min, x_max, y_max) in enumerate(regions):
                    # Convert to center format and normalize
                    cx = ((x_min + x_max) / 2) / w
                    cy = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    
                    # Get average confidence
                    region_scores = text_scores[y_min:y_max+1, x_min:x_max+1]
                    confidence = float(region_scores.mean().item())
                    
                    text_regions.append({
                        'box': [cx, cy, width, height],
                        'confidence': confidence,
                        'region_id': region_id
                    })
        
        elif text_scores.dim() == 1 and boxes.shape[0] == text_scores.shape[0]:
            # [N] scores with matching boxes
            text_mask = text_scores > self.text_threshold
            text_boxes = boxes[text_mask]
            text_confidences = text_scores[text_mask]
            
            for i, (box, conf) in enumerate(zip(text_boxes, text_confidences)):
                text_regions.append({
                    'box': box.tolist() if isinstance(box, torch.Tensor) else box,
                    'confidence': float(conf.item()) if isinstance(conf, torch.Tensor) else conf,
                    'region_id': i
                })
        
        return text_regions
    
    def _cluster_text_pixels(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        h: int,
        w: int,
        cluster_distance: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Simple clustering of text pixels into regions.
        
        Args:
            x_coords: X coordinates of text pixels
            y_coords: Y coordinates of text pixels
            h: Image height
            w: Image width
            cluster_distance: Maximum distance for clustering
        
        Returns:
            List of (x_min, y_min, x_max, y_max) bounding boxes
        """
        if len(x_coords) == 0:
            return []
        
        # Convert to numpy for easier processing
        coords = torch.stack([x_coords, y_coords], dim=1).cpu().numpy()
        
        # Simple distance-based clustering
        regions = []
        used = set()
        
        for i, (x, y) in enumerate(coords):
            if i in used:
                continue
            
            # Start new region
            cluster = [i]
            used.add(i)
            x_min, y_min, x_max, y_max = x, y, x, y
            
            # Find nearby pixels
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
    
    def extract_text_from_region(
        self,
        image: Image.Image,
        region_box: List[float],
        use_vision_framework: bool = False
    ) -> Optional[str]:
        """
        Extract text from a specific image region.
        
        Args:
            image: PIL Image
            region_box: Bounding box [cx, cy, w, h] in normalized coordinates
            use_vision_framework: If True, use iOS Vision framework (requires iOS)
        
        Returns:
            Extracted text string or None
        """
        # Crop region from image
        # PIL Image.size is (width, height), not (height, width)
        w, h = image.size
        cx, cy, width, height = region_box
        
        # Convert center format to corner format
        x1 = int((cx - width / 2) * w)
        y1 = int((cy - height / 2) * h)
        x2 = int((cx + width / 2) * w)
        y2 = int((cy + height / 2) * h)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop region
        region_image = image.crop((x1, y1, x2, y2))
        
        if use_vision_framework:
            # iOS Vision framework integration (for iOS app)
            # This would call VNRecognizeTextRequest in Swift
            return self._extract_text_vision_framework(region_image)
        else:
            # Python fallback: simple OCR using pytesseract or similar
            return self._extract_text_fallback(region_image)
    
    def _extract_text_vision_framework(self, image: Image.Image) -> Optional[str]:
        """
        Extract text using iOS Vision framework.
        This is a placeholder - actual implementation in iOS app.
        """
        # In iOS app, this would be:
        # let request = VNRecognizeTextRequest { request, error in
        #     guard let observations = request.results else { return }
        #     // Extract text from observations
        # }
        # request.recognitionLevel = .accurate
        # try? VNImageRequestHandler(cgImage: image.cgImage!).perform([request])
        
        return None  # Placeholder
    
    def _extract_text_fallback(self, image: Image.Image) -> Optional[str]:
        """
        Fallback text extraction for Python (development/testing).
        Uses pytesseract if available, otherwise returns placeholder.
        """
        try:
            import pytesseract
            # Preprocess image for better OCR
            # Convert to grayscale, enhance contrast
            gray = image.convert('L')
            # Simple threshold
            threshold = 128
            binary = gray.point(lambda p: 255 if p > threshold else 0, mode='1')
            
            # Extract text
            text = pytesseract.image_to_string(binary, config='--psm 7')
            return text.strip() if text.strip() else None
        except ImportError:
            # pytesseract not available - return placeholder
            return "[Text detected - install pytesseract for extraction]"
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return None
    
    def process_image_for_ocr(
        self,
        image: Image.Image,
        text_scores: torch.Tensor,
        boxes: torch.Tensor,
        max_regions: int = 10
    ) -> List[Dict]:
        """
        Complete OCR pipeline: detect regions and extract text.
        
        Args:
            image: PIL Image
            text_scores: Text detection scores from model
            boxes: Bounding boxes from model
            max_regions: Maximum number of text regions to process
        
        Returns:
            List of dicts with 'box', 'text', 'confidence', 'region_id'
        """
        # Detect text regions
        text_regions = self.detect_text_regions_from_model(text_scores, boxes)
        
        # Sort by confidence and limit
        text_regions.sort(key=lambda x: x['confidence'], reverse=True)
        text_regions = text_regions[:max_regions]
        
        # Extract text from each region
        results = []
        for region in text_regions:
            text = self.extract_text_from_region(image, region['box'])
            if text:
                results.append({
                    'box': region['box'],
                    'text': text,
                    'confidence': region['confidence'],
                    'region_id': region['region_id']
                })
        
        return results


def create_text_description(text_results: List[Dict], verbosity: str = 'normal') -> str:
    """
    Create natural language description of detected text.
    
    Args:
        text_results: List of OCR results from process_image_for_ocr
        verbosity: 'brief', 'normal', or 'detailed'
    
    Returns:
        Natural language description
    """
    if not text_results:
        return "No text detected"
    
    if verbosity == 'brief':
        return f"Text detected: {len(text_results)} region(s)"
    
    elif verbosity == 'normal':
        texts = [r['text'] for r in text_results[:3]]  # First 3 texts
        if len(texts) == 1:
            return f"Text: {texts[0]}"
        else:
            return f"Text detected: {', '.join(texts)}"
    
    else:  # detailed
        descriptions = []
        for i, result in enumerate(text_results[:5], 1):
            pos = "left" if result['box'][0] < 0.33 else ("right" if result['box'][0] > 0.67 else "center")
            descriptions.append(f"Text {i} ({pos}): {result['text']}")
        return "; ".join(descriptions)


if __name__ == "__main__":
    # Test OCR integration
    print("OCR Integration Module Test")
    print("=" * 50)
    
    ocr = OCRIntegration(text_threshold=0.5)
    
    # Create dummy text detection output
    dummy_text_scores = torch.rand(14, 14) * 0.3  # Low scores
    dummy_text_scores[5:8, 5:8] = 0.8  # Text region
    dummy_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])  # Center box
    
    regions = ocr.detect_text_regions_from_model(dummy_text_scores, dummy_boxes)
    print(f"Detected {len(regions)} text regions")
    
    for region in regions:
        print(f"  Region {region['region_id']}: confidence={region['confidence']:.2f}, box={region['box']}")

