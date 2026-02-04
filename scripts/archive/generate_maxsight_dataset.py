#!/usr/bin/env python3
"""
MaxSight Comprehensive Dataset Generator
=========================================

Highly variable, production-grade dataset generator that:
1. Generates COCO-format annotations from existing images or creates synthetic scenes
2. Applies diverse accessibility-focused augmentations
3. Links with training, testing, and simulation pipelines
4. Supports external datasets (COCO, Open Images) or standalone generation

Features:
- 14 visual impairment simulations (glaucoma, AMD, cataracts, etc.)
- 6 lighting conditions (bright, normal, dim, dark, mixed, outdoor)
- 10+ scenario types (indoor, outdoor, transit, retail, medical, etc.)
- Multi-object scene generation with realistic spatial relationships
- Automatic urgency and distance estimation
- Full COCO format output compatible with MaxSightDataset

Usage:
    python scripts/generate_maxsight_dataset.py --mode full --train-samples 1000 --val-samples 200
    python scripts/generate_maxsight_dataset.py --mode from-coco --coco-path datasets/coco
    python scripts/generate_maxsight_dataset.py --mode synthetic --output datasets/synthetic
"""

import sys
import json
import random
import math
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import COCO_CLASSES, COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for dataset generation."""
    seed: int = 42
    image_size: Tuple[int, int] = (224, 224)
    max_objects_per_image: int = 15
    min_objects_per_image: int = 1
    
    # Augmentation probabilities
    impairment_probability: float = 0.7
    lighting_variation_probability: float = 0.8
    noise_probability: float = 0.3
    
    # Class distribution weights (higher = more common)
    accessibility_class_boost: float = 2.0  # Boost accessibility classes
    
    # Scenario distribution
    scenario_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scenario_weights is None:
            self.scenario_weights = {
                'indoor_home': 0.15,
                'indoor_office': 0.10,
                'indoor_retail': 0.12,
                'indoor_medical': 0.08,
                'outdoor_street': 0.15,
                'outdoor_park': 0.08,
                'transit_station': 0.10,
                'transit_vehicle': 0.07,
                'building_entrance': 0.08,
                'emergency_scenario': 0.07
            }


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIO_OBJECTS = {
    'indoor_home': [
        'door', 'door_handle', 'chair', 'couch', 'dining table', 'bed', 
        'tv', 'refrigerator', 'microwave', 'sink', 'toilet', 'stairs',
        'light_fixture', 'window', 'floor', 'wall', 'coffee_table', 'lamp'
    ],
    'indoor_office': [
        'door', 'desk', 'office_chair', 'computer_monitor', 'keyboard', 'mouse',
        'printer', 'phone', 'whiteboard', 'filing_cabinet', 'elevator', 
        'elevator_button', 'exit_sign', 'fire_extinguisher', 'stairs'
    ],
    'indoor_retail': [
        'door', 'automatic_door', 'shopping_cart', 'shelf', 'checkout', 
        'cash_register', 'price_tag', 'exit_sign', 'restroom_sign', 
        'escalator', 'elevator', 'wheelchair_accessible', 'product_display'
    ],
    'indoor_medical': [
        'door', 'wheelchair', 'walker', 'hospital', 'reception_desk',
        'waiting_room', 'exit_sign', 'first_aid', 'defibrillator', 
        'elevator', 'wheelchair_ramp', 'accessibility_button', 'braille_sign'
    ],
    'outdoor_street': [
        'crosswalk', 'traffic light', 'stop sign', 'car', 'bus', 'truck',
        'bicycle', 'person', 'sidewalk', 'curb', 'curb_cut', 'fire hydrant',
        'street', 'road', 'parking_meter', 'bus_stop', 'bench'
    ],
    'outdoor_park': [
        'bench', 'tree', 'grass', 'path', 'fountain', 'person', 'dog',
        'bicycle', 'trash_can', 'drinking_fountain', 'playground', 'stairs',
        'fence', 'gate', 'sign'
    ],
    'transit_station': [
        'stairs', 'escalator', 'elevator', 'platform', 'train', 'bus',
        'ticket_machine', 'turnstile', 'exit_sign', 'information_sign',
        'braille_sign', 'tactile_paving', 'handrail', 'bench', 'clock'
    ],
    'transit_vehicle': [
        'seat', 'handrail', 'door', 'window', 'priority_seat', 'wheelchair_space',
        'exit_door', 'emergency_button', 'stop_button', 'display', 'speaker'
    ],
    'building_entrance': [
        'door', 'automatic_door', 'revolving_door', 'stairs', 'ramp',
        'wheelchair_ramp', 'handrail', 'accessibility_button', 'intercom',
        'doorbell', 'entrance', 'exit', 'lobby', 'reception_desk'
    ],
    'emergency_scenario': [
        'fire_extinguisher', 'fire_alarm', 'exit_sign', 'emergency_exit',
        'emergency_door', 'smoke_detector', 'defibrillator', 'first_aid',
        'emergency_light', 'fire_hydrant', 'emergency_phone', 'alarm_system'
    ]
}

LIGHTING_CONDITIONS = {
    'bright': {'brightness': 1.3, 'contrast': 1.1, 'description': 'Well-lit environment'},
    'normal': {'brightness': 1.0, 'contrast': 1.0, 'description': 'Standard lighting'},
    'dim': {'brightness': 0.6, 'contrast': 0.9, 'description': 'Low light conditions'},
    'dark': {'brightness': 0.3, 'contrast': 0.7, 'description': 'Very low visibility'},
    'mixed': {'brightness': 0.9, 'contrast': 1.2, 'description': 'Mixed lighting with shadows'},
    'outdoor_sunny': {'brightness': 1.4, 'contrast': 1.3, 'description': 'Bright outdoor sunlight'},
    'outdoor_overcast': {'brightness': 0.85, 'contrast': 0.95, 'description': 'Overcast outdoor'},
    'glare': {'brightness': 1.5, 'contrast': 0.8, 'description': 'Strong glare present'}
}

IMPAIRMENT_TYPES = [
    'none', 'glaucoma', 'amd', 'cataracts', 'diabetic_retinopathy',
    'retinitis_pigmentosa', 'color_blindness_protanopia', 
    'color_blindness_deuteranopia', 'color_blindness_tritanopia',
    'myopia', 'hyperopia', 'astigmatism', 'low_vision', 'night_blindness'
]


# =============================================================================
# Visual Impairment Simulator
# =============================================================================

class ImpairmentSimulator:
    """Simulates various visual impairments for accessibility testing."""
    
    @staticmethod
    def apply_glaucoma(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate glaucoma (peripheral vision loss)."""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Create radial mask (center clear, edges dark)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Tunnel vision effect
        radius = max_dist * (1 - severity * 0.6)
        mask = np.clip(1 - (dist - radius) / (max_dist * 0.3), 0, 1)
        
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        result = arr * mask
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_amd(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate AMD (central vision loss)."""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Create central dark spot
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        spot_radius = min(h, w) * severity * 0.3
        mask = np.clip((dist - spot_radius) / (spot_radius * 0.5), 0, 1)
        
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        # Add distortion to central area
        distorted = arr * 0.3 + 127  # Gray out center
        result = arr * mask + distorted * (1 - mask)
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_cataracts(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate cataracts (blur + reduced contrast)."""
        # Apply blur
        blur_radius = 1 + severity * 4
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Reduce contrast
        enhancer = ImageEnhance.Contrast(blurred)
        result = enhancer.enhance(1 - severity * 0.5)
        
        # Add slight yellow tint (common with cataracts)
        arr = np.array(result).astype(np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + severity * 0.1), 0, 255)  # Red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + severity * 0.08), 0, 255)  # Green
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def apply_diabetic_retinopathy(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate diabetic retinopathy (spots, blur)."""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Add random dark spots
        num_spots = int(severity * 20)
        for _ in range(num_spots):
            spot_x = random.randint(0, w - 1)
            spot_y = random.randint(0, h - 1)
            spot_radius = random.randint(5, 20)
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
            mask = np.clip(1 - dist / spot_radius, 0, 1) * severity * 0.5
            
            if len(arr.shape) == 3:
                mask = mask[:, :, np.newaxis]
            
            arr = arr * (1 - mask)
        
        result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        
        # Add slight blur
        blur_radius = severity * 2
        if blur_radius > 0.5:
            result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return result
    
    @staticmethod
    def apply_color_blindness(img: Image.Image, cb_type: str = 'protanopia') -> Image.Image:
        """Simulate color blindness."""
        arr = np.array(img).astype(np.float32)
        
        if cb_type == 'protanopia':
            # Red-green (no red)
            matrix = np.array([
                [0.567, 0.433, 0.0],
                [0.558, 0.442, 0.0],
                [0.0, 0.242, 0.758]
            ])
        elif cb_type == 'deuteranopia':
            # Red-green (no green)
            matrix = np.array([
                [0.625, 0.375, 0.0],
                [0.7, 0.3, 0.0],
                [0.0, 0.3, 0.7]
            ])
        elif cb_type == 'tritanopia':
            # Blue-yellow
            matrix = np.array([
                [0.95, 0.05, 0.0],
                [0.0, 0.433, 0.567],
                [0.0, 0.475, 0.525]
            ])
        else:
            return img
        
        result = np.dot(arr.reshape(-1, 3), matrix.T).reshape(arr.shape)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_low_vision(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate general low vision."""
        # Reduce resolution
        scale = 1 - severity * 0.6
        new_size = (int(img.width * scale), int(img.height * scale))
        if new_size[0] > 10 and new_size[1] > 10:
            img = img.resize(new_size, Image.LANCZOS).resize(img.size, Image.LANCZOS)
        
        # Reduce contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1 - severity * 0.3)
        
        # Slight blur
        if severity > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=severity * 2))
        
        return img
    
    @staticmethod
    def apply_night_blindness(img: Image.Image, severity: float = 0.5) -> Image.Image:
        """Simulate night blindness (reduced sensitivity in low light)."""
        arr = np.array(img).astype(np.float32)
        
        # Calculate luminance
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        
        # Dark areas become even darker
        dark_mask = np.clip(1 - luminance / 128, 0, 1) * severity
        if len(arr.shape) == 3:
            dark_mask = dark_mask[:, :, np.newaxis]
        
        arr = arr * (1 - dark_mask * 0.7)
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @classmethod
    def apply_impairment(cls, img: Image.Image, impairment_type: str, 
                         severity: float = 0.5) -> Image.Image:
        """Apply specified impairment to image."""
        if impairment_type == 'none':
            return img
        elif impairment_type == 'glaucoma':
            return cls.apply_glaucoma(img, severity)
        elif impairment_type == 'amd':
            return cls.apply_amd(img, severity)
        elif impairment_type == 'cataracts':
            return cls.apply_cataracts(img, severity)
        elif impairment_type == 'diabetic_retinopathy':
            return cls.apply_diabetic_retinopathy(img, severity)
        elif impairment_type.startswith('color_blindness'):
            cb_type = impairment_type.replace('color_blindness_', '')
            return cls.apply_color_blindness(img, cb_type)
        elif impairment_type == 'low_vision':
            return cls.apply_low_vision(img, severity)
        elif impairment_type == 'night_blindness':
            return cls.apply_night_blindness(img, severity)
        elif impairment_type in ['myopia', 'hyperopia', 'astigmatism']:
            # All refractive errors simulated as blur
            blur = 1 + severity * 3
            return img.filter(ImageFilter.GaussianBlur(radius=blur))
        else:
            return img


# =============================================================================
# Scene Generator
# =============================================================================

class SceneGenerator:
    """Generates synthetic scenes with objects for training."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        random.seed(config.seed)
        
        # Build class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(COCO_CLASSES)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(COCO_CLASSES)}
    
    def generate_base_image(self, scenario: str) -> Image.Image:
        """Generate a base image for a scenario."""
        w, h = self.config.image_size
        
        # Generate gradient background based on scenario
        if 'outdoor' in scenario:
            # Sky to ground gradient
            top_color = (135, 206, 235)  # Sky blue
            bottom_color = (34, 139, 34)  # Forest green (grass)
        elif 'indoor' in scenario:
            # Indoor walls
            colors = [(245, 245, 240), (230, 220, 200), (200, 200, 210)]
            top_color = random.choice(colors)
            bottom_color = (180, 160, 140)  # Floor
        elif 'transit' in scenario:
            top_color = (200, 200, 200)
            bottom_color = (100, 100, 100)
        elif 'emergency' in scenario:
            top_color = (200, 180, 180)
            bottom_color = (150, 140, 140)
        else:
            top_color = (200, 200, 200)
            bottom_color = (150, 150, 150)
        
        # Create gradient
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            ratio = y / h
            for c in range(3):
                arr[y, :, c] = int(top_color[c] * (1 - ratio) + bottom_color[c] * ratio)
        
        # Add some texture/noise
        noise = np.random.randint(-15, 15, (h, w, 3))
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(arr)
    
    def place_objects(self, scenario: str) -> List[Dict[str, Any]]:
        """Generate object placements for a scenario."""
        objects = []
        scenario_classes = SCENARIO_OBJECTS.get(scenario, COCO_BASE_CLASSES[:20])
        
        # Determine number of objects
        num_objects = random.randint(
            self.config.min_objects_per_image,
            self.config.max_objects_per_image
        )
        
        # Track occupied regions to avoid overlap
        occupied = []
        
        for _ in range(num_objects):
            # Select class (weighted toward accessibility classes)
            if random.random() < 0.6:
                # Use scenario-specific class
                category = random.choice(scenario_classes)
            else:
                # Use random class from full list
                if random.random() < 0.3:
                    category = random.choice(ACCESSIBILITY_CLASSES)
                else:
                    category = random.choice(COCO_BASE_CLASSES)
            
            # Ensure category exists in our class list
            if category not in self.class_to_idx:
                category = random.choice(COCO_BASE_CLASSES)
            
            # Generate bounding box (normalized coordinates)
            # Try to avoid overlap
            for attempt in range(10):
                cx = random.uniform(0.15, 0.85)
                cy = random.uniform(0.15, 0.85)
                w = random.uniform(0.05, 0.4)
                h = random.uniform(0.05, 0.4)
                
                # Check overlap with existing boxes
                box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                overlap = False
                for occ in occupied:
                    if self._boxes_overlap(box, occ):
                        overlap = True
                        break
                
                if not overlap:
                    occupied.append(box)
                    break
            else:
                continue  # Skip if couldn't find non-overlapping position
            
            # Estimate distance and urgency
            box_area = w * h
            distance_zone = self._estimate_distance(box_area)
            urgency = self._estimate_urgency(category, box_area)
            
            objects.append({
                'category': category,
                'class_idx': self.class_to_idx[category],
                'bbox': [cx, cy, w, h],  # Center format, normalized
                'distance_zone': distance_zone,
                'urgency': urgency,
                'confidence': random.uniform(0.7, 1.0)
            })
        
        return objects
    
    def _boxes_overlap(self, box1: List[float], box2: List[float], 
                       threshold: float = 0.3) -> bool:
        """Check if two boxes overlap significantly."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-8)
        return iou > threshold
    
    def _estimate_distance(self, box_area: float) -> int:
        """Estimate distance zone from box area."""
        if box_area > 0.1:
            return 0  # Near
        elif box_area > 0.04:
            return 1  # Medium
        else:
            return 2  # Far
    
    def _estimate_urgency(self, category: str, box_area: float) -> int:
        """Estimate urgency from category and size."""
        category_lower = category.lower()
        
        danger_keywords = {'car', 'truck', 'bus', 'vehicle', 'fire', 'emergency', 
                          'traffic', 'alarm', 'hazard'}
        warning_keywords = {'person', 'bicycle', 'motorcycle', 'dog', 'stairs', 
                           'escalator', 'obstacle'}
        caution_keywords = {'door', 'curb', 'step', 'ramp', 'elevator'}
        
        if any(kw in category_lower for kw in danger_keywords):
            return 3 if box_area > 0.08 else 2
        elif any(kw in category_lower for kw in warning_keywords):
            return 2 if box_area > 0.1 else 1
        elif any(kw in category_lower for kw in caution_keywords):
            return 1 if box_area > 0.08 else 0
        else:
            return 0
    
    def apply_lighting(self, img: Image.Image, lighting: str) -> Image.Image:
        """Apply lighting conditions to image."""
        config = LIGHTING_CONDITIONS.get(lighting, LIGHTING_CONDITIONS['normal'])
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(config['brightness'])
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(config['contrast'])
        
        return img
    
    def draw_objects_on_image(self, img: Image.Image, 
                              objects: List[Dict[str, Any]]) -> Image.Image:
        """Draw object representations on image (for visualization)."""
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        for obj in objects:
            cx, cy, bw, bh = obj['bbox']
            
            # Convert to pixel coordinates
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Color based on urgency
            urgency = obj.get('urgency', 0)
            colors = [(0, 200, 0), (200, 200, 0), (255, 128, 0), (255, 0, 0)]
            color = colors[min(urgency, 3)]
            
            # Draw rectangle with some fill
            fill_color = tuple(list(color) + [50])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Add object shape (simple representation)
            if random.random() < 0.5:
                # Filled shape
                inner_color = tuple(c // 2 + 64 for c in color)
                draw.rectangle([x1+2, y1+2, x2-2, y2-2], fill=inner_color)
        
        return img


# =============================================================================
# COCO Format Annotation Generator
# =============================================================================

class COCOAnnotationGenerator:
    """Generates COCO-format annotations from generated scenes."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.annotation_id = 0
        
        # Build category mapping
        self.categories = []
        for idx, cls_name in enumerate(COCO_CLASSES):
            self.categories.append({
                'id': idx,
                'name': cls_name,
                'supercategory': self._get_supercategory(cls_name)
            })
    
    def _get_supercategory(self, class_name: str) -> str:
        """Get supercategory for a class."""
        class_lower = class_name.lower()
        
        if any(kw in class_lower for kw in ['door', 'window', 'wall', 'floor', 'stairs']):
            return 'building_structure'
        elif any(kw in class_lower for kw in ['car', 'bus', 'truck', 'vehicle', 'bicycle']):
            return 'vehicle'
        elif any(kw in class_lower for kw in ['sign', 'signal', 'light']):
            return 'signage'
        elif any(kw in class_lower for kw in ['chair', 'table', 'desk', 'bed', 'couch']):
            return 'furniture'
        elif any(kw in class_lower for kw in ['person', 'dog', 'cat', 'animal']):
            return 'living_being'
        elif any(kw in class_lower for kw in ['fire', 'emergency', 'alarm', 'exit']):
            return 'safety'
        elif any(kw in class_lower for kw in ['wheelchair', 'ramp', 'accessible', 'braille']):
            return 'accessibility'
        else:
            return 'object'
    
    def generate_annotation(self, image_id: int, image_info: Dict, 
                           objects: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Generate COCO-format annotation for an image."""
        w, h = self.config.image_size
        
        # Image info
        img_annotation = {
            'id': image_id,
            'file_name': image_info['file_name'],
            'width': w,
            'height': h,
            'date_captured': datetime.now().isoformat(),
            'license': 1,
            'coco_url': '',
            'flickr_url': ''
        }
        
        # Object annotations
        obj_annotations = []
        for obj in objects:
            cx, cy, bw, bh = obj['bbox']
            
            # Convert to COCO format [x, y, width, height] in pixels
            x = (cx - bw/2) * w
            y = (cy - bh/2) * h
            box_w = bw * w
            box_h = bh * h
            
            self.annotation_id += 1
            obj_annotations.append({
                'id': self.annotation_id,
                'image_id': image_id,
                'category_id': obj['class_idx'],
                'bbox': [x, y, box_w, box_h],
                'area': box_w * box_h,
                'iscrowd': 0,
                'segmentation': [],
                # MaxSight extensions
                'urgency': obj.get('urgency', 0),
                'distance_zone': obj.get('distance_zone', 1),
                'confidence': obj.get('confidence', 1.0)
            })
        
        return img_annotation, obj_annotations


# =============================================================================
# Main Dataset Generator
# =============================================================================

class MaxSightDatasetGenerator:
    """Main class for generating MaxSight training/validation datasets."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.scene_gen = SceneGenerator(config)
        self.coco_gen = COCOAnnotationGenerator(config)
        self.impairment_sim = ImpairmentSimulator()
        
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def generate_dataset(self, output_dir: Path, num_train: int = 1000, 
                        num_val: int = 200, num_test: int = 0,
                        use_existing_images: Optional[Path] = None
                        ) -> Dict[str, Any]:
        """
        Generate complete train/val/test dataset.
        
        Args:
            output_dir: Output directory for dataset
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples (optional held-out set)
            use_existing_images: Path to existing images to use as base
        
        Returns:
            Statistics dictionary
        """
        output_dir = Path(output_dir)
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        test_dir = output_dir / 'test'
        
        # Create directories
        if num_train > 0:
            (train_dir / 'images').mkdir(parents=True, exist_ok=True)
        if num_val > 0:
            (val_dir / 'images').mkdir(parents=True, exist_ok=True)
        if num_test > 0:
            (test_dir / 'images').mkdir(parents=True, exist_ok=True)
        
        print(f"Generating MaxSight Dataset")
        print(f"  Train samples: {num_train}")
        print(f"  Val samples: {num_val}")
        print(f"  Test samples: {num_test}")
        print(f"  Output: {output_dir}")
        
        # Load existing images if provided
        existing_images = []
        if use_existing_images and use_existing_images.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                existing_images.extend(list(use_existing_images.glob(ext)))
            print(f"  Using {len(existing_images)} existing images as base")
        
        # Generate training set
        train_stats = None
        if num_train > 0:
            print("\nGenerating training set...")
            train_stats = self._generate_split(
                train_dir, num_train, existing_images, 'train'
            )
        
        # Generate validation set
        val_stats = None
        if num_val > 0:
            print("\nGenerating validation set...")
            val_stats = self._generate_split(
                val_dir, num_val, existing_images, 'val'
            )
        
        # Generate test set
        test_stats = None
        if num_test > 0:
            print("\nGenerating test set...")
            test_stats = self._generate_split(
                test_dir, num_test, existing_images, 'test'
            )
        
        # Compile statistics
        stats = {
            'total_images': num_train + num_val + num_test,
            'config': asdict(self.config),
            'generated_at': datetime.now().isoformat(),
            'classes': len(COCO_CLASSES),
            'scenarios': list(SCENARIO_OBJECTS.keys()),
            'impairments': IMPAIRMENT_TYPES,
            'lighting_conditions': list(LIGHTING_CONDITIONS.keys())
        }
        if train_stats:
            stats['train'] = train_stats
        if val_stats:
            stats['val'] = val_stats
        if test_stats:
            stats['test'] = test_stats
        
        # Save statistics
        stats_file = output_dir / 'generation_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        if train_stats:
            print(f"   Train: {train_stats['num_images']} images, {train_stats['num_annotations']} annotations")
        if val_stats:
            print(f"   Val: {val_stats['num_images']} images, {val_stats['num_annotations']} annotations")
        if test_stats:
            print(f"   Test: {test_stats['num_images']} images, {test_stats['num_annotations']} annotations")
        print(f"   Stats saved to: {stats_file}")
        
        return stats
    
    def _generate_split(self, split_dir: Path, num_samples: int,
                       existing_images: List[Path], split_name: str) -> Dict:
        """Generate a train or val split."""
        images_dir = split_dir / 'images'
        
        # COCO format containers
        coco_images = []
        coco_annotations = []
        
        # Statistics
        scenario_counts = defaultdict(int)
        lighting_counts = defaultdict(int)
        impairment_counts = defaultdict(int)
        urgency_counts = defaultdict(int)
        class_counts = defaultdict(int)
        
        scenarios = list(self.config.scenario_weights.keys())
        scenario_probs = list(self.config.scenario_weights.values())
        
        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_samples} samples...")
            
            # Select scenario
            scenario = random.choices(scenarios, weights=scenario_probs)[0]
            scenario_counts[scenario] += 1
            
            # Generate or load base image
            if existing_images and random.random() < 0.5:
                # Use existing image as base
                base_img_path = random.choice(existing_images)
                try:
                    base_img = Image.open(base_img_path).convert('RGB')
                    base_img = base_img.resize(self.config.image_size)
                except Exception:
                    base_img = self.scene_gen.generate_base_image(scenario)
            else:
                base_img = self.scene_gen.generate_base_image(scenario)
            
            # Generate objects
            objects = self.scene_gen.place_objects(scenario)
            
            # Track class distribution
            for obj in objects:
                class_counts[obj['category']] += 1
                urgency_counts[obj['urgency']] += 1
            
            # Draw objects on image (creates visual representation)
            img = self.scene_gen.draw_objects_on_image(base_img, objects)
            
            # Apply lighting variation
            if random.random() < self.config.lighting_variation_probability:
                lighting = random.choice(list(LIGHTING_CONDITIONS.keys()))
            else:
                lighting = 'normal'
            img = self.scene_gen.apply_lighting(img, lighting)
            lighting_counts[lighting] += 1
            
            # Apply visual impairment
            if random.random() < self.config.impairment_probability:
                impairment = random.choice(IMPAIRMENT_TYPES[1:])  # Skip 'none'
                severity = random.uniform(0.3, 0.8)
                img = self.impairment_sim.apply_impairment(img, impairment, severity)
            else:
                impairment = 'none'
            impairment_counts[impairment] += 1
            
            # Add noise
            if random.random() < self.config.noise_probability:
                arr = np.array(img).astype(np.float32)
                noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            
            # Save image
            image_id = i + 1
            file_name = f"{split_name}_{image_id:06d}.jpg"
            img.save(images_dir / file_name, quality=95)
            
            # Generate annotations
            img_info = {'file_name': file_name}
            img_ann, obj_anns = self.coco_gen.generate_annotation(
                image_id, img_info, objects
            )
            
            # Add metadata
            img_ann['scenario'] = scenario
            img_ann['lighting'] = lighting
            img_ann['impairment'] = impairment
            
            coco_images.append(img_ann)
            coco_annotations.extend(obj_anns)
        
        # Build COCO format annotation file
        coco_dataset = {
            'info': {
                'description': f'MaxSight {split_name} dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'MaxSight Generator',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{'id': 1, 'name': 'MaxSight License', 'url': ''}],
            'categories': self.coco_gen.categories,
            'images': coco_images,
            'annotations': coco_annotations
        }
        
        # Save annotations
        ann_file = split_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_dataset, f)
        
        return {
            'num_images': len(coco_images),
            'num_annotations': len(coco_annotations),
            'scenario_distribution': dict(scenario_counts),
            'lighting_distribution': dict(lighting_counts),
            'impairment_distribution': dict(impairment_counts),
            'urgency_distribution': dict(urgency_counts),
            'top_classes': dict(sorted(class_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:20])
        }
    
    def generate_from_coco(self, coco_path: Path, output_dir: Path,
                          num_train: int = 5000, num_val: int = 1000) -> Dict:
        """
        Generate MaxSight dataset from COCO dataset.
        
        Applies augmentations and adds accessibility annotations.
        """
        from ml.data.generate_annotations import generate_annotations_from_coco
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find COCO annotations
        coco_ann_file = coco_path / 'annotations' / 'instances_train2017.json'
        coco_img_dir = coco_path / 'train2017'
        
        if not coco_ann_file.exists():
            raise FileNotFoundError(f"COCO annotations not found at {coco_ann_file}")
        
        print(f"Converting COCO dataset to MaxSight format...")
        print(f"  Source: {coco_path}")
        print(f"  Output: {output_dir}")
        
        # Generate MaxSight annotations from COCO
        train_file, val_file = generate_annotations_from_coco(
            coco_annotation_file=coco_ann_file,
            image_dir=coco_img_dir,
            output_file=output_dir / 'maxsight_annotations.json',
            num_samples=num_train + num_val,
            train_split=num_train / (num_train + num_val)
        )
        
        return {
            'train_annotations': str(train_file),
            'val_annotations': str(val_file),
            'source': str(coco_path),
            'num_train': num_train,
            'num_val': num_val
        }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MaxSight Comprehensive Dataset Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full synthetic dataset
  python generate_maxsight_dataset.py --mode full --train-samples 1000 --val-samples 200

  # Generate from existing test images
  python generate_maxsight_dataset.py --mode full --use-existing test_images --train-samples 500

  # Convert COCO to MaxSight format
  python generate_maxsight_dataset.py --mode from-coco --coco-path datasets/coco

  # Quick test generation
  python generate_maxsight_dataset.py --mode quick --train-samples 50 --val-samples 10
        """
    )
    
    parser.add_argument('--mode', choices=['full', 'from-coco', 'quick', 'synthetic'],
                       default='full', help='Generation mode')
    parser.add_argument('--output', type=Path, default=Path('datasets'),
                       help='Output directory')
    parser.add_argument('--train-samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=200,
                       help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=0,
                       help='Number of test samples (optional held-out set)')
    parser.add_argument('--coco-path', type=Path, default=None,
                       help='Path to COCO dataset (for from-coco mode)')
    parser.add_argument('--use-existing', type=Path, default=None,
                       help='Path to existing images to use as base')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (square)')
    
    args = parser.parse_args()
    
    # Create config
    config = GeneratorConfig(
        seed=args.seed,
        image_size=(args.image_size, args.image_size)
    )
    
    # Create generator
    generator = MaxSightDatasetGenerator(config)
    
    if args.mode == 'from-coco':
        if not args.coco_path:
            print("Error: --coco-path required for from-coco mode")
            return
        stats = generator.generate_from_coco(
            args.coco_path, args.output,
            num_train=args.train_samples,
            num_val=args.val_samples
        )
    elif args.mode == 'quick':
        # Quick test mode with minimal samples
        stats = generator.generate_dataset(
            args.output,
            num_train=50,
            num_val=10,
            use_existing_images=args.use_existing or Path('test_images')
        )
    else:
        # Full or synthetic mode
        stats = generator.generate_dataset(
            args.output,
            num_train=args.train_samples,
            num_val=args.val_samples,
            num_test=args.test_samples,
            use_existing_images=args.use_existing
        )
    
    print("\n📊 Generation Statistics:")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == '__main__':
    main()

