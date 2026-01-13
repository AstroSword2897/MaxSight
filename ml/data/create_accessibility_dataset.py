#!/usr/bin/env python3
"""
MaxSight Accessibility Dataset (Production Version)

High-quality dataset system for therapy-focused visual accessibility features.

Features:
- Contrast sensitivity
- Glare risk
- Object findability
- Navigation difficulty
- Optional uncertainty weighting

Production improvements:
- No label corruption (synthetic labels separate from real labels)
- Medically-grounded augmentations
- Pre-computed augmentations (fast training)
- Clean dataset structure
- Therapy-oriented label schema
"""

import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ============================================================================
# Core Dataset (Pure Loader - No On-the-Fly Augmentation)
# ============================================================================

class AccessibilityDataset(Dataset):
    """
    Pure dataset loader for accessibility features.
    
    Production design:
    - Augmentations happen ONLY in synthetic generation, NOT on-the-fly
    - Real labels remain untouched
    - Fast __getitem__ (no augmentation overhead)
    - Clean separation of real vs synthetic data
    
        Arguments:
        image_dir: Directory containing images
        label_file: Path to JSON label file
        target_size: Target image size (height, width)
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_file: Path,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        
        # Load labels
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
        # Sort images with labels only (enforce label-image matching)
        self.image_files = [
            p for p in sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
            if p.stem in self.labels
        ]
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir} with matching labels in {label_file}")
        
        # Standard ImageNet transforms
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Returns:
            Dictionary with:
                - 'image': torch.Tensor [3, H, W] - Normalized image tensor
                - 'image_path': str - Path to image file
                - 'labels': Dict - Accessibility labels
        """
        path = self.image_files[idx]
        
        # Load and resize image
        img = Image.open(path).convert("RGB").resize(self.target_size)
        
        # Convert to tensor and normalize
        tensor = self.norm(self.to_tensor(img))
        
        # Get labels for this image
        labels = self.labels[path.stem]
        
        # Validate and format labels
        formatted_labels = {
            "contrast_sensitivity": float(labels.get("contrast_sensitivity", 0.5)),
            "glare_risk_level": int(labels.get("glare_risk_level", 0)),
            "object_findability": float(labels.get("object_findability", 0.5)),
            "navigation_difficulty": float(labels.get("navigation_difficulty", 0.5)),
        }
        
        # Validate ranges
        formatted_labels["contrast_sensitivity"] = np.clip(formatted_labels["contrast_sensitivity"], 0.0, 1.0)
        formatted_labels["glare_risk_level"] = np.clip(formatted_labels["glare_risk_level"], 0, 3)
        formatted_labels["object_findability"] = np.clip(formatted_labels["object_findability"], 0.0, 1.0)
        formatted_labels["navigation_difficulty"] = np.clip(formatted_labels["navigation_difficulty"], 0.0, 1.0)
        
        return {
            "image": tensor,
            "image_path": str(path),
            "labels": formatted_labels
        }


# ============================================================================
# Synthetic Augmentation Engine (Medically-Aligned)
# ============================================================================

class SyntheticImpairmentEngine:
    """
    Engine for generating synthetic impairments more realistically.
    
    Medically-grounded augmentations:
    - Contrast loss (simulates cataracts, visual acuity reduction)
    - Veiling glare (simulates lens flare, bright light sensitivity)
    - Peripheral blur (simulates visual field loss, glaucoma)
    - Depth flattening (simulates depth perception issues)
    - Halo effects (simulates post-cataract surgery, lens artifacts)
    """
    
    @staticmethod
    def apply_contrast_loss(img: Image.Image, level: float) -> Image.Image:
        """
        Apply contrast loss (medically-grounded).
        
        Simulates reduced contrast sensitivity (cataracts, visual acuity issues).
        Uses mean-preserving contrast reduction (not linear scaling).
        
        Arguments:
            img: PIL Image
            level: Contrast level [0, 1] where 1.0 = full contrast, 0.0 = no contrast
        
        Returns:
            Contrast-reduced PIL Image
        """
        arr = np.array(img).astype(np.float32)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        
        # Mean-preserving contrast reduction
        # Level 1.0 = full contrast, level 0.0 = gray (no contrast)
        contrast_factor = level
        arr = (arr - mean) * contrast_factor + mean
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_glare(img: Image.Image, intensity: float) -> Image.Image:
        """
        Apply veiling glare (medically-grounded).
        
        Simulates lens flare, bright light sensitivity, veiling glare.
        Uses Gaussian-based glare mask (not random).
        
        Arguments:
            img: PIL Image
            intensity: Glare intensity [0, 1]
        
        Returns:
            Image with veiling glare
        """
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Place glare source (prefer upper regions, like sunlight)
        gx = random.randint(int(w * 0.2), int(w * 0.8))
        gy = random.randint(0, int(h * 0.4))  # Upper region
        
        # Create Gaussian glare mask (high-frequency like real lens flare)
        y, x = np.ogrid[:h, :w]
        distance_sq = (x - gx)**2 + (y - gy)**2
        sigma_sq = (min(w, h) / 6.0) ** 2
        mask = np.exp(-distance_sq / (2 * sigma_sq))
        
        # Apply veiling glare (additive, not multiplicative)
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        glare = mask * intensity * 200  # Scale glare intensity
        arr = np.clip(arr + glare, 0, 255)
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def apply_peripheral_blur(img: Image.Image, amount: float) -> Image.Image:
        """
        Apply peripheral blur (medically-grounded).
        
        Simulates visual field loss, glaucoma, peripheral vision issues.
        Blur increases from center to periphery (not isotropic).
        
        Arguments:
            img: PIL Image
            amount: Blur amount (sigma for Gaussian) [0, 10]
        
        Returns:
            Image with peripheral blur
        """
        arr = np.array(img).astype(np.float32)
        
        # Apply Gaussian blur (handles both grayscale and RGB)
        # Pre-allocate output to avoid memory leaks from temporary arrays
        if len(arr.shape) == 3:
            # Multi-channel: apply filter to each channel with pre-allocated output
            blurred = np.empty_like(arr)
            for i in range(arr.shape[2]):
                ndimage.gaussian_filter(arr[:, :, i], sigma=amount, output=blurred[:, :, i])
        else:
            # Grayscale: single channel
            blurred = np.empty_like(arr)
            ndimage.gaussian_filter(arr, sigma=amount, output=blurred)
        
        # Create radial mask (center clear, periphery blurred)
        h, w = arr.shape[:2]
        mask = SyntheticImpairmentEngine._radial_mask((h, w))
        
        # Blend: center = original, periphery = blurred
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        blended = arr * mask + blurred * (1 - mask)
        
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_depth_flattening(img: Image.Image, strength: float) -> Image.Image:
        """
        Apply depth flattening (medically-grounded).
        
        Simulates reduced depth perception, stereopsis issues.
        Reduces contrast in depth cues.
        
        Arguments:
            img: PIL Image
            strength: Flattening strength [0, 1]
        
        Returns:
            Image with reduced depth cues
        """
        arr = np.array(img).astype(np.float32)
        
        # Reduce local contrast (depth cues rely on contrast)
        # Apply subtle blur to depth-separating edges
        if strength > 0:
            blurred = ndimage.gaussian_filter(arr, sigma=strength * 2.0)
            arr = arr * (1 - strength * 0.3) + blurred * (strength * 0.3)
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_halo_effect(img: Image.Image, intensity: float) -> Image.Image:
        """
        Apply halo effect (medically-grounded).
        
        Simulates post-cataract surgery halos, lens artifacts.
        Creates bright rings around high-contrast edges.
        
        Arguments:
            img: PIL Image
            intensity: Halo intensity [0, 1]
        
        Returns:
            Image with halo effects
        """
        arr = np.array(img).astype(np.float32)
        
        # Detect edges (high contrast regions)
        gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr
        edges = ndimage.gaussian_gradient_magnitude(gray, sigma=1.0)
        edges = edges / (edges.max() + 1e-8)
        
        # Create halo (bright ring around edges)
        halo_mask = np.clip(edges * intensity * 0.5, 0, 1)
        if len(arr.shape) == 3:
            halo_mask = halo_mask[:, :, np.newaxis]
        
        arr = np.clip(arr + halo_mask * intensity * 100, 0, 255)
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def apply_low_resolution(img: Image.Image, acuity_drop: float) -> Image.Image:
        """
        Apply low resolution (visual acuity drop).
        
        Simulates reduced visual acuity, low-resolution vision.
        
        Arguments:
            img: PIL Image
            acuity_drop: Acuity reduction factor [0, 1] where 1.0 = severe reduction
        
        Returns:
            Lower resolution image
        """
        if acuity_drop <= 0:
            return img
        
        # Reduce resolution then upscale (simulates low acuity)
        scale_factor = 1.0 - acuity_drop * 0.7  # Max 70% reduction
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        
        # Downsample then upsample (creates pixelation/blur)
        low_res = img.resize(new_size, Image.LANCZOS)
        upscaled = low_res.resize(img.size, Image.LANCZOS)
        
        return upscaled
    
    @staticmethod
    def _radial_mask(shape: Tuple[int, int]) -> np.ndarray:
        """
        Create radial mask (center = 1.0, edges = 0.0).
        
        Arguments:
            shape: (height, width)
        
        Returns:
            Radial mask [H, W] with values [0, 1]
        """
        h, w = shape
        y, x = np.ogrid[-h/2:h/2, -w/2:w/2]
        r = np.sqrt(x*x + y*y)
        r = r / (r.max() + 1e-8)  # Normalize to [0, 1]
        
        # Invert: center = 1.0, edges = 0.0
        mask = 1.0 - r
        
        # Apply smooth falloff
        mask = np.power(mask, 2.0)
        
        return mask


# ============================================================================
# Synthetic Dataset Generator
# ============================================================================

def generate_synthetic_dataset(
    source: Path,
    output: Path,
    n_per_image: int = 8,
    augmentation_types: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Generate synthetic dataset with pre-computed augmentations.
    
    Production design:
    - Pre-computes all augmentations (fast training)
    - Maintains label distribution (stratified)
    - Separate from real labels (no corruption)
    - Medically-grounded augmentations
    
        Arguments:
        source: Source image directory
        output: Output directory for synthetic images
        n_per_image: Number of augmentations per source image
        augmentation_types: List of augmentation types to use
                          (default: ['contrast', 'glare', 'peripheral_blur'])
    
    Returns:
        Dictionary with generation statistics
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    if augmentation_types is None:
        augmentation_types = ['contrast', 'glare', 'peripheral_blur']
    
    synthetic_labels = {}
    engine = SyntheticImpairmentEngine()
    
    # Get source images
    images = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    
    if not images:
        raise ValueError(f"No images found in {source}")
    
    print(f"Generating synthetic dataset from {len(images)} source images...")
    
    for img_path in images:
        try:
            original = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue
        
        for i in range(n_per_image):
            # Select augmentation type
            mode = random.choice(augmentation_types)
            aug_image = original.copy()
            labels = {}
            
            if mode == "contrast":
                level = random.uniform(0.3, 0.9)
                aug_image = engine.apply_contrast_loss(original, level)
                labels = {
                    "contrast_sensitivity": level,
                    "glare_risk_level": 0,
                    "object_findability": max(0.1, level * 0.9),  # Lower contrast = harder to find
                    "navigation_difficulty": 1.0 - level,  # Lower contrast = harder navigation
                }
            
            elif mode == "glare":
                intensity = random.uniform(0.3, 1.0)
                aug_image = engine.apply_glare(original, intensity)
                labels = {
                    "contrast_sensitivity": 0.6,  # Glare reduces effective contrast
                    "glare_risk_level": min(3, int(intensity * 3)),
                    "object_findability": max(0.2, 0.6 - intensity * 0.3),  # Glare obscures objects
                    "navigation_difficulty": 0.5 + intensity * 0.3,  # Glare increases difficulty
                }
            
            elif mode == "peripheral_blur":
                sigma = random.uniform(2.0, 5.0)
                aug_image = engine.apply_peripheral_blur(original, sigma)
                labels = {
                    "contrast_sensitivity": 0.8,  # Center still clear
                    "glare_risk_level": 0,
                    "object_findability": max(0.3, 1.0 - sigma / 5.0),  # Peripheral objects harder to find
                    "navigation_difficulty": min(1.0, sigma / 5.0),  # Peripheral blur increases difficulty
                }
            
            elif mode == "depth_flattening":
                strength = random.uniform(0.3, 0.8)
                aug_image = engine.apply_depth_flattening(original, strength)
