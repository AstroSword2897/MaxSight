"""
Dataset Creation Utilities for Accessibility Features
Creates synthetic and user-labeled datasets for contrast, glare, findability, navigation.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import random


class AccessibilityDataset(Dataset):
    """
    Dataset for accessibility feature training.
    Supports synthetic augmentation and user-labeled data.
    """
    
    def __init__(
        self,
        image_dir: Path,
        annotations_file: Optional[Path] = None,
        synthetic_augment: bool = True,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.synthetic_augment = synthetic_augment
        
        # Load annotations if provided
        self.annotations = {}
        if annotations_file and annotations_file.exists():
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
        
        # Get image list
        self.images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        # Standard transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size)
        
        # Get annotations for this image
        image_id = image_path.stem
        annotation = self.annotations.get(image_id, {})
        
        # Apply synthetic augmentation if enabled
        if self.synthetic_augment:
            image, aug_labels = self._apply_synthetic_augmentation(image, annotation)
        else:
            aug_labels = {}
        
        # Convert to tensor
        image_tensor = self.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        
        # Combine annotations
        labels = {
            'contrast_sensitivity': annotation.get('contrast_sensitivity', aug_labels.get('contrast_sensitivity', 0.5)),
            'glare_risk_level': annotation.get('glare_risk_level', aug_labels.get('glare_risk_level', 0)),
            'object_findability': annotation.get('object_findability', aug_labels.get('object_findability', 0.5)),
            'navigation_difficulty': annotation.get('navigation_difficulty', aug_labels.get('navigation_difficulty', 0.5)),
        }
        
        return {
            'image': image_tensor,
            'image_path': str(image_path),
            'labels': labels
        }
    
    def _apply_synthetic_augmentation(
        self,
        image: Image.Image,
        annotation: Dict
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply synthetic augmentations and generate labels.
        
        Returns:
            (augmented_image, synthetic_labels)
        """
        aug_labels = {}
        aug_image = image.copy()
        
        # Random augmentation selection
        aug_type = random.choice(['contrast', 'glare', 'blur', 'brightness', 'none'])
        
        if aug_type == 'contrast':
            # Reduce contrast
            factor = random.uniform(0.3, 0.7)
            aug_image = self._reduce_contrast(aug_image, factor)
            aug_labels['contrast_sensitivity'] = factor
            aug_labels['glare_risk_level'] = 0
        
        elif aug_type == 'glare':
            # Add synthetic glare
            intensity = random.uniform(0.5, 1.0)
            aug_image = self._add_glare(aug_image, intensity)
            aug_labels['glare_risk_level'] = min(3, int(intensity * 3))
            aug_labels['contrast_sensitivity'] = 0.7
        
        elif aug_type == 'blur':
            # Add blur (affects findability)
            sigma = random.uniform(1.0, 3.0)
            aug_image = self._add_blur(aug_image, sigma)
            aug_labels['object_findability'] = max(0.0, 1.0 - sigma / 3.0)
            aug_labels['navigation_difficulty'] = min(1.0, sigma / 3.0)
        
        elif aug_type == 'brightness':
            # Adjust brightness (affects contrast sensitivity)
            factor = random.uniform(0.5, 1.5)
            aug_image = self._adjust_brightness(aug_image, factor)
            aug_labels['contrast_sensitivity'] = 0.8 if 0.8 < factor < 1.2 else 0.5
        
        else:
            # No augmentation - use default labels
            aug_labels = {
                'contrast_sensitivity': 0.8,
                'glare_risk_level': 0,
                'object_findability': 0.7,
                'navigation_difficulty': 0.3
            }
        
        return aug_image, aug_labels
    
    def _reduce_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """Reduce image contrast"""
        enhancer = transforms.functional.adjust_contrast
        return enhancer(image, factor)
    
    def _add_glare(self, image: Image.Image, intensity: float) -> Image.Image:
        """Add synthetic glare effect"""
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create glare mask (bright spot)
        center_x, center_y = random.randint(0, w), random.randint(0, h)
        y, x = np.ogrid[:h, :w]
        mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (w/4)**2))
        mask = mask[:, :, np.newaxis] if len(img_array.shape) == 3 else mask
        
        # Add glare
        glare = mask * intensity * 100
        img_array = np.clip(img_array + glare, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _add_blur(self, image: Image.Image, sigma: float) -> Image.Image:
        """Add Gaussian blur"""
        from scipy import ndimage
        img_array = np.array(image)
        blurred = ndimage.gaussian_filter(img_array, sigma=sigma)
        return Image.fromarray(blurred.astype(np.uint8))
    
    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust image brightness"""
        enhancer = transforms.functional.adjust_brightness
        return enhancer(image, factor)


def create_synthetic_dataset(
    source_dir: Path,
    output_dir: Path,
    num_augmentations: int = 5
):
    """
    Create synthetic dataset with augmentations.
    
    Args:
        source_dir: Directory with source images
        output_dir: Directory to save augmented images and labels
        num_augmentations: Number of augmentations per source image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = {}
    source_images = list(Path(source_dir).glob("*.jpg")) + list(Path(source_dir).glob("*.png"))
    
    dataset = AccessibilityDataset(source_dir, synthetic_augment=True)
    
    for source_img in source_images:
        for aug_idx in range(num_augmentations):
            # Load and augment
            item = dataset[source_images.index(source_img)]
            
            # Save augmented image
            aug_id = f"{source_img.stem}_aug{aug_idx}"
            aug_path = output_dir / f"{aug_id}.jpg"
            
            # Convert tensor back to image for saving
            img_tensor = item['image']
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            Image.fromarray(img_array).save(aug_path)
            
            # Save annotation
            annotations[aug_id] = item['labels']
    
    # Save annotations
    with open(output_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {len(annotations)} synthetic samples in {output_dir}")


def create_labeling_template(output_file: Path):
    """
    Create a labeling template for user annotation.
    
    Args:
        output_file: Path to save template JSON
    """
    template = {
        "image_id": {
            "contrast_sensitivity": 0.0,  # 0-1 score
            "glare_risk_level": 0,  # 0-3 integer
            "object_findability": 0.0,  # 0-1 score (average across objects)
            "navigation_difficulty": 0.0,  # 0-1 score
            "notes": "Optional notes about the scene"
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created labeling template at {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create accessibility dataset")
    parser.add_argument("--source_dir", type=Path, required=True, help="Source image directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--num_aug", type=int, default=5, help="Number of augmentations per image")
    parser.add_argument("--create_template", action="store_true", help="Create labeling template")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_labeling_template(args.output_dir / "labeling_template.json")
    else:
        create_synthetic_dataset(args.source_dir, args.output_dir, args.num_aug)

