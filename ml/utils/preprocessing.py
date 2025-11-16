"""
Preprocessing Pipeline for Environmental Structuring
Image transforms, audio MFCC, distance estimation, text detection
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import Tuple, Optional
from PIL import Image

# OpenCV is optional - used for advanced image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Some image enhancement features will be disabled.")
    print("Install with: pip install opencv-python")


class ImagePreprocessor:
    """Image preprocessing with condition-specific augmentations"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        condition_mode: Optional[str] = None
    ):
        self.image_size = image_size
        self.condition_mode = condition_mode
        
        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply preprocessing with condition-specific visual enhancements (visual is primary focus)"""
        # Apply condition-specific transforms for all visual impairments
        if self.condition_mode == 'cataracts':
            image = self._enhance_contrast(image)  # High contrast for reduced acuity
        elif self.condition_mode == 'retinitis_pigmentosa':
            image = self._low_light_enhancement(image)  # Brightness for night blindness
        elif self.condition_mode in ['myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors']:
            image = self._simulate_refractive_error(image)  # Blur simulation for refractive errors
        elif self.condition_mode == 'glaucoma':
            image = self._enhance_peripheral(image)  # Peripheral emphasis
        elif self.condition_mode == 'amd':
            image = self._enhance_central(image)  # Central emphasis
        elif self.condition_mode == 'diabetic_retinopathy':
            image = self._enhance_edges(image)  # Edge enhancement for spotty vision
        elif self.condition_mode == 'color_blindness':
            image = self._simulate_color_blindness(image)  # Color shift simulation
        
        # Apply base transforms (converts Image to Tensor)
        return self.base_transform(image)  # type: ignore  # base_transform converts Image -> Tensor
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """High-contrast enhancement for cataracts"""
        if not CV2_AVAILABLE:
            # Fallback: Use PIL's enhance function
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Convert to numpy for OpenCV
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_array)
    
    def _low_light_enhancement(self, image: Image.Image) -> Image.Image:
        """Brightness enhancement for retinitis pigmentosa (night blindness/tunnel vision)"""
        img_array = np.array(image).astype(np.float32)
        
        # Gamma correction for brightness
        gamma = 0.5
        img_array = np.power(img_array / 255.0, gamma) * 255.0
        
        # Histogram stretching
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255.0
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _simulate_refractive_error(self, image: Image.Image) -> Image.Image:
        """Simulate blurry vision from refractive errors (myopia, hyperopia, astigmatism, presbyopia)"""
        if not CV2_AVAILABLE:
            # Fallback: Use PIL's filter for blur
            from PIL import ImageFilter
            return image.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Convert to numpy for OpenCV
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply Gaussian blur to simulate blurry vision
        # Different blur levels for different refractive errors
        blur_kernel = 5  # Moderate blur for general refractive errors
        blurred = cv2.GaussianBlur(img_array, (blur_kernel, blur_kernel), 0)
        
        # Enhance contrast to compensate for blur
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_array)
    
    def _enhance_peripheral(self, image: Image.Image) -> Image.Image:
        """Enhance peripheral regions for glaucoma (peripheral vision loss)"""
        if not CV2_AVAILABLE:
            return image
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create mask that emphasizes peripheral regions
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        peripheral_mask = 1.0 + 0.5 * (dist_from_center / max_dist)  # Boost peripheral
        
        img_array = (img_array * peripheral_mask[..., np.newaxis]).astype(np.uint8)
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array)
    
    def _enhance_central(self, image: Image.Image) -> Image.Image:
        """Enhance central regions for AMD (central vision loss)"""
        if not CV2_AVAILABLE:
            return image
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create mask that emphasizes central regions
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        central_mask = 1.0 + 0.8 * (1.0 - dist_from_center / max_dist)  # Boost central
        
        img_array = (img_array * central_mask[..., np.newaxis]).astype(np.uint8)
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array)
    
    def _enhance_edges(self, image: Image.Image) -> Image.Image:
        """Enhance edges for diabetic retinopathy (spotty/blurry vision)"""
        if not CV2_AVAILABLE:
            return image
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        img_array = cv2.addWeighted(img_array, 0.7, sharpened, 0.3, 0)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_array)
    
    def _simulate_color_blindness(self, image: Image.Image) -> Image.Image:
        """Simulate color blindness (red-green color confusion)"""
        img_array = np.array(image).astype(np.float32)
        
        # Red-green color blindness: mix red and green channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        mixed = (r + g) / 2
        
        # Replace red and green with mixed value
        img_array[:, :, 0] = mixed
        img_array[:, :, 1] = mixed
        
        return Image.fromarray(img_array.astype(np.uint8))


class AudioPreprocessor:
    """Audio preprocessing - MFCC feature extraction"""
    
    def __init__(self, n_mfcc: int = 128, sample_rate: int = 16000):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
    
    def extract_mfcc(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio signal [samples] or [batch, samples]
        
        Returns:
            MFCC features [n_mfcc] or [batch, n_mfcc]
        """
        # TODO: Implement actual MFCC extraction using librosa or torchaudio
        # For now, return dummy features
        if audio.ndim == 1:
            return torch.randn(self.n_mfcc)
        else:
            batch_size = audio.shape[0]
            return torch.randn(batch_size, self.n_mfcc)


class DistanceEstimator:
    """Distance estimation preprocessing using perspective analysis"""
    
    def __init__(self):
        pass
    
    def estimate_distance_zones(
        self,
        bbox: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> int:
        """
        Estimate distance zone from bounding box size
        
        Args:
            bbox: Bounding box [x, y, w, h] normalized [0, 1]
            image_size: Image dimensions
        
        Returns:
            Distance zone: 0=near, 1=medium, 2=far
        """
        # Use bbox area as proxy for distance
        # Larger boxes = closer objects
        area = bbox[2] * bbox[3]  # w * h
        
        if area > 0.3:  # Large box = close
            return 0  # near
        elif area > 0.1:  # Medium box
            return 1  # medium
        else:  # Small box = far
            return 2  # far


class TextRegionDetector:
    """Text region detection preprocessing for OCR integration"""
    
    def __init__(self):
        pass
    
    def detect_text_regions(
        self,
        image: np.ndarray
    ) -> list:
        """
        Detect text regions in image
        
        Args:
            image: Image array [H, W, 3]
        
        Returns:
            List of bounding boxes [x, y, w, h] for text regions
        """
        # TODO: Implement text detection using OpenCV or EAST detector
        # For now, return empty list
        return []


# Synthetic Impairment Functions
def apply_refractive_error_blur(image: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """Apply Gaussian blur for refractive errors"""
    kernel_size = int(2 * sigma * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_cataract_contrast(image: torch.Tensor, contrast_factor: float = 0.5) -> torch.Tensor:
    """Reduce contrast for cataracts simulation"""
    return TF.adjust_contrast(image, contrast_factor)


def apply_glaucoma_vignette(image: torch.Tensor, center_percent: float = 0.4) -> torch.Tensor:
    """Apply peripheral masking for glaucoma"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) * center_percent
    
    # Create circular mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist < radius).float()
    mask = mask.unsqueeze(0).expand_as(image)
    
    return image * mask


def apply_amd_central_darkening(image: torch.Tensor, darken_factor: float = 0.3) -> torch.Tensor:
    """Darken center region for AMD simulation"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) * 0.2
    
    # Create circular darkening mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = 1.0 - (dist < radius).float() * darken_factor
    mask = mask.unsqueeze(0).expand_as(image)
    
    return image * mask


def apply_low_light(image: torch.Tensor, brightness_factor: float = 0.3) -> torch.Tensor:
    """Reduce brightness for retinitis pigmentosa"""
    return image * brightness_factor


def apply_color_shift(image: torch.Tensor, shift_type: str = 'red_green') -> torch.Tensor:
    """Apply color shifts for color blindness simulation"""
    if shift_type == 'red_green':
        # Simulate red-green color blindness
        # Mix red and green channels
        r, g, b = image[0], image[1], image[2]
        mixed = (r + g) / 2
        image = torch.stack([mixed, mixed, b], dim=0)
    return image


if __name__ == "__main__":
    print("Preprocessing pipeline created successfully!")
    print("\nAvailable components:")
    print("- ImagePreprocessor: Image transforms with condition-specific augmentations")
    print("- AudioPreprocessor: MFCC feature extraction")
    print("- DistanceEstimator: Distance zone estimation")
    print("- TextRegionDetector: Text region detection")
    print("\nSynthetic impairment functions:")
    print("- apply_refractive_error_blur")
    print("- apply_cataract_contrast")
    print("- apply_glaucoma_vignette")
    print("- apply_amd_central_darkening")
    print("- apply_low_light")
    print("- apply_color_shift")

