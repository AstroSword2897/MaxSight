"""
Preprocessing Pipeline for Environmental Structuring
Image transforms, audio MFCC, distance estimation, text detection
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image

# OpenCV is optional - used for advanced image processing
# Falls back to PIL if not available
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Some image enhancement features will be disabled.")
    print("Install with: pip install opencv-python")


class ImagePreprocessor:
    """
    Image preprocessing with condition-specific augmentations for visual impairments.
    
    Preprocesses images for MaxSight model with condition-specific enhancements that simulate
    or compensate for various visual impairments (cataracts, glaucoma, AMD, etc.). Applies
    standard ImageNet normalization and optional lighting condition detection/augmentation.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        condition_mode: Optional[str] = None
    ):
        """
        Initialize image preprocessor.
        
        Args:
            image_size: Target image dimensions (height, width) - default (224, 224) for ImageNet
            condition_mode: Visual condition to simulate ('glaucoma', 'amd', 'cataracts', etc.)
        """
        self.image_size = image_size
        self.condition_mode = condition_mode
        
        # Standard ImageNet normalization for pretrained ResNet compatibility
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB channel means
            std=[0.229, 0.224, 0.225]   # ImageNet RGB channel standard deviations
        )
        
        # Base transform pipeline: resize -> tensor -> normalize
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply preprocessing with condition-specific visual enhancements.
        
        Preprocesses image with condition-specific transforms (if enabled) followed by standard
        ImageNet preprocessing. All visual conditions are supported.
        
        Args:
            image: PIL Image to preprocess
        
        Returns:
            Preprocessed image as PyTorch Tensor [3, H, W] with ImageNet normalization applied
        """
        # Apply condition-specific transforms based on condition_mode
        if self.condition_mode == 'cataracts':
            image = self._enhance_contrast(image)
        elif self.condition_mode == 'retinitis_pigmentosa':
            image = self._low_light_enhancement(image)
        elif self.condition_mode in ['myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors']:
            image = self._simulate_refractive_error(image)
        elif self.condition_mode == 'glaucoma':
            image = self._enhance_peripheral(image)
        elif self.condition_mode == 'amd':
            image = self._enhance_central(image)
        elif self.condition_mode == 'diabetic_retinopathy':
            image = self._enhance_edges(image)
        elif self.condition_mode == 'color_blindness':
            image = self._simulate_color_blindness(image)
        
        # Apply standard ImageNet preprocessing
        return self.base_transform(image)  # type: ignore
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        High-contrast enhancement for cataracts (reduced visual acuity compensation).
        
        Purpose: Enhance image contrast to compensate for reduced visual acuity in cataracts.
                 Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) if OpenCV is available
                 for better results, or falls back to PIL's contrast enhancement. High contrast makes
                 objects more distinguishable for users with reduced visual acuity.
        
        Complexity: O(H*W) where H=height, W=width - processes all pixels for contrast enhancement
                   CLAHE: O(H*W*T) where T=tile size (8x8), but typically O(H*W) in practice
        Relationship: Cataract adaptation - improves image visibility for users with cataracts
        
        Args:
            image: PIL Image to enhance
        
        Returns:
            Enhanced PIL Image with increased contrast
        """
        # Check if OpenCV is available for advanced contrast enhancement
        # Purpose: Use OpenCV's CLAHE (superior) if available, otherwise fall back to PIL (simpler)
        # Complexity: O(1) - simple boolean check
        # Relationship: Dependency check - determines which enhancement method to use
        if not CV2_AVAILABLE:
            # Fallback: Use PIL's contrast enhancement (simpler, but less effective than CLAHE)
            # Purpose: Apply basic contrast enhancement using PIL when OpenCV is not available.
            #          Increases contrast by 50% (factor 1.5) to make objects more distinguishable.
            # Complexity: O(H*W) - processes all pixels for contrast adjustment
            # Relationship: Fallback method - ensures contrast enhancement works without OpenCV
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Convert PIL Image to numpy array for OpenCV processing
        # Purpose: Convert image format from PIL to numpy array (OpenCV format). Also convert RGB
        #          to BGR color space (OpenCV uses BGR by default).
        # Complexity: O(H*W) - converts image format and color space
        # Relationship: Format conversion - prepares image for OpenCV processing
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for superior contrast enhancement
        # Purpose: Apply CLAHE which adaptively enhances contrast in local regions (8x8 tiles) while
        #          limiting over-enhancement. This is superior to global contrast enhancement because
        #          it preserves local details and prevents over-saturation. Works in LAB color space
        #          (only enhances L channel) to preserve color information.
        # Complexity: O(H*W*T) where T=tile size (8x8) - processes image in tiles for adaptive enhancement
        #            In practice, this is O(H*W) since tile processing is efficient
        # Relationship: Advanced contrast enhancement - provides superior results for cataract compensation
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
        l, a, b = cv2.split(lab)  # Split into L (lightness), A, B channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE with 8x8 tiles
        l = clahe.apply(l)  # Apply CLAHE only to L channel (preserves color)
        lab = cv2.merge([l, a, b])  # Merge channels back
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
        
        # Convert back to PIL Image
        # Purpose: Convert processed numpy array back to PIL Image format for return
        # Complexity: O(H*W) - creates PIL Image from array
        # Relationship: Format conversion - returns image in expected format
        return Image.fromarray(img_array)
    
    def _low_light_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Brightness enhancement for retinitis pigmentosa (night blindness/tunnel vision compensation).
        
        Purpose: Enhance image brightness to compensate for night blindness and tunnel vision in
                 retinitis pigmentosa. Uses gamma correction to brighten image and histogram stretching
                 to maximize dynamic range. This makes images more visible in low-light conditions
                 where users with retinitis pigmentosa struggle.
        
        Complexity: O(H*W) where H=height, W=width - processes all pixels for brightness enhancement
                   Gamma correction: O(H*W) - element-wise power operation
                   Histogram stretching: O(H*W) - finds min/max and scales pixels
        Relationship: Retinitis pigmentosa adaptation - improves visibility in low-light conditions
        
        Args:
            image: PIL Image to enhance
        
        Returns:
            Enhanced PIL Image with increased brightness
        """
        # Convert PIL Image to numpy array with float32 precision for calculations
        # Purpose: Convert image to numpy array with float precision for accurate brightness calculations.
        #          Float32 provides sufficient precision while being memory-efficient.
        # Complexity: O(H*W) - converts image format
        # Relationship: Format conversion - prepares image for numerical operations
        img_array = np.array(image).astype(np.float32)
        
        # Apply gamma correction to brighten image (gamma < 1 brightens, gamma > 1 darkens)
        # Purpose: Brighten image using gamma correction. Gamma=0.5 means we raise pixel values to
        #          power of 0.5, which brightens the image (compresses bright values, expands dark values).
        #          This is effective for low-light enhancement because it makes dark regions more visible.
        # Complexity: O(H*W) - element-wise power operation for all pixels
        # Relationship: Brightness enhancement - first step in low-light compensation
        gamma = 0.5  # Gamma < 1 brightens image
        img_array = np.power(img_array / 255.0, gamma) * 255.0  # Normalize, apply gamma, scale back
        
        # Apply histogram stretching to maximize dynamic range
        # Purpose: Stretch histogram to use full [0, 255] range, maximizing contrast and brightness.
        #          Formula: (pixel - min) / (max - min) * 255 maps [min, max] to [0, 255].
        #          This ensures darkest pixel becomes 0 and brightest becomes 255, maximizing visibility.
        # Complexity: O(H*W) - finds min/max (O(H*W)) and scales all pixels (O(H*W))
        # Relationship: Dynamic range maximization - second step in low-light compensation
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255.0
        # Add epsilon (1e-8) to prevent division by zero if all pixels are same value
        
        # Convert back to uint8 and PIL Image format
        # Purpose: Convert processed float array back to uint8 (0-255 range) and PIL Image format
        # Complexity: O(H*W) - type conversion and PIL Image creation
        # Relationship: Format conversion - returns image in expected format
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _analyze_lighting_condition(self, image: Image.Image) -> str:
        """
        Analyze image brightness and classify lighting condition.
        
        Purpose: Classify images into lighting categories (bright, normal, dim, dark) for lighting-aware
                 evaluation and training. This enables tracking model performance across different lighting
                 conditions, which is critical for accessibility applications where users may encounter
                 various lighting scenarios.
        
        Complexity: O(H*W) where H=height, W=width - requires scanning all pixels for brightness analysis
        Relationship: Used by preprocessing pipeline to label images with lighting metadata, enabling
                     lighting-stratified metrics in validation and training.
        
        Algorithm:
        1. Convert image to grayscale (average RGB channels)
        2. Calculate mean brightness (average pixel value)
        3. Calculate standard deviation (brightness variation)
        4. Classify based on thresholds:
           - bright: mean > 180 AND std > 30 (overexposed, high contrast)
           - normal: 120 <= mean <= 180 AND std > 20 (typical daylight)
           - dim: 60 <= mean < 120 OR (mean >= 120 AND std < 20) (low light, low contrast)
           - dark: mean < 60 (very low light, night conditions)
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Lighting condition string: 'bright', 'normal', 'dim', or 'dark'
        """
        # Convert to grayscale for brightness analysis - average RGB channels
        # Complexity: O(H*W) - processes all pixels once
        img_array = np.array(image).astype(np.float32)
        if len(img_array.shape) == 3:
            gray_image = np.mean(img_array, axis=2)  # Average RGB channels to get grayscale
        else:
            gray_image = img_array  # Already grayscale
        
        # Calculate mean brightness - average of all pixel values
        # Complexity: O(H*W) - sums all pixels, then divides
        mean_brightness = np.mean(gray_image)
        
        # Calculate standard deviation - measures brightness variation across image
        # Complexity: O(H*W) - computes variance then square root
        std_brightness = np.std(gray_image)
        
        # Classification based on brightness thresholds
        # Thresholds chosen based on typical image brightness distributions:
        # - Bright: >180 (overexposed, sunny conditions)
        # - Normal: 120-180 (typical indoor/outdoor daylight)
        # - Dim: 60-120 (low light, evening, cloudy)
        # - Dark: <60 (night, very low light)
        # Standard deviation helps distinguish high-contrast bright images from uniform bright images
        if mean_brightness > 180 and std_brightness > 30:
            return 'bright'  # Overexposed, high contrast (sunny, bright indoor)
        elif 120 <= mean_brightness <= 180 and std_brightness > 20:
            return 'normal'  # Typical daylight conditions
        elif 60 <= mean_brightness < 120 or (mean_brightness >= 120 and std_brightness < 20):
            return 'dim'  # Low light, low contrast (evening, cloudy, dim indoor)
        else:  # mean_brightness < 60
            return 'dark'  # Very low light (night, dark room)
    
    def _simulate_bright_lighting(self, image: Image.Image, brightness_factor: float = 1.5) -> Image.Image:
        """
        Simulate overexposed/bright lighting conditions.
        
        Purpose: Augment images to simulate bright lighting scenarios (sunny day, bright indoor lighting)
                 for training and evaluation. Helps model learn to handle overexposed images where
                 details may be washed out.
        
        Complexity: O(H*W) - processes all pixels once for brightness enhancement
        Relationship: Used for data augmentation and testing model robustness to bright lighting.
                     Complements _simulate_dim_lighting and _simulate_dark_lighting for comprehensive
                     lighting condition coverage.
        
        Args:
            image: PIL Image to brighten
            brightness_factor: Multiplier for brightness (default 1.5 = 50% brighter)
        
        Returns:
            Brightened PIL Image with increased brightness
        """
        from PIL import ImageEnhance
        
        # Use PIL's ImageEnhance for efficient brightness adjustment
        # Complexity: O(H*W) - applies brightness multiplier to all pixels
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(brightness_factor)  # Increase brightness by factor
        
        # Clamp values to valid range [0, 255] to prevent overflow
        # Note: PIL handles clamping automatically, but explicit conversion ensures correctness
        return brightened
    
    def _simulate_dim_lighting(self, image: Image.Image, brightness_factor: float = 0.6) -> Image.Image:
        """
        Simulate dim lighting conditions.
        
        Purpose: Augment images to simulate dim lighting scenarios (evening, cloudy day, dim indoor)
                 for training and evaluation. Helps model learn to handle low-light images where
                 details may be harder to distinguish.
        
        Complexity: O(H*W) - processes all pixels once for brightness reduction
        Relationship: Used for data augmentation and testing model robustness to dim lighting.
                     Part of lighting augmentation suite with _simulate_bright_lighting and
                     _simulate_dark_lighting.
        
        Args:
            image: PIL Image to dim
            brightness_factor: Multiplier for brightness (default 0.6 = 40% darker)
        
        Returns:
            Dimmed PIL Image with reduced brightness
        """
        from PIL import ImageEnhance
        
        # Use PIL's ImageEnhance for efficient brightness adjustment
        # Complexity: O(H*W) - applies brightness multiplier to all pixels
        enhancer = ImageEnhance.Brightness(image)
        dimmed = enhancer.enhance(brightness_factor)  # Reduce brightness by factor
        
        return dimmed
    
    def _simulate_dark_lighting(self, image: Image.Image, brightness_factor: float = 0.3) -> Image.Image:
        """
        Simulate very dark lighting conditions.
        
        Purpose: Augment images to simulate very dark lighting scenarios (night, dark room, low-light)
                 for training and evaluation. Critical for accessibility applications where users
                 may encounter night conditions. Tests model's ability to detect objects in extreme
                 low-light situations.
        
        Complexity: O(H*W) - processes all pixels once for brightness reduction and gamma correction
        Relationship: Used for data augmentation and testing model robustness to dark lighting.
                     Most extreme lighting condition, complements other lighting augmentations.
        
        Args:
            image: PIL Image to darken
            brightness_factor: Multiplier for brightness (default 0.3 = 70% darker)
        
        Returns:
            Darkened PIL Image with significantly reduced brightness
        """
        from PIL import ImageEnhance
        
        # First apply brightness reduction
        # Complexity: O(H*W) - applies brightness multiplier
        enhancer = ImageEnhance.Brightness(image)
        darkened = enhancer.enhance(brightness_factor)
        
        # Optionally apply gamma correction for more realistic dark lighting
        # Gamma correction: output = (input/255)^gamma * 255
        # Gamma > 1 darkens image, gamma < 1 brightens
        # For dark simulation, we use gamma = 2.0 to further darken mid-tones
        img_array = np.array(darkened).astype(np.float32)
        gamma = 2.0  # Darken mid-tones more aggressively
        img_array = np.power(img_array / 255.0, gamma) * 255.0
        img_array = np.clip(img_array, 0, 255)  # Clamp to valid range
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def preprocess_with_lighting(self, image: Image.Image) -> Dict[str, Any]:
        """
        Preprocess image and return both tensor and lighting metadata.
        
        Purpose: Enhanced preprocessing that includes lighting condition analysis. Returns both the
                 preprocessed image tensor and lighting classification, enabling lighting-aware
                 training and evaluation. Maintains backward compatibility with __call__ method.
        
        Complexity: O(H*W) - same as __call__ plus lighting analysis (both O(H*W))
        Relationship: Extends __call__ method to provide lighting metadata. Used by datasets to
                     include lighting information in training batches for lighting-stratified metrics.
        
        Args:
            image: PIL Image to preprocess
        
        Returns:
            Dictionary with:
                - 'image': torch.Tensor [3, H, W] - preprocessed image tensor
                - 'lighting': str - lighting condition ('bright', 'normal', 'dim', 'dark')
        """
        # Analyze lighting condition before preprocessing (preserves original brightness)
        # Complexity: O(H*W) - analyzes all pixels for brightness
        lighting = self._analyze_lighting_condition(image)
        
        # Apply condition-specific transforms if needed (same as __call__)
        # Complexity: O(H*W) - applies transforms to all pixels
        if self.condition_mode == 'cataracts':
            image = self._enhance_contrast(image)
        elif self.condition_mode == 'retinitis_pigmentosa':
            image = self._low_light_enhancement(image)
        elif self.condition_mode in ['myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors']:
            image = self._simulate_refractive_error(image)
        elif self.condition_mode == 'glaucoma':
            image = self._enhance_peripheral(image)
        elif self.condition_mode == 'amd':
            image = self._enhance_central(image)
        elif self.condition_mode == 'diabetic_retinopathy':
            image = self._enhance_edges(image)
        elif self.condition_mode == 'color_blindness':
            image = self._simulate_color_blindness(image)
        
        # Apply base transforms (resize, to tensor, normalize)
        # Complexity: O(H*W) - standard image transforms
        tensor = self.base_transform(image)  # type: ignore
        
        # Return both tensor and lighting metadata
        return {
            'image': tensor,
            'lighting': lighting
        }
    
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
    """Text region detection preprocessing for OCR integration. Uses model's text_head output."""
    
    def __init__(self, text_threshold: float = 0.5):
        """
        Initialize text region detector.
        
        Args:
            text_threshold: Confidence threshold for text detection
        """
        self.text_threshold = text_threshold
    
    def detect_text_regions(
        self,
        image: np.ndarray,
        text_scores: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None
    ) -> list:
        """
        Detect text regions in image using model's text_head output.
        
        Args:
            image: Image array [H, W, 3]
            text_scores: Text probability scores from model [N] (optional)
            boxes: Bounding boxes from model [N, 4] in center format (optional)
        
        Returns:
            List of bounding boxes [x, y, w, h] for text regions
        """
        # If model outputs are provided, use them
        if text_scores is not None and boxes is not None:
            text_mask = text_scores > self.text_threshold
            if text_mask.any():
                text_boxes = boxes[text_mask]
                # Convert from center format to corner format if needed
                results = []
                for box in text_boxes:
                    x, y, w, h = box.tolist()
                    results.append([x - w/2, y - h/2, w, h])
                return results
        
        # Fallback: simple edge-based detection (basic implementation)
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            h, w = image.shape[:2]
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                # Filter by aspect ratio (text is usually wider than tall)
                if bw > 10 and bh > 10 and bw / max(bh, 1) > 1.2:
                    # Normalize to [0, 1]
                    text_regions.append([x / w, y / h, bw / w, bh / h])
            
            return text_regions
        except ImportError:
            # OpenCV not available
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

