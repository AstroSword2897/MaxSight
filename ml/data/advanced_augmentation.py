"""Advanced data augmentation for real-world robustness.

Box contract (MVP): absolute xyxy in pixel space matching image H,W.
Geometric transforms update boxes; photometric/noise/weather leave them unchanged.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""

    # Geometric.
    rotation_range: Tuple[float, float] = (-30, 30)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translate_range: Tuple[float, float] = (-0.1, 0.1)
    perspective_strength: float = 0.2
    flip_horizontal_prob: float = 0.5
    flip_vertical_prob: float = 0.1

    # Photometric.
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    gamma_range: Tuple[float, float] = (0.8, 1.2)

    # Noise.
    gaussian_noise_std: float = 0.05
    salt_pepper_prob: float = 0.02
    motion_blur_kernel: int = 7
    motion_blur_prob: float = 0.3

    # Occlusion.
    random_erasing_prob: float = 0.3
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2)
    cutout_prob: float = 0.2
    cutout_size: int = 32

    # Weather.
    fog_prob: float = 0.1
    rain_prob: float = 0.1
    snow_prob: float = 0.05

    # Camera artifacts.
    lens_distortion_prob: float = 0.1
    jpeg_compression_prob: float = 0.2
    jpeg_quality_range: Tuple[int, int] = (50, 95)

    # Edge cases.
    extreme_lighting_prob: float = 0.15
    partial_occlusion_prob: float = 0.25


@dataclass
class Sample:
    """Single-image sample. Boxes are absolute xyxy pixels when present."""

    image: torch.Tensor
    boxes: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class Transform(Protocol):
    def __call__(self, sample: Sample) -> Sample: ...


def _image_hw(image: torch.Tensor) -> Tuple[int, int]:
    return int(image.shape[-2]), int(image.shape[-1])


def _drop_degenerate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > 0) & (heights > 0)
    return boxes[keep]


def _transform_boxes_affine(
    boxes: torch.Tensor,
    theta: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map xyxy boxes through inverse of affine_grid theta (align_corners=False)."""
    if boxes.numel() == 0:
        return boxes

    x1, y1, x2, y2 = boxes.unbind(-1)
    corners = torch.stack(
        [
            torch.stack([x1, y1], dim=-1),
            torch.stack([x2, y1], dim=-1),
            torch.stack([x2, y2], dim=-1),
            torch.stack([x1, y2], dim=-1),
        ],
        dim=1,
    )

    # Pixel -> normalized coords matching F.affine_grid(..., align_corners=False).
    xn = (corners[..., 0] + 0.5) / width * 2.0 - 1.0
    yn = (corners[..., 1] + 0.5) / height * 2.0 - 1.0
    p_in = torch.stack([xn, yn], dim=-1)

    # grid_sample samples input at A @ p_out + t; invert for box corners in input space.
    a_mat = theta[:, :2]
    t_vec = theta[:, 2]
    a_inv = torch.linalg.inv(a_mat)
    delta = p_in - t_vec.view(1, 1, 2)
    p_out = torch.matmul(delta, a_inv.T)

    x = (p_out[..., 0] + 1.0) / 2.0 * width - 0.5
    y = (p_out[..., 1] + 1.0) / 2.0 * height - 0.5

    x_min = x.min(dim=-1).values.clamp(0, max(width - 1, 0))
    y_min = y.min(dim=-1).values.clamp(0, max(height - 1, 0))
    x_max = x.max(dim=-1).values.clamp(0, max(width - 1, 0))
    y_max = y.max(dim=-1).values.clamp(0, max(height - 1, 0))

    out = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    return _drop_degenerate_boxes(out)


class HorizontalFlip:
    def __call__(self, sample: Sample) -> Sample:
        image = torch.flip(sample.image, dims=[-1])
        boxes = sample.boxes
        if boxes is not None and len(boxes) > 0:
            width = image.shape[-1]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return Sample(image=image, boxes=boxes, labels=sample.labels)


class VerticalFlip:
    def __call__(self, sample: Sample) -> Sample:
        image = torch.flip(sample.image, dims=[-2])
        boxes = sample.boxes
        if boxes is not None and len(boxes) > 0:
            height = image.shape[-2]
            boxes = boxes.clone()
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        return Sample(image=image, boxes=boxes, labels=sample.labels)


class RandomAffine:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        angle = random.uniform(*self.config.rotation_range)
        scale = random.uniform(*self.config.scale_range)
        tx = random.uniform(*self.config.translate_range)
        ty = random.uniform(*self.config.translate_range)

        theta = torch.tensor(
            [
                [
                    scale * math.cos(math.radians(angle)),
                    -scale * math.sin(math.radians(angle)),
                    tx,
                ],
                [
                    scale * math.sin(math.radians(angle)),
                    scale * math.cos(math.radians(angle)),
                    ty,
                ],
            ],
            dtype=image.dtype,
            device=image.device,
        )

        squeeze = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True

        # list(...) satisfies affine_grid's size typing (torch.Size is rejected).
        size = list(image.size())
        grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=False)
        image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode="bilinear",
            padding_mode="reflection",
        )

        if squeeze:
            image = image.squeeze(0)

        boxes = sample.boxes
        if boxes is not None:
            height, width = _image_hw(image)
            boxes = _transform_boxes_affine(boxes, theta, height, width)

        return Sample(image=image, boxes=boxes, labels=sample.labels)


class ColorJitter:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        brightness = random.uniform(*self.config.brightness_range)
        image = image * brightness

        contrast = random.uniform(*self.config.contrast_range)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast + mean

        if image.shape[-3] == 3:
            saturation = random.uniform(*self.config.saturation_range)
            gray = image.mean(dim=-3, keepdim=True)
            image = (image - gray) * saturation + gray

        image = image.clamp(0, 1)
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class ExtremeLighting:
    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        condition = random.choice(["overexposed", "underexposed", "harsh_shadows"])

        if condition == "overexposed":
            image = (image * random.uniform(1.5, 2.5)).clamp(0, 1)
        elif condition == "underexposed":
            image = image * random.uniform(0.2, 0.5)
        else:
            h, w = _image_hw(image)
            shadow = torch.linspace(0.3, 1.0, w, device=image.device)
            shadow = shadow.reshape(1, 1, 1, w).expand_as(image)
            image = image * shadow

        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class GaussianNoise:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        noise = torch.randn_like(sample.image) * self.config.gaussian_noise_std
        image = (sample.image + noise).clamp(0, 1)
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class SaltPepperNoise:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image.clone()
        mask = torch.rand_like(image)
        salt_mask = mask < self.config.salt_pepper_prob / 2
        pepper_mask = mask > (1 - self.config.salt_pepper_prob / 2)
        image[salt_mask] = 1.0
        image[pepper_mask] = 0.0
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class MotionBlur:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        kernel_size = self.config.motion_blur_kernel
        angle = random.uniform(0, 360)

        kernel = torch.zeros(kernel_size, kernel_size, device=image.device)
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * math.cos(math.radians(angle)))
            y = int(center + offset * math.sin(math.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, kernel_size, kernel_size)

        squeeze = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True

        channels = image.shape[1]
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        image = F.conv2d(image, kernel, padding=padding, groups=channels)

        if squeeze:
            image = image.squeeze(0)

        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class RandomErasing:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        h, w = _image_hw(image)
        area = h * w

        for _ in range(random.randint(1, 3)):
            target_area = random.uniform(*self.config.random_erasing_scale) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))
            if eh < h and ew < w:
                x = random.randint(0, w - ew)
                y = random.randint(0, h - eh)
                if random.random() < 0.5:
                    image[..., y : y + eh, x : x + ew] = torch.rand_like(
                        image[..., y : y + eh, x : x + ew]
                    )
                else:
                    image[..., y : y + eh, x : x + ew] = image.mean()

        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class Cutout:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image.clone()
        h, w = _image_hw(image)
        size = self.config.cutout_size
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        x1 = max(0, x - size // 2)
        x2 = min(w, x + size // 2)
        y1 = max(0, y - size // 2)
        y2 = min(h, y + size // 2)
        image[..., y1:y2, x1:x2] = 0
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class PartialOcclusion:
    def __call__(self, sample: Sample) -> Sample:
        image = sample.image.clone()
        h, w = _image_hw(image)
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(h // 8, h // 3)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )
        dist = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        occlusion_mask = dist < radius
        occlusion_color = random.uniform(0.1, 0.3)
        image[..., occlusion_mask] = occlusion_color
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class FogEffect:
    def __call__(self, sample: Sample) -> Sample:
        fog_intensity = random.uniform(0.3, 0.7)
        fog_color = torch.ones_like(sample.image) * 0.8
        image = sample.image * (1 - fog_intensity) + fog_color * fog_intensity
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class RainEffect:
    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        h, w = _image_hw(image)
        num_drops = random.randint(50, 200)
        rain_layer = torch.zeros_like(image)

        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(3, 10)
            for i in range(length):
                ny = min(y + i, h - 1)
                if 0 <= ny < h and 0 <= x < w:
                    rain_layer[..., ny, x] = 0.7

        image = (image * 0.8 + rain_layer * 0.2).clamp(0, 1)
        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class JpegCompression:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image
        quality = random.randint(*self.config.jpeg_quality_range)
        block_size = 8
        h, w = _image_hw(image)
        new_h = (h // block_size) * block_size
        new_w = (w // block_size) * block_size

        if new_h > 0 and new_w > 0:
            image_reshaped = image[..., :new_h, :new_w]
            image_reshaped = image_reshaped.reshape(
                *image.shape[:-2],
                new_h // block_size,
                block_size,
                new_w // block_size,
                block_size,
            )
            block_means = image_reshaped.mean(dim=(-3, -1), keepdim=True)
            noise_scale = (100 - quality) / 500
            noise = torch.randn_like(block_means) * noise_scale
            block_means = (block_means + noise).clamp(0, 1)
            image_reshaped = block_means.expand_as(image_reshaped)
            image = image.clone()
            image[..., :new_h, :new_w] = image_reshaped.reshape(*image.shape[:-2], new_h, new_w)

        return Sample(image=image, boxes=sample.boxes, labels=sample.labels)


class Pipeline:
    """Probabilistic compose matching prior AdvancedAugmentation selection."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def select(self) -> List[Transform]:
        cfg = self.config
        transforms: List[Transform] = []

        if random.random() < cfg.flip_horizontal_prob:
            transforms.append(HorizontalFlip())
        if random.random() < cfg.flip_vertical_prob:
            transforms.append(VerticalFlip())

        transforms.append(RandomAffine(cfg))
        transforms.append(ColorJitter(cfg))

        if random.random() < cfg.motion_blur_prob:
            transforms.append(MotionBlur(cfg))
        if random.random() < 0.3:
            transforms.append(GaussianNoise(cfg))
        if random.random() < cfg.random_erasing_prob:
            transforms.append(RandomErasing(cfg))
        if random.random() < cfg.cutout_prob:
            transforms.append(Cutout(cfg))
        if random.random() < cfg.fog_prob:
            transforms.append(FogEffect())
        if random.random() < cfg.rain_prob:
            transforms.append(RainEffect())
        if random.random() < cfg.extreme_lighting_prob:
            transforms.append(ExtremeLighting())
        if random.random() < cfg.partial_occlusion_prob:
            transforms.append(PartialOcclusion())
        if random.random() < cfg.jpeg_compression_prob:
            transforms.append(JpegCompression(cfg))

        return transforms

    def __call__(self, sample: Sample, transforms: Optional[Sequence[Transform]] = None) -> Sample:
        for transform in transforms if transforms is not None else self.select():
            sample = transform(sample)
        return sample


def _targets_to_sample(
    image: torch.Tensor, targets: Optional[Dict]
) -> Tuple[Sample, bool]:
    """Build Sample from facade targets. Returns (sample, had_boxes_key)."""
    if targets is None or "boxes" not in targets:
        return Sample(image=image), False
    boxes = targets["boxes"]
    labels = targets.get("labels") if isinstance(targets.get("labels"), torch.Tensor) else None
    return Sample(image=image, boxes=boxes, labels=labels), True


def _sample_to_targets(
    sample: Sample,
    targets: Optional[Dict],
    had_boxes: bool,
) -> Optional[Dict]:
    """Write boxes back only when the key was originally present."""
    if targets is None:
        return None
    if had_boxes:
        targets["boxes"] = sample.boxes
    return targets


class AdvancedAugmentation:
    """Facade over Pipeline. Signature and target-key behavior preserved."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.pipeline = Pipeline(self.config)

    def __call__(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = self.pipeline(sample)
        targets = _sample_to_targets(sample, targets, had_boxes)
        return sample.image, targets

    # Compatibility wrappers for direct method calls.
    def horizontal_flip(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = HorizontalFlip()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def vertical_flip(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = VerticalFlip()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def random_affine(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = RandomAffine(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def color_jitter(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = ColorJitter(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def extreme_lighting(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = ExtremeLighting()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def gaussian_noise(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = GaussianNoise(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def salt_pepper_noise(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = SaltPepperNoise(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def motion_blur(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = MotionBlur(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def random_erasing(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = RandomErasing(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def cutout(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = Cutout(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def partial_occlusion(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = PartialOcclusion()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def fog_effect(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = FogEffect()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def rain_effect(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = RainEffect()(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)

    def jpeg_compression(
        self, image: torch.Tensor, targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        sample, had_boxes = _targets_to_sample(image, targets)
        sample = JpegCompression(self.config)(sample)
        return sample.image, _sample_to_targets(sample, targets, had_boxes)


class StressTestAugmentation(AdvancedAugmentation):
    """Aggressive transforms for edge-case robustness testing."""

    def __init__(self):
        config = AugmentationConfig(
            rotation_range=(-45, 45),
            scale_range=(0.5, 1.5),
            brightness_range=(0.3, 2.0),
            contrast_range=(0.3, 2.0),
            gaussian_noise_std=0.15,
            motion_blur_prob=0.5,
            random_erasing_prob=0.5,
            random_erasing_scale=(0.1, 0.4),
            fog_prob=0.3,
            rain_prob=0.2,
            extreme_lighting_prob=0.4,
            partial_occlusion_prob=0.4,
            jpeg_compression_prob=0.4,
            jpeg_quality_range=(20, 70),
        )
        super().__init__(config)


class MixUp:
    """MixUp augmentation for regularization."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, labels, labels[index], lam


class CutMix:
    """CutMix augmentation for regularization."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        h, w = images.shape[-2:]
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))

        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))

        mixed_images = images.clone()
        mixed_images[..., y1:y2, x1:x2] = images[index, ..., y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
        return mixed_images, labels, labels[index], float(lam)


def create_augmentation_pipeline(mode: str = "train") -> AdvancedAugmentation:
    """Create augmentation pipeline based on mode: train, val, test, or stress_test."""
    if mode == "train":
        return AdvancedAugmentation()
    if mode == "stress_test":
        return StressTestAugmentation()
    return AdvancedAugmentation(
        AugmentationConfig(
            rotation_range=(0, 0),
            scale_range=(1, 1),
            flip_horizontal_prob=0,
            flip_vertical_prob=0,
            motion_blur_prob=0,
            random_erasing_prob=0,
            fog_prob=0,
            rain_prob=0,
            extreme_lighting_prob=0,
            partial_occlusion_prob=0,
            cutout_prob=0,
            jpeg_compression_prob=0,
        )
    )
