"""Multi-Modal Augmentation for MaxSight 3.0."""

import torch
import torchaudio.transforms as AT
import torchvision.transforms as T

_UINT8_MAX = 255.0


class MultiModalAugmentation:
    """Augmentations for vision and audio.

    Image: one ``[C, H, W]`` tensor, either uint8 or float in ``[0, 1]`` (before mean/std
    normalisation); the same dtype comes back. The vision chain runs on uint8 because RandomEqualize
    accepts nothing else (float input used to crash about half of all calls).
    Audio: a spectrogram or MFCC with at least ``(freq, time)`` dimensions, e.g. ``[1, 13, T]`` from
    ``ml/data/dataset.py``; time masking is meaningless on a raw 1-D waveform, so that is refused.
    """

    def __init__(self):
        self.vision_aug = T.Compose(
            [
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.RandomAutocontrast(),
                T.RandomEqualize(),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.3, contrast=0.3),
            ]
        )
        self.audio_aug = AT.TimeMasking(time_mask_param=10)

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            return self.vision_aug(image)
        if not image.is_floating_point():
            raise ValueError(f"image must be uint8 or float in [0, 1], got {image.dtype}")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            # Clamping a mean/std-normalised image would silently destroy it: refuse instead.
            raise ValueError("float image must be in [0, 1]; augment before mean/std normalisation")
        as_uint8 = (image * _UINT8_MAX).round().to(torch.uint8)
        return self.vision_aug(as_uint8).to(image.dtype) / _UINT8_MAX

    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() < 2:
            raise ValueError(
                f"audio must be a spectrogram/MFCC with (freq, time) dims, got shape {tuple(audio.shape)}"
            )
        return self.audio_aug(audio)

    def __call__(
        self, image: torch.Tensor, audio: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply augmentations."""
        aug_audio = self._augment_audio(audio) if audio is not None else None
        return self._augment_image(image), aug_audio
