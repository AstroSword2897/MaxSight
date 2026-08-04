"""Tests for MVP advanced augmentation facade and geometry-aware boxes."""

from __future__ import annotations

import torch

from ml.data.advanced_augmentation import (
    AdvancedAugmentation,
    AugmentationConfig,
    ColorJitter,
    HorizontalFlip,
    Pipeline,
    RandomAffine,
    Sample,
    StressTestAugmentation,
    create_augmentation_pipeline,
)


def _rgb(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.rand(3, h, w)


def test_horizontal_flip_maps_boxes():
    image = _rgb(40, 80)
    boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0]])
    out = HorizontalFlip()(Sample(image=image, boxes=boxes))
    expected = torch.tensor([[50.0, 5.0, 70.0, 25.0]])
    assert out.boxes is not None
    assert torch.allclose(out.boxes, expected)


def test_affine_identity_preserves_boxes():
    image = _rgb()
    boxes = torch.tensor([[8.0, 10.0, 24.0, 30.0]])
    cfg = AugmentationConfig(
        rotation_range=(0, 0),
        scale_range=(1.0, 1.0),
        translate_range=(0.0, 0.0),
    )
    out = RandomAffine(cfg)(Sample(image=image, boxes=boxes.clone()))
    assert out.boxes is not None
    assert out.boxes.shape == boxes.shape
    assert torch.allclose(out.boxes, boxes, atol=1.5)


def test_affine_boxes_stay_inside_image():
    image = _rgb(64, 64)
    boxes = torch.tensor([[5.0, 5.0, 20.0, 20.0], [30.0, 30.0, 50.0, 55.0]])
    cfg = AugmentationConfig(rotation_range=(-20, 20), scale_range=(0.9, 1.1))
    out = RandomAffine(cfg)(Sample(image=image, boxes=boxes))
    assert out.boxes is not None
    assert (out.boxes[:, 0] >= 0).all()
    assert (out.boxes[:, 1] >= 0).all()
    assert (out.boxes[:, 2] <= 63).all()
    assert (out.boxes[:, 3] <= 63).all()
    assert ((out.boxes[:, 2] - out.boxes[:, 0]) > 0).all()
    assert ((out.boxes[:, 3] - out.boxes[:, 1]) > 0).all()


def test_affine_empty_boxes_ok():
    image = _rgb()
    boxes = torch.zeros(0, 4)
    out = RandomAffine(AugmentationConfig())(Sample(image=image, boxes=boxes))
    assert out.boxes is not None
    assert out.boxes.shape == (0, 4)


def test_photometric_leaves_boxes_unchanged():
    image = _rgb()
    boxes = torch.tensor([[4.0, 6.0, 18.0, 22.0]])
    out = ColorJitter(AugmentationConfig())(Sample(image=image, boxes=boxes.clone()))
    assert out.boxes is not None
    assert torch.equal(out.boxes, boxes)


def test_facade_preserves_unknown_keys_and_optional_boxes():
    aug = AdvancedAugmentation(
        AugmentationConfig(
            rotation_range=(0, 0),
            scale_range=(1, 1),
            translate_range=(0, 0),
            flip_horizontal_prob=0,
            flip_vertical_prob=0,
            motion_blur_prob=0,
            random_erasing_prob=0,
            cutout_prob=0,
            fog_prob=0,
            rain_prob=0,
            extreme_lighting_prob=0,
            partial_occlusion_prob=0,
            jpeg_compression_prob=0,
        )
    )
    image = _rgb()

    out_img, out_none = aug(image, None)
    assert out_img.shape == image.shape
    assert out_none is None

    meta = {"labels": torch.tensor([1, 2]), "custom": "keep"}
    out_img2, out_targets = aug(image, meta)
    assert out_targets is not None
    assert "boxes" not in out_targets
    assert out_targets["custom"] == "keep"
    assert torch.equal(out_targets["labels"], meta["labels"])

    with_boxes = {
        "boxes": torch.tensor([[2.0, 2.0, 10.0, 12.0]]),
        "extra": 7,
    }
    _, out_with = aug(image, with_boxes)
    assert out_with is not None
    assert "boxes" in out_with
    assert out_with["extra"] == 7


def test_create_augmentation_pipeline_modes():
    train = create_augmentation_pipeline("train")
    stress = create_augmentation_pipeline("stress_test")
    val = create_augmentation_pipeline("val")
    assert isinstance(train, AdvancedAugmentation)
    assert isinstance(stress, StressTestAugmentation)
    assert val.config.flip_horizontal_prob == 0
    assert val.config.motion_blur_prob == 0


def test_pipeline_runs_on_sample():
    pipeline = Pipeline(
        AugmentationConfig(
            flip_horizontal_prob=0,
            flip_vertical_prob=0,
            motion_blur_prob=0,
            random_erasing_prob=0,
            cutout_prob=0,
            fog_prob=0,
            rain_prob=0,
            extreme_lighting_prob=0,
            partial_occlusion_prob=0,
            jpeg_compression_prob=0,
            rotation_range=(0, 0),
            scale_range=(1, 1),
            translate_range=(0, 0),
        )
    )
    sample = Sample(image=_rgb(), boxes=torch.tensor([[1.0, 1.0, 8.0, 8.0]]))
    out = pipeline(sample)
    assert out.image.shape == sample.image.shape
    assert out.boxes is not None
