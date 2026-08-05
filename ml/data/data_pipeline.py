"""Data pipeline for MaxSight training."""

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ml.data.dataset import MaxSightDataset
from ml.data.dataset_registry import DatasetRegistry
from ml.data.gold.dataset import GoldManifestDataset

if TYPE_CHECKING:
    from ml.training.run_config import ResolvedTrainingConfig


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Custom collate function for MaxSight batches. Handles variable-length objects, optional audio, and optional frame sequences."""
    # Sequence mode uses 'frames' as primary image tensor; otherwise use single-frame 'images'.
    has_frames = any("frames" in item for item in batch)
    if has_frames:
        frame_lists = [item["frames"] for item in batch]
        batch_size = len(frame_lists)
        max_t = max(frames.shape[0] for frames in frame_lists)
        c, h, w = frame_lists[0].shape[1:]
        frames_batch = torch.zeros(batch_size, max_t, c, h, w, dtype=frame_lists[0].dtype)
        frame_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, frames in enumerate(frame_lists):
            t = frames.shape[0]
            frames_batch[i, :t] = frames
            frame_lengths[i] = t
        images = frames_batch
    else:
        images = torch.stack([item["images"] for item in batch])
        batch_size = len(batch)
        frame_lengths = None

    max_objects = max(item["num_objects"].item() for item in batch) if batch else 10
    max_objects = min(max_objects, batch[0].get("labels", torch.zeros(10)).shape[0])

    labels = torch.zeros(batch_size, max_objects, dtype=torch.long)
    boxes = torch.zeros(batch_size, max_objects, 4, dtype=torch.float32)
    distance = torch.zeros(batch_size, max_objects, dtype=torch.long)
    num_objects = torch.zeros(batch_size, dtype=torch.long)
    urgency = torch.zeros(batch_size, dtype=torch.long)

    has_audio = any("audio" in item for item in batch)
    audio_tensors: list[torch.Tensor] = []
    audio_lengths: list[int] = []

    for i, item in enumerate(batch):
        num_obj = item["num_objects"].item()
        num_objects[i] = num_obj

        if num_obj > 0:
            labels[i, :num_obj] = item["labels"][:num_obj]
            item_boxes = item["boxes"][:num_obj].clone()
            item_boxes[:, 2] = torch.clamp(item_boxes[:, 2], min=1e-4)
            item_boxes[:, 3] = torch.clamp(item_boxes[:, 3], min=1e-4)
            boxes[i, :num_obj] = item_boxes
            distance[i, :num_obj] = item["distance"][:num_obj]

        urgency[i] = item["urgency"]

        if has_audio and "audio" in item:
            audio = item["audio"]
            audio_tensors.append(audio.squeeze(0))
            audio_lengths.append(audio.shape[-1])
        elif has_audio:
            audio_tensors.append(torch.zeros(13, 100))
            audio_lengths.append(100)

    result: dict[str, Any] = {
        "images": images,
        "labels": labels,
        "boxes": boxes,
        "distance": distance,
        "num_objects": num_objects,
        "urgency": urgency,
    }

    if has_frames:
        result["frame_lengths"] = frame_lengths

    if has_audio and audio_tensors:
        max_audio_len = max(audio_lengths) if audio_lengths else 100
        padded_audio = torch.zeros(batch_size, 13, max_audio_len)
        for i, audio in enumerate(audio_tensors):
            padded_audio[i, :, : audio.shape[-1]] = audio
        result["audio"] = padded_audio
        result["audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)

    if "condition_mode" in batch[0]:
        result["condition_mode"] = batch[0]["condition_mode"]

    if has_frames and all("temporal_consistency" in item for item in batch):
        result["temporal_consistency"] = torch.stack(
            [item["temporal_consistency"].reshape(-1)[0] for item in batch]
        )
    if has_frames and all("flicker" in item for item in batch):
        result["flicker"] = torch.stack([item["flicker"].reshape(-1)[0] for item in batch])
    if all("clip_id" in item for item in batch):
        result["clip_ids"] = [str(item["clip_id"]) for item in batch]

    if all("dataset_source" in item for item in batch):
        result["dataset_source"] = [str(item["dataset_source"]) for item in batch]

    return result


def create_data_loaders(
    train_annotation_file: Path,
    val_annotation_file: Path,
    test_annotation_file: Path | None = None,
    image_dir: Path | None = None,
    train_image_dir: Path | None = None,
    val_image_dir: Path | None = None,
    test_image_dir: Path | None = None,
    audio_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    condition_mode: str | None = None,
    tag_lighting_metadata: bool = True,
    lighting_pixel_augmentation: bool = False,
    max_objects: int = 10,
    shuffle_train: bool = True,
    drop_last: bool = False,
    use_weighted_sampling: bool = False,
    class_weights: dict[int, float] | None = None,
    strict_images: bool = True,
    dataset_source_key: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train/val/test data loaders for MaxSight training.

    Per-split image_dir overrides (train_image_dir, val_image_dir, test_image_dir)
    take precedence over the shared image_dir fallback. This is required for
    synthetic datasets whose image files live under split-specific roots.
    strict_images=True (the default) raises on missing image files so corrupt
    paths fail fast instead of silently training on random noise.
    """

    # Resolve per-split image directories.
    # Priority: explicit per-split arg > shared image_dir > auto-detect fallback.
    def _resolve_image_dir(per_split: Path | None, shared: Path | None, subdir: str) -> Path | None:
        if per_split is not None:
            return per_split
        if shared is not None:
            candidate = shared / subdir
            return candidate if candidate.exists() else shared
        return None

    if image_dir is None and train_image_dir is None:
        # Auto-detect for real-world COCO layout.
        for candidate in [
            train_annotation_file.parent.parent / "train2017",
            train_annotation_file.parent.parent.parent / "coco_raw" / "train2017",
            train_annotation_file.parent.parent / "images",
        ]:
            if candidate.exists():
                image_dir = candidate.parent
                break
        if image_dir is None:
            image_dir = train_annotation_file.parent.parent

    eff_train_dir = _resolve_image_dir(train_image_dir, image_dir, "train2017")
    eff_val_dir = _resolve_image_dir(val_image_dir, image_dir, "val2017")
    eff_test_dir = _resolve_image_dir(test_image_dir, image_dir, "val2017")

    # Create datasets.
    train_dataset = MaxSightDataset(
        data_dir=eff_train_dir or train_annotation_file.parent,
        annotation_file=train_annotation_file,
        image_dir=eff_train_dir,
        audio_dir=audio_dir,
        condition_mode=condition_mode,
        tag_lighting_metadata=tag_lighting_metadata,
        lighting_pixel_augmentation=lighting_pixel_augmentation,
        max_objects=max_objects,
        strict_images=strict_images,
        dataset_source_key=dataset_source_key,
    )

    val_dataset = MaxSightDataset(
        data_dir=eff_val_dir or val_annotation_file.parent,
        annotation_file=val_annotation_file,
        image_dir=eff_val_dir,
        audio_dir=audio_dir,
        condition_mode=None,
        tag_lighting_metadata=False,
        lighting_pixel_augmentation=False,
        max_objects=max_objects,
        strict_images=strict_images,
        dataset_source_key=dataset_source_key,
    )

    test_dataset = None
    if test_annotation_file and test_annotation_file.exists():
        test_dataset = MaxSightDataset(
            data_dir=eff_test_dir or test_annotation_file.parent,
            annotation_file=test_annotation_file,
            image_dir=eff_test_dir,
            audio_dir=audio_dir,
            condition_mode=None,
            tag_lighting_metadata=False,
            lighting_pixel_augmentation=False,
            max_objects=max_objects,
            strict_images=strict_images,
            dataset_source_key=dataset_source_key,
        )

    # Create samplers if using weighted sampling.
    train_sampler = None
    if use_weighted_sampling and class_weights:
        # Compute sample weights based on class distribution.
        sample_weights = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            labels = sample["labels"]
            num_obj = sample["num_objects"].item()

            # Weight by most frequent class in sample.
            if num_obj > 0:
                class_idx = labels[0].item()
                weight = class_weights.get(class_idx, 1.0)
            else:
                weight = 1.0

            sample_weights.append(weight)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling.

    # Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train and train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader


def _gold_shard_uris_from_cfg(
    meta_path: str | None,
    shard_paths: tuple[str, ...] | None,
    repo_root: Path,
    split_label: str,
) -> tuple[list[str], dict[str, Any] | None]:
    """Resolve shard URIs + optional meta dict for a single gold split.

    Priority: meta file (self-describing artifact) > explicit shard paths.
    When meta is given, shard URIs come from meta["shards"][].uri resolved
    relative to the meta file's directory so the artifact is portable.
    """
    if meta_path:
        from ml.data.gold.dataset import load_gold_meta

        meta_file = Path(meta_path)
        meta = load_gold_meta(meta_file)
        meta_dir = meta_file.parent
        uris: list[str] = []
        for shard in meta.get("shards", []):
            uri = shard["uri"]
            if uri.startswith("s3://") or Path(uri).is_absolute():
                uris.append(uri)
            else:
                # Relative URI → resolve against artifact directory.
                uris.append(str((meta_dir / uri).resolve()))
        if not uris:
            raise ValueError(f"{split_label}: gold meta at {meta_path} has no shard entries")
        return uris, meta

    if shard_paths:
        uris = []
        for p in shard_paths:
            if p.startswith("s3://") or Path(p).is_absolute():
                uris.append(p)
            else:
                uris.append(str((repo_root / p).resolve()))
        return uris, None

    raise ValueError(
        f"{split_label}: gold mode requires either a meta file or explicit shard paths"
    )


def _create_gold_data_loaders_for_resolved(
    resolved: "ResolvedTrainingConfig",
    *,
    repo_root: Path,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Train/val/test loaders from gold artifact shards — no registry call.

    Identity and invariants (num_classes, label_space) come from the artifact
    meta when present; otherwise they fall back to model/training config values.
    """
    cfg = resolved.data
    pin = cfg.pin_memory and device == "cuda"

    train_uris, train_meta = _gold_shard_uris_from_cfg(
        cfg.gold_train_meta,
        cfg.gold_train_shard_paths,
        repo_root,
        "data.gold_train",
    )
    val_uris, val_meta = _gold_shard_uris_from_cfg(
        cfg.gold_val_meta,
        cfg.gold_val_shard_paths,
        repo_root,
        "data.gold_val",
    )

    num_classes = resolved.model.num_classes
    label_space = resolved.training.label_space

    train_ds = GoldManifestDataset(
        train_uris,
        meta=train_meta,
        repo_root=repo_root,
        max_objects=cfg.max_objects,
        condition_mode=cfg.condition_mode,
        tag_lighting_metadata=cfg.tag_lighting_metadata,
        strict_images=True,
        expected_label_space=label_space,
        num_classes=num_classes,
    )
    val_ds = GoldManifestDataset(
        val_uris,
        meta=val_meta,
        repo_root=repo_root,
        max_objects=cfg.max_objects,
        strict_images=True,
        expected_label_space=label_space,
        num_classes=num_classes,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=cfg.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader: DataLoader | None = None
    if cfg.gold_test_meta or cfg.gold_test_shard_paths:
        test_uris, test_meta = _gold_shard_uris_from_cfg(
            cfg.gold_test_meta,
            cfg.gold_test_shard_paths,
            repo_root,
            "data.gold_test",
        )
        test_ds = GoldManifestDataset(
            test_uris,
            meta=test_meta,
            repo_root=repo_root,
            max_objects=cfg.max_objects,
            strict_images=True,
            expected_label_space=label_space,
            num_classes=num_classes,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            collate_fn=collate_fn,
            drop_last=False,
        )
    return train_loader, val_loader, test_loader


def create_data_loaders_for_resolved(
    resolved: "ResolvedTrainingConfig",
    *,
    registry: DatasetRegistry,
    repo_root: Path,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build loaders from ResolvedTrainingConfig + registry (single source or composition).

    Single-source uses ``data.*`` paths only after run_config has checked they match
    the registry. Multi-source concatenates per-source ``MaxSightDataset`` instances
    and mixes training batches with a ``WeightedRandomSampler`` proportional to
    ``dataset.sources[].weight``.
    """
    cfg = resolved.data
    if cfg.data_plane == "gold":
        return _create_gold_data_loaders_for_resolved(resolved, repo_root=repo_root, device=device)
    sources = resolved.dataset.sources
    if sources:
        if cfg.use_weighted_sampling:
            raise ValueError(
                "data.use_weighted_sampling is incompatible with dataset.sources; "
                "disable weighted sampling or use a single dataset."
            )
        train_parts: list[MaxSightDataset] = []
        val_parts: list[MaxSightDataset] = []
        weight_list: list[float] = []
        for src in sources:
            entry = registry.resolve(
                src.dataset_id,
                src.dataset_version,
                tier=resolved.model.tier,
                require_active=True,
            )
            weight_list.append(float(src.weight))
            ta = repo_root / (entry.annotation_path("train") or "")
            va = repo_root / (entry.annotation_path("val") or "")
            tr_img = entry.resolved_image_dir("train")
            vl_img = entry.resolved_image_dir("val")
            train_parts.append(
                MaxSightDataset(
                    data_dir=(repo_root / tr_img) if tr_img else ta.parent,
                    annotation_file=ta,
                    image_dir=(repo_root / tr_img) if tr_img else None,
                    audio_dir=Path(cfg.audio_dir) if cfg.audio_dir else None,
                    condition_mode=cfg.condition_mode,
                    tag_lighting_metadata=cfg.tag_lighting_metadata,
                    lighting_pixel_augmentation=cfg.lighting_pixel_augmentation,
                    max_objects=cfg.max_objects,
                    strict_images=True,
                    dataset_source_key=entry.key,
                )
            )
            val_parts.append(
                MaxSightDataset(
                    data_dir=(repo_root / vl_img) if vl_img else va.parent,
                    annotation_file=va,
                    image_dir=(repo_root / vl_img) if vl_img else None,
                    audio_dir=Path(cfg.audio_dir) if cfg.audio_dir else None,
                    condition_mode=None,
                    tag_lighting_metadata=False,
                    lighting_pixel_augmentation=False,
                    max_objects=cfg.max_objects,
                    strict_images=True,
                    dataset_source_key=entry.key,
                )
            )
        s = sum(weight_list)
        if abs(s - 1.0) > 1e-4:
            raise ValueError(f"dataset.sources weights must sum to 1.0, got {s}")
        train_c = ConcatDataset(train_parts)
        val_c = ConcatDataset(val_parts)
        sample_weights: list[float] = []
        for ds, w in zip(train_parts, weight_list):
            wi = w / len(ds)
            sample_weights.extend([wi] * len(ds))
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        pin = cfg.pin_memory and device == "cuda"
        train_loader = DataLoader(
            train_c,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            collate_fn=collate_fn,
            drop_last=cfg.drop_last,
        )
        val_loader = DataLoader(
            val_c,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            collate_fn=collate_fn,
            drop_last=False,
        )
        # Test split is omitted for composition until every source exposes test with
        # compatible layout; mixed eval semantics are not defined yet.
        test_loader: DataLoader | None = None
        return train_loader, val_loader, test_loader

    assert resolved.dataset.dataset_id is not None and resolved.dataset.dataset_version is not None
    entry = registry.resolve(
        resolved.dataset.dataset_id,
        resolved.dataset.dataset_version,
        tier=resolved.model.tier,
        require_active=True,
    )
    train_ann = repo_root / (entry.annotation_path("train") or "")
    val_ann = repo_root / (entry.annotation_path("val") or "")
    test_ann_rel = entry.annotation_path("test")
    test_ann = (repo_root / test_ann_rel) if test_ann_rel else None
    tr_img = entry.resolved_image_dir("train")
    vl_img = entry.resolved_image_dir("val")
    tst_img = entry.resolved_image_dir("test")
    return create_data_loaders(
        train_annotation_file=train_ann,
        val_annotation_file=val_ann,
        test_annotation_file=test_ann if test_ann and test_ann.exists() else None,
        image_dir=Path(cfg.image_dir) if cfg.image_dir else None,
        train_image_dir=(repo_root / tr_img) if tr_img else None,
        val_image_dir=(repo_root / vl_img) if vl_img else None,
        test_image_dir=(repo_root / tst_img) if tst_img else None,
        audio_dir=Path(cfg.audio_dir) if cfg.audio_dir else None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device == "cuda",
        condition_mode=cfg.condition_mode,
        tag_lighting_metadata=cfg.tag_lighting_metadata,
        lighting_pixel_augmentation=cfg.lighting_pixel_augmentation,
        max_objects=cfg.max_objects,
        shuffle_train=cfg.shuffle_train,
        drop_last=cfg.drop_last,
        use_weighted_sampling=cfg.use_weighted_sampling,
        class_weights=cfg.class_weights,
        strict_images=True,
        dataset_source_key=entry.key,
    )


def compute_class_weights(annotation_file: Path) -> dict[int, float]:
    """Compute class weights from annotations for handling class imbalance. Returns: Dictionary mapping class_idx -> weight (inverse frequency)"""
    with open(annotation_file) as f:
        data = json.load(f)

    # Count class frequencies.
    class_counts = defaultdict(int)
    total_objects = 0

    if "images" in data and "annotations" in data:
        # COCO format.
        for ann in data["annotations"]:
            category_id = ann.get("category_id", 0)
            class_counts[category_id] += 1
            total_objects += 1
    else:
        # Custom format.
        for ann in data:
            for obj in ann.get("objects", []):
                class_idx = obj.get("class", 0)
                class_counts[class_idx] += 1
                total_objects += 1

    # Compute inverse frequency weights.
    if total_objects == 0:
        return {}

    class_weights = {}
    for class_idx, count in class_counts.items():
        # Inverse frequency: more frequent = lower weight.
        class_weights[class_idx] = total_objects / (len(class_counts) * count)

    return class_weights


def get_data_info(loader: DataLoader) -> dict[str, Any]:
    """Get information about a data loader (dataset size, batch count, etc.). Returns: Dictionary with dataset statistics."""
    from collections.abc import Sized

    dataset = loader.dataset
    batch_size = loader.batch_size
    # Dataset protocol is not Sized; only report length when __len__ exists.
    dataset_size: int | str = len(dataset) if isinstance(dataset, Sized) else "unknown"

    info = {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "num_batches": len(loader),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }

    # Sample a batch to get tensor shapes.
    try:
        sample_batch = next(iter(loader))
        info["batch_shapes"] = {
            key: list(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            for key, value in sample_batch.items()
        }
    except Exception:
        info["batch_shapes"] = "Unable to sample batch"

    return info
