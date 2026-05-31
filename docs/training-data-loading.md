# Training data loading

This document describes how training data is loaded, batched, and fed to the model: dataset class, data pipeline, and preprocessing.

## Dataset

- **`ml/data/dataset.py`:** Defines **MaxSightDataset**, the main dataset used for training. It loads images and annotations (COCO-format JSON), applies preprocessing, and returns items with keys such as `images`, `labels`, `boxes`, `distance`, `urgency`, `num_objects`, and optionally `audio` and `condition_mode`.
- **Annotations:** Expected format is COCO-style for boxes/labels, but the dataset layer also supports COCO panoptic-style supervision.
  - Panoptic meaning: panoptic annotations provide `segments_info` (segment ids + categories). `ml/data/dataset.py` derives per-segment bounding boxes and uses segment category/metadata to derive training targets like distance zones and urgency labels.
  - Any extra fields (e.g. `distance`, `urgency`) are consumed when present; otherwise derived from annotation metadata where supported.
- **Condition mode:** The dataset can be built with a `condition_mode` (e.g. glaucoma, AMD) so that preprocessing and augmentation match that condition.

## Preprocessing

- **`ml/utils/preprocessing.py`:** **ImagePreprocessor** applies condition-specific transforms: normalization (e.g. ImageNet mean/std), resizing (e.g. 224×224), and condition-based augmentation (e.g. contrast, blur, central/peripheral emphasis for different vision conditions). Used inside the dataset or in the data pipeline.
- **Augmentation:** **`ml/data/advanced_augmentation.py`** (and related) provide stronger augmentation for training (e.g. random crop, color jitter, blur). Usually applied only to training, not validation.

## Data pipeline and loaders

- **`ml/data/data_pipeline.py`:** Builds DataLoaders from MaxSightDataset. Key functions:
  - **create_data_loaders()** (or equivalent): Takes train/val annotation paths, image dir, batch size, num workers, and optional condition_mode. Returns train and val DataLoaders.
  - **collate_fn:** Custom collate that stacks images and pads variable-length targets (labels, boxes, distance, etc.) so each batch is a dict of tensors. Handles optional audio.
- **Sampling:** May use **WeightedRandomSampler** for class balancing if configured. Shuffling is typically enabled for training and disabled for validation.

## Typical usage in scripts

1. **Paths:** Scripts (e.g. `train_maxsight.py`) accept `--train-annotation`, `--val-annotation`, `--image-dir` (and optionally `--condition-mode`).
2. **Creation:** Data pipeline creates MaxSightDataset instances for train and val, then wraps them in DataLoader with the appropriate collate_fn, batch size, and workers.
3. **Batch shape:** Batches are dicts: `images` [B, 3, H, W], `labels` [B, max_objects], `boxes` [B, max_objects, 4], `num_objects` [B], `urgency` [B], etc. The model and losses expect these keys and shapes.

## Splits and annotation generation

- **Splits:** Train/val (and optionally test) splits are defined by separate annotation JSON files. They can be produced by `scripts/gather_training_data.py` or by splitting a single COCO JSON (e.g. `ml/data/coco_dataset_splitter.py`).
- **Gold JSONL + meta (canonical, SageMaker-friendly):** `scripts/ops/build_gold_manifest.py` produces sharded lines + `meta.json`; training uses `data_plane: gold` and meta URIs in YAML. See root **[README.md](../README.md#training-data-plane-gold-jsonl--meta-vs-medallion-index-d2)** and **[infra/README.md](../infra/README.md)** for S3 prefixes.
- **Assistive-derived labels:** COCO-style boxes use the shared formula in **`ml/data/assistive_supervision.py`** with weights in **`ml/config/assistive_supervision.yaml`**. Dataset roles (BDD100K, Epic-Kitchens, VOS, sim) are summarized in **[video_and_navigation_datasets.md](video_and_navigation_datasets.md)**.
- **Medallion (legacy D2):** Optional layout under `datasets/medallion/` with `gold/training_index.json` for path-indexed flows; see **[medallion_data.md](medallion_data.md)**.
- **Inference datasets:** For evaluation (e.g. mAP), inference datasets and annotations may be prepared by `scripts/download_inference_datasets.py` or similar; those are separate from the training data pipeline but follow similar path and annotation conventions.

## Best practices

- Ensure annotation paths and image directories are consistent (e.g. `file_name` in JSON matches files under `image_dir`).
- Use enough workers for DataLoader to avoid CPU bottleneck, but not so many that memory is exhausted.
- Keep validation preprocessing consistent with training (same resize, normalization), but usually without heavy augmentation.
- If you add new keys to the dataset (e.g. new targets), update the collate_fn and the model/loss to consume them.

## Video / sequence tensor contract

For T5 temporal training, the data pipeline supports sequence batches:

- `ml/data/data_pipeline.py` collate function detects sequence mode via a `frames` key.
- It pads variable-length frame sequences into a single tensor shaped `[B, T, C, H, W]`.
- It also returns `frame_lengths` so the temporal encoder can ignore padded time steps safely.
- Optional keys from **video clip manifests** are stacked when every item in the batch has them: `temporal_consistency`, `flicker` (supervision targets aligned with model output names), and string `clip_ids`.

### Clip manifest (v1) and offline scripts

- **Spec:** **[video_panoptic_manifest.md](video_panoptic_manifest.md)** — fixed-stride **T = 8** MVP contract, segment fields, validation rules.
- **Schema:** `docs/schemas/video_panoptic_manifest_v1.schema.json`.
- **Validation:** `ml.data.video_manifest.validate_manifest_v1`.
- **Dataset:** `ml.data.video_clip_dataset.VideoClipManifestDataset` — one sample per manifest clip; detection targets from the **last** frame; temporal proxies via `ml.data.temporal_clip_targets.derive_temporal_clip_targets`.
- **Scripts:** `scripts/ops/sample_video_clips.py` (video or frame dir → paths-only manifest), `scripts/ops/build_pseudo_panoptic_manifest.py` (`--use-stub-segmenter` for smoke runs).
- **Training:** `python scripts/ops/train_maxsight.py ... --temporal-supervision` adds **`ScalarMSELoss`** on **`temporal_consistency`** and **`flicker`** when those keys exist in the batch (use **`VideoClipManifestDataset`** + **`collate_fn`**). The model exposes **`flicker`** and **`temporal_consistency`** tensors for T5 temporal paths.
- **Simulator QA:** `GET /api/dev/sprint-self-tests` and `POST /api/dev/validate-manifest` (enable with **`enable_dev_sprint_tests`** in `tools/simulation/config.py`); the web UI includes a **Sprint self-tests** panel on the main page.

### Dataset performance (no model)

Track I/O and loader throughput on **your** manifest before training or CoreML work:

- **`ml.data.video_dataset_perf`:** `summarize_manifest_frame_files` (how many `frame_paths` exist on disk), `time_manifest_parse_and_validate_ms`, and `profile_video_clip_dataset` (init, sequential `__getitem__`, `DataLoader` + `collate_fn`).
- **CLI:** `python scripts/ops/profile_video_dataset.py --manifest /path/to/manifest.json [--manifest-root DIR] [--summary-only] [--num-workers N] [--json-out report.json]`

Use this to find missing frames, JSON/validation cost, and clips/sec for decode + batching—without adding training fields that only matter once a model consumes them.

If your model/head expects temporal signals (motion stability, flicker suppression, predictive alerts), ensure your training annotations and batching are producing `frames` and `frame_lengths`, not only single-frame `images`.

## Summary

Training data loading is centered on **MaxSightDataset** and **create_data_loaders()** in `ml/data/`. Annotations are COCO-format; preprocessing is condition-aware; batching uses a custom collate. Use `--train-annotation`, `--val-annotation`, and `--image-dir` in training scripts and ensure splits are prepared (e.g. via `gather_training_data.py`). For how to obtain and verify datasets, see `docs/downloads.md`; for the training loop and losses, see `docs/training_architecture.md`.
