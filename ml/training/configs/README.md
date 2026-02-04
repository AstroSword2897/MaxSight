# MaxSight Training Configurations

This directory contains YAML configuration files for training different capability tiers of MaxSight.

## Available Configurations

### T0_BASELINE_CNN (`t0_baseline.yaml`)
- **Parameters**: ~29M
- **Features**: ResNet50 + FPN, Tier 1 heads only
- **Batch Size**: 32
- **Use Case**: Baseline model, fastest training

### T1_ATTENTION (`t1_attention.yaml`)
- **Parameters**: ~50M
- **Features**: + SE/CBAM attention, Tier 2 heads
- **Batch Size**: 24
- **Use Case**: Enhanced detection with attention mechanisms

### T2_HYBRID_VIT (`t2_hybrid_vit.yaml`)
- **Parameters**: ~210M
- **Features**: + Hybrid CNN-ViT, Dynamic Convolution
- **Batch Size**: 16
- **Use Case**: Full Tier 2 capabilities with hybrid architecture

### T3_CROSS_TASK (`t3_cross_task.yaml`)
- **Parameters**: ~250M
- **Features**: + Cross-task attention, Tier 3 heads
- **Batch Size**: 12
- **Use Case**: Cross-task learning, scene understanding

### T4_CROSS_MODAL (`t4_cross_modal.yaml`)
- **Parameters**: ~280M
- **Features**: + Cross-modal attention, Audio-visual fusion
- **Batch Size**: 8
- **Use Case**: Multi-modal understanding with retrieval

### T5_TEMPORAL (`t5_temporal.yaml`)
- **Parameters**: ~320M
- **Features**: + Temporal modeling, Video understanding
- **Batch Size**: 4
- **Use Case**: Full temporal understanding (requires video data)

## Usage

`train_maxsight.py` does not take `--config`; it uses explicit arguments. Use the same data layout as `scripts/gather_training_data.py`:

```bash
# Train (default tier T0)
python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --epochs 100 --device cuda
```

The YAML files in this directory describe hyperparameters and tier settings for reference; to use a different tier you would need to pass a tier config into `create_model()` (e.g. via a future `--tier` flag).

## Configuration Structure

Each config file contains:

- **model**: Model architecture settings (tier, attention, etc.)
- **data**: Dataset paths and data loader settings
- **training**: Training hyperparameters (LR, epochs, optimizer, etc.)
- **loss**: Loss function configuration and weights
- **validation**: Validation settings
- **checkpoint**: Checkpoint saving configuration
- **logging**: Logging and tensorboard settings
- **device**: Device selection (auto/cuda/mps/cpu)

## Notes

- All tiers require **cloud GPU (CUDA)** for training
- Batch sizes are optimized for GPU memory constraints
- Gradient accumulation is used for larger models to maintain effective batch size
- Mixed precision (AMP) is enabled for faster training
- GradNorm is used for multi-task loss balancing (T1+)

