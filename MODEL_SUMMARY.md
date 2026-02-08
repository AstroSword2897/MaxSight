# MaxSight model – quick reference

## One-liner

**MaxSightCNN**: ResNet50/FPN backbone → detection (cls, box, obj, text) + scene/urgency/depth heads; T5 tier adds hybrid backbone, temporal encoder, cross-task/cross-modal attention. Condition-specific (e.g. glaucoma, amd) via `condition_mode`. Export: JIT (default) or ExecuTorch PTE.

## Key files

| Path | Role |
|------|------|
| `ml/models/maxsight_cnn.py` | Main model, `create_model()`, `TierConfig`, `CapabilityTier` |
| `ml/models/backbone/hybrid_backbone.py` | Hybrid CNN (ResNet + ViT patches) |
| `ml/models/backbone/vit_backbone.py` | ViT patch embedding / transformer |
| `ml/models/heads/*.py` | Detection, depth, motion, uncertainty, etc. |
| `ml/models/temporal/temporal_encoder.py` | T5 temporal (ConvLSTM / TimeSformer) |
| `ml/models/scene_graph/scene_graph_encoder.py` | Scene graph (disabled for JIT export) |
| `ml/training/export.py` | `export_ios_bundle()`, JIT/ExecuTorch/CoreML/ONNX |

## Deploy (quick)

```bash
# JIT-only export (default): fast, no ExecuTorch
python scripts/inference_and_deploy_top7.py --checkpoints-base /path/to/MaxSight --output-dir /path/to/exports_top7 --quick

# Or deploy only (skip inference). Replace <BASE> with your checkpoints root (e.g. . or /content/drive/MyDrive/MaxSight).
python scripts/inference_and_deploy_top7.py --skip-inference --quick --checkpoints-base <BASE> --output-dir <BASE>/exports
```

Colab: clone, `%cd /content/2026-Prototype`, then run the same with `--checkpoints-base /content/drive/MyDrive/MaxSight` (and `--quick` for fast export).

## Create model (code)

```python
from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier

model = create_model(
    num_classes=80,
    use_audio=False,
    condition_mode="glaucoma",  # or amd, cataracts, etc.
    tier_config=TierConfig.for_tier(CapabilityTier.T5_TEMPORAL),
)
# Forward: out = model(images)  # [B, 3, 224, 224] -> dict with classifications, boxes, objectness, ...
```

## Architecture (short)

- **Stage A**: Backbone (ResNet50 or hybrid) → FPN → detection head → cls/box/obj/text + scene embedding, urgency, depth, distances.
- **Stage B (T5)**: Hybrid backbone, temporal encoder, cross-task attention, cross-modal (CLIP); scene graph off for export.
- **Heads**: Classification (80 COCO + accessibility), box regression, objectness, text regions, urgency (4), distance zones, depth map, motion (optional).

See `docs/architecture.md` for full design.
