# Evaluation Guide

## Overview

This guide covers evaluating trained MaxSight models.

## Quick Evaluation

### Basic Evaluation

```bash
python -m ml.eval.quick_eval \
  --ckpt runs/experiment1/checkpoint_best.pth \
  --split val
```

### Full Evaluation

```bash
python -m ml.eval.eval_full \
  --ckpt runs/experiment1/checkpoint_best.pth \
  --split val \
  --out eval/results.json
```

## Metrics

### Object Detection Metrics

- **mAP (mean Average Precision)**: Primary metric for object detection
- **mAP@0.5**: mAP at IoU threshold 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds 0.5-0.95

### Per-Head Metrics

- **Classification Accuracy**: Per-class accuracy
- **Localization IoU**: Bounding box accuracy
- **Distance MAE**: Mean absolute error for distance estimation
- **Urgency Accuracy**: Urgency level classification accuracy

### Accessibility Metrics

- **False Reassurance Rate**: Rate of missed dangerous objects
- **Alert Latency**: Time to alert for hazards
- **Navigation Accuracy**: Path planning accuracy

## Evaluation Scripts

### `ml/eval/quick_eval.py`

Quick sanity check:
- Loads model
- Runs on validation set
- Prints basic metrics

### `ml/eval/eval_full.py`

Comprehensive evaluation:
- All metrics
- Per-class breakdown
- Failure case analysis
- Generates reports

## Robustness Evaluation

### Corrupted Images

Test on corrupted images to measure robustness:

```bash
python -m ml.eval.eval_full \
  --ckpt runs/experiment1/checkpoint_best.pth \
  --split val_corrupted \
  --out eval/corrupted_results.json
```

### Cross-Condition Performance

Evaluate performance across different visual conditions:
- Brightness variations
- Contrast variations
- Blur levels

## Model Analysis

### Per-Class Performance

Analyze performance per object class:
- Confusion matrices
- Precision/recall per class
- Common failure modes

### Failure Cases

Extract and analyze failure cases:
- False positives
- False negatives
- Low confidence detections

## Reporting

Evaluation generates:
- **JSON reports**: Machine-readable metrics
- **Visualizations**: Graphs and charts (if matplotlib available)
- **Failure cases**: Sample images with annotations

## Best Practices

1. **Evaluate on validation set**: Never evaluate on training set
2. **Report multiple metrics**: Don't rely on single metric
3. **Analyze failure cases**: Understand model limitations
4. **Compare baselines**: Compare against T0 baseline
5. **Document results**: Keep evaluation results for comparison

## Expected Results

### Baseline (T0)

- mAP@0.5: ~0.35-0.40
- Classification Accuracy: ~0.70-0.75

### Hybrid (T2)

- mAP@0.5: ~0.40-0.45
- Classification Accuracy: ~0.75-0.80

### Temporal (T5)

- mAP@0.5: ~0.45-0.50
- Classification Accuracy: ~0.80-0.85

*Note: Results vary based on training data and hyperparameters*

