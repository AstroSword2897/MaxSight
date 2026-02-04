# MaxSight Tier Transfer Guide

## T2_HYBRID_VIT → T5_TEMPORAL Transfer Plan

This guide implements a disciplined tier transfer that cuts weeks off training while preventing negative transfer.

---

## Prerequisites (Non-Negotiable)

Before transfer, your **T2 checkpoint must satisfy ALL**:

- ✅ Val loss plateaued ≥ **10 epochs**
- ✅ Box AP stable (±1%)
- ✅ Accessibility heads (urgency, distance, navigation) have non-zero gradients
- ✅ No NaNs in last 20 epochs
- ✅ GradNorm scaling stabilized

**If not, do not transfer. Fix T2 first.**

---

## What Gets Transferred (and What Doesn't)

### ✅ Transfer (Copy Weights)

| Component                | Action   |
|--------------------------|----------|
| CNN backbone             | **Copy** |
| ViT blocks               | **Copy** |
| SE / CBAM                | **Copy** |
| Dynamic Conv             | **Copy** |
| Detection head           | **Copy** |
| Box regression head      | **Copy** |
| Classification head      | **Copy** |
| Distance / urgency heads | **Copy** |

### ❌ DO NOT Transfer

| Component                       | Why                     |
|--------------------------------|-------------------------|
| Temporal modules                | Random init required    |
| Cross-task attention            | Dimensional mismatch    |
| Cross-modal attention           | No learned alignment    |
| Retrieval modules               | Task distribution shift |
| Scene graph / OCR / sound heads | New loss geometry       |

**Rule:** Transfer *representation*, not *coordination*.

---

## Freeze Schedule (Corrected Timing)

### Epochs 0–5
```
FROZEN:
- CNN backbone
- ViT backbone
- Detection + box heads

TRAINABLE:
- New T5 heads only (temporal, cross-attention, new heads)
```

**Why:** New heads need signal early, before spatial adaptation.

### Epochs 5–15
```
UNFREEZE:
- Detection head
- Classification head

Still frozen:
- CNN + ViT backbone
```

### Epochs 15–30
```
UNFREEZE:
- ViT blocks (top 40%)

Still frozen:
- CNN backbone
- Early ViT layers
```

**Why:** ViT should adapt before temporal losses unlock.

### Epochs 30–45
```
UNFREEZE:
- Full ViT

Still frozen:
- CNN backbone
```

### Epochs 45+
```
UNFREEZE:
- Entire model (including CNN)
```

CNN unfreezes last — always.

---

## Learning Rate Multipliers

Use **parameter-grouped AdamW** with these multipliers:

| Module                | LR multiplier |
|-----------------------|---------------|
| CNN backbone          | ×0.2          |
| ViT backbone          | ×0.5          |
| Detection / box heads | ×0.6          |
| Temporal modules      | ×1.0          |
| Cross-task / modal    | ×1.0          |
| New heads             | ×1.3          |

### Example (base LR = `7.5e-5`)

```
CNN:        1.5e-5
ViT:        3.75e-5
Detection:  4.5e-5
Temporal:   7.5e-5
New heads:  9.75e-5
```

**Rationale:**
- ViT needs more plasticity to support temporal modeling
- New heads must move fastest or they lag permanently

This prevents catastrophic forgetting.

---

## Loss Unlock Schedule (Aligned with Representation Readiness)

### Epochs 0–10 (Phase 1)

**Enable only:**
- detection
- classification
- box_regression

**Disable:**
- All other tasks

### Epochs 10–25 (Phase 2)

**Enable:**
- distance
- urgency
- motion
- roi_priority
- navigation_difficulty

**Still disabled:**
- therapy_state
- scene_description
- ocr
- scene_graph
- sound_events
- personalization
- predictive_alerts

### Epochs 25–40 (Phase 3)

**Enable:**
- therapy_state

**Still disabled:**
- scene_description
- ocr
- scene_graph
- sound_events
- personalization
- predictive_alerts

### Epochs 40+ (Phase 4)

**Enable all losses.**

GradNorm should be active the entire time.

---

## Validation Expectations (Revised)

If transfer is healthy:

| Epoch range | Expected behavior             |
|-------------|-------------------------------|
| 0–5         | Metrics noisy, loss spikes    |
| 5–15        | Detection stabilizes          |
| 15–30       | Navigation loss drops         |
| 30–45       | Temporal heads wake up        |
| 45–70       | T5 surpasses T2 metrics      |
| 70+         | Diminishing returns           |

### 🚨 Red Flags

- **T5 beats T2 before epoch 30** → data leakage or frozen layers misconfigured
- **Immediate val improvement** (means leakage)
- **Box AP collapse** after epoch 5
- **Temporal loss dominates** total loss

---

## Usage

### 1. Validate Source Checkpoint

```bash
python scripts/transfer_t2_to_t5.py \
  --config ml/training/configs/t2_to_t5_transfer.yaml \
  --validate-only
```

### 2. Run Transfer Training

```bash
python scripts/transfer_t2_to_t5.py \
  --config ml/training/configs/t2_to_t5_transfer.yaml
```

### 3. Monitor Training

Watch for:
- Loss smoothness (not absolute value) in first 25 epochs
- Box AP stability
- Temporal loss gradually increasing
- No NaN values

---

## Checkpoint Hygiene

- Save **last T2 checkpoint** separately (never overwrite)
- Save T5 checkpoints every **5 epochs** until epoch 50
- After epoch 50, revert to every 10 epochs

---

## One Brutal Truth

**If your T2 model is weak, T5 will amplify that weakness.**

Tier transfer is a **force multiplier**, not a fix.

---

## TL;DR

```text
1. Copy spatial weights only
2. Freeze backbone (0–10)
3. Gradual unfreeze (10–40)
4. LR multipliers by module
5. Losses unlocked in phases
```

---

## Files

- `ml/training/transfer_learning.py` - Transfer manager implementation
- `ml/training/configs/t2_to_t5_transfer.yaml` - Transfer configuration
- `scripts/transfer_t2_to_t5.py` - Transfer training script

