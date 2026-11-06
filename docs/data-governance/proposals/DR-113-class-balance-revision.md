# Proposal: revise DR-113 (class balance)

- **Status:** Proposed, for owner review. Nothing in the code, data, audit thresholds or training configuration has
  been changed by this proposal.
- **Date:** 2026-11-04. **Evidence revision:** branch `pr/6-data-governance` (PR #23).
- **Current DR-113** ([decision register](../../architecture/v1/01-decision-register.md)): "cap 8,000 instances per
  class; no class > 15 % of instances; car/bus/truck merged as `vehicle`" (Proposed; alternative listed: balanced
  sampler).

## Why

The 15 % share rule measures whether box counts are balanced. That is not the property that matters. A natural
street and indoor dataset is dominated by people and vehicles, and every source fails the rule (COCO val2017
person 55.2 %; Open Images validation vehicle 34.3 %). Meeting it by deleting images would remove more than 80 % of
person boxes and, through co-occurrence, many rare-class boxes. Meanwhile the real risk, too few examples of
mobility-critical classes, is checked only by a warning when a class has fewer than one box.

Established long-tail detection practice keeps the natural distribution, compensates during training, and evaluates
per class (LVIS frequency groups and Repeat Factor Sampling; class-balanced loss, Cui et al., CVPR 2019;
Equalization Loss v2, Tan et al., CVPR 2021).

## Findings: what the code does today

| # | Finding | Evidence |
|---|---|---|
| F1 | `datasets/cleaned_splits/class_weights.json` is **not read by any code**. It is keyed by the 622-class `accessibility_622` label names (e.g. `access_ramp`), not the 17 detector classes, and its own statistics report 353 of 622 classes with fewer than 5 annotations. Last changed 2026-01-11. | `git grep class_weights.json` (no reader); file contents |
| F2 | Weighted sampling is **off in every training config** (`use_weighted_sampling: false` in all 8 tier YAMLs) and `data.class_weights` is never set (dataclass default `None`). | `ml/training/configs/t*.yaml:34`; `ml/training/run_config.py:131` |
| F3 | If enabled, the sampler weights each image by the class of its **first** box (`labels[0]`), not its rarest or most frequent class, using **inverse instance frequency** keyed by COCO category id. It loads and augments every training sample once just to compute the weights. | `ml/data/data_pipeline.py:216-236`, `:562-594` |
| F4 | Class weighting is **impossible** in the two other loading paths: multi-source composition raises if weighted sampling is on, and the gold data plane has no sampler. | `ml/data/data_pipeline.py:436-440`, `:319-418`; `run_config.py:871` |
| F5 | The detection classification loss is **focal loss with one α = 0.25 for every class**. It down-weights easy examples (mainly background) but has no per-class term. | `ml/training/losses.py:27-46`; wired in `ml/training/runner.py:74` |
| F6 | A class-weighted loss with `inverse_freq`, `inverse_sqrt` and `effective_samples` (the CVPR 2019 effective-number formula, β = 0.9999) **exists but is unused**. Its manager checks label smoothing (default 0.1) before class weights, so the weights would never apply under defaults. | `ml/training/regularization.py:147-205`, `:375-398`; no callers |
| F7 | The training runner trains the multi-head MaxSight model, **not SSD-MobileNet** (no SSD code exists). Any sampler or loss decision must be re-implemented for the SSD baseline. | `ml/models/detector/` contains only `.gitkeep` |

## Findings: Repeat Factor Sampling on the current data

Computed on the cleaned training-split records (11,084 images, 59,117 boxes) using Detectron2's definition:
r(c) = max(1, √(t / f(c))), where f(c) is the fraction of training images containing class c, and each image takes
the largest r(c) among its classes.

| Class | Training images | f(c) | Box share | t = 0.001 (LVIS default) | t = 0.01 | t = 0.05 |
|---|---|---|---|---|---|---|
| person | 5,739 | 0.518 | 42.6 % | r 1.00 → 42.6 % | 1.00 → 42.5 % | 1.00 → 41.8 % |
| vehicle | 4,130 | 0.373 | 20.9 % | 1.00 → 20.9 % | 1.00 → 20.8 % | 1.00 → 20.4 % |
| door | 135 | 0.0122 | 0.3 % | 1.00 → 0.3 % | 1.00 → 0.3 % | 2.03 → 0.7 % |
| pole_bollard | 22 | 0.0020 | 0.1 % | 1.00 → 0.1 % | 2.24 → 0.2 % | 5.02 → 0.5 % |
| stairs | 25 | 0.0023 | 0.1 % | 1.00 → 0.1 % | 2.11 → 0.1 % | 4.71 → 0.2 % |
| trash_bin | 11 | 0.0010 | 0.0 % | 1.00 → 0.0 % | 3.17 → 0.1 % | 7.10 → 0.2 % |
| traffic_cone | 0 | 0 | 0 % | — | — | — |

Epoch length grows 1.00×, 1.01× and 1.05× respectively. Conclusions:

1. At the LVIS threshold nothing is repeated, because the rarest class here (f = 0.001) sits exactly at t.
2. Even aggressive thresholds barely move the shares. Repeating a stairs image also repeats the people in it
   (co-occurrence), and 25 images repeated five times are still 25 distinct staircases.
3. **Sampling and loss weighting cannot fix this dataset's main problem, which is scarcity.** They are worth
   comparing once the critical classes have enough distinct examples.

## Proposed revision

### 1. Split the audit into three gates

| Gate | Purpose | Pass/fail | Contents |
|---|---|---|---|
| **G1 Integrity** | Is the data structurally valid? | Hard fail | Missing files, invalid or out-of-bounds boxes, duplicate ids, unknown classes, split leakage by group, near-duplicates across splits (all existing checks except the share rule) |
| **G2 Coverage** | Is there enough evidence for each critical class? | Hard fail for critical classes | Per class: boxes, distinct images, distinct groups (scenes), distinct sources, object-size spread; separately for train and for held-out evaluation |
| **G3 Training readiness** | Is the imbalance understood and handled? | Report, reviewed per experiment | Box share and image share per class, per-source-label counts inside merged classes, co-occurrence (e.g. people per stairs image), chosen sampler and loss, expected exposure per class |

### 2. Changes to DR-113

| Current | Proposed |
|---|---|
| No class > 15 % of instances (hard) | Remove as a pass/fail rule. Report box and image shares in G3. |
| Cap 8,000 instances per class | No cap on the dataset. Any cap becomes an experiment-level sampling parameter, recorded with the run. |
| Warn below one box | G2 coverage minimums for critical classes (below) |
| Merged `vehicle` (car/bus/truck), `animal` (dog/cat) | Keep the merge for the detector; G3 reports each source label separately |
| Alternative: balanced sampler | Controlled comparison (section 4); choose by critical-class recall, not overall mAP |

### 3. Coverage minimums: derive them, do not guess them

- **Evaluation floor (proposed rule):** each critical class needs enough held-out positives to estimate recall to a
  stated precision. For a 95 % interval of about ±10 percentage points that is roughly 100 instances
  (n ≈ 1.96² × 0.25 / 0.10²), drawn from distinct groups. The exact precision target belongs to DR-205.
- **Training floor:** set from a learning-curve experiment (per-class recall versus number of distinct training
  images), not from a fixed number. Until it is measured, G2 reports the counts and blocks only on the evaluation
  floor.
- **Diversity:** minimums count distinct images and distinct groups, not boxes, so 43 boxes from three photos of one
  staircase cannot pass.
- **Critical classes:** depend on DR-202 (Open). Candidates from the plan: stairs, door, pole_bollard, trash_bin,
  traffic_cone, plus person, bicycle and vehicle for moving hazards.

### 4. Experiment matrix (after G2 passes)

Validation and test stay at their natural distribution in every run.

| Run | Sampling | Loss |
|---|---|---|
| A Baseline | Random | Current (focal, uniform α) |
| B | Repeat Factor Sampling | Current |
| C | Random | Class-balanced (effective number), reusing `ClassWeightedLoss` once its precedence issue is fixed |
| D | Repeat Factor Sampling | Class-balanced |
| E | Best of B–D | EQLv2, or a softmax-compatible equivalent if the SSD head stays softmax (EQLv2 is formulated for sigmoid classifiers; to confirm) |

Report overall mAP@50:95, per-class AP and recall, false-positive rate, small-object results and per-environment
results. Pick on critical-class recall at an acceptable false-positive rate; check that weighting does not inflate
false positives.

## What the proposed G2 would say today

Measured on the cleaned records; held-out means the val and test splits together.

| Class | Train images / boxes | Held-out images / boxes | Sources | G2 (evaluation floor ≈ 100) |
|---|---|---|---|---|
| stairs | 25 / 31 | 10 / 12 | Open Images only | Fail |
| trash_bin | 11 / 20 | 7 / 9 | Open Images only | Fail |
| pole_bollard | 22 / 62 | 8 / 24 | Open Images only | Fail |
| door | 135 / 194 | 62 / 83 | Open Images only | Fail |
| traffic_cone | 0 / 0 | 0 / 0 | none | Fail |

That is the correct outcome: these classes need more distinct data (Open Images train and test splits, Mapillary
Vistas for poles and bollards, and MaxSight captures), not reweighting. With 7–10 held-out images, any recall
measured for stairs, trash_bin or pole_bollard today would have an interval roughly ±30 points wide or more.

## Decisions requested

1. Approve removing the 15 % rule and the 8,000 cap as dataset pass/fail rules (DR-113 revision).
2. Approve the three-gate structure and the evaluation-floor rule.
3. Name the critical classes (DR-202) so G2 can be implemented.
4. Confirm that `class_weights.json` (622-class) can be marked obsolete for the 17-class detector.
