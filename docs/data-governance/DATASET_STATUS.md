# Dataset status

- **As of:** 2026-11-03, branch `pr/6-data-governance`, code revision `37731e6`.
- **Live check:** `python scripts/data_sources.py status` (derived from marker files; this page is a snapshot).
- **Owner decisions:** research use; SANPO and Ego4D not acquired; full downloads wait for the S3 bucket.

Status values: `NOT_DOWNLOADED`, `DOWNLOADED_UNVERIFIED`, `RAW_VERIFIED`, `PREPROCESSED_UNVERIFIED`, and
`READY_FOR_<purpose>` only after the quality audit passes. No source is ready for any purpose yet.

## Summary

| Dataset | Intended role | Download status | Integrity status | Licence status | Preprocessing status |
|---|---|---|---|---|---|
| COCO 2017 | Detector pretraining | Partial: val2017 mapped images (all), train2017 subset (1 GB) | **RAW_VERIFIED** (9,504 / 9,504) | Research use OK; per-image licence recorded | **PREPROCESSED_UNVERIFIED**: audit failed |
| Open Images V7 | Detector pretraining | Partial: validation split, reselected | **RAW_VERIFIED** (6,321 / 6,321) | Research use OK; per-image author not recorded | **PREPROCESSED_UNVERIFIED**: audit failed |
| Mapillary Vistas v2.0 | Pretraining (research, non-commercial) | NOT_DOWNLOADED: needs your registration | — | CC BY-NC-SA; research OK | No converter yet |
| BDD100K | Evaluation only | NOT_DOWNLOADED: needs your account | — | **Unverified**: read at registration | Converter exists, untested on real data |
| SANPO | Research | NOT_DOWNLOADED: excluded by owner | — | CC BY 4.0 | No reader |
| DIODE | Research (depth reference) | NOT_DOWNLOADED: waits for S3 (≈ 6 GB with archive) | — | MIT | Scanner exists, untested on real data |
| PIE | Research (pedestrian behaviour) | NOT_DOWNLOADED | — | **Video terms unverified** | No reader |
| Ego4D | Research | NOT_DOWNLOADED: excluded by owner | — | Signed licence needed | No linker |

## Acquired splits

| Split | Selected | Downloaded | Not downloaded | Why | On disk |
|---|---|---|---|---|---|
| COCO val2017 | 3,648 (every image with a mapped class) | 3,648 | 0 | — | Images + `manifest_val2017.jsonl` |
| COCO train2017 | 15,261 (4,000 per class, rare classes first) | 5,856 | 9,405 | All `byte budget reached` (`--max-gb 1.0`); 0 network errors | Images + `manifest_train2017.jsonl` |
| Open Images validation | 6,321 (4,000 per class, trainable boxes) | 6,321 | 0 | — | Images + `manifest_validation.jsonl` |

Raw disk use (measured): COCO 2.3 GB including the 241 MB annotation zip and 448 MB `instances_train2017.json`; Open Images
2.3 GB, including 1,241 images from the earlier selection that the current filters no longer pick (kept, not
deleted). Free space after this slice: 2.9 GB.

Every image has a SHA-256 in its manifest. Each split's entry in `datasets/raw/<id>/ACQUIRED.json` records the
source URL, code revision, selection parameters, archive hashes, failure reasons and the integrity result.

## Cleaning and audit (per source)

Run with `scripts/validate_data.py --clean --split`. Outputs are revision-tagged and earlier outputs are kept:
`datasets/processed/<id>/<split>_clean.<rev>.jsonl`, `datasets/reports/<id>_<split>_audit.<rev>.json`.

| Split | Kept | Dropped | Small boxes → ignore | Split (train / val / test) | Audit |
|---|---|---|---|---|---|
| COCO val2017 | 3,646 | 2 near-duplicates | 0 at cleaning (already ignored at selection) | 2,572 / 530 / 544 | **FAILED**: person 55.2 % |
| COCO train2017 subset | 5,854 | 2 near-duplicates | 0 at cleaning (already ignored at selection) | 4,063 / 897 / 894 | **FAILED**: person 42.2 %, vehicle 19.5 % |
| Open Images validation | 6,319 | 1 near-duplicate, 1 with no boxes left | 53 | 4,449 / 923 / 947 | **FAILED**: vehicle 34.3 %, person 26.9 %, animal 15.2 % |

### Box counts per class (after cleaning)

| Class | COCO val2017 | COCO train2017 subset | Open Images val | Total |
|---|---|---|---|---|
| person | 10,005 | 21,461 | 3,959 | 35,425 |
| vehicle | 2,423 | 9,932 | 5,061 | 17,416 |
| bicycle | 308 | 5,807 | 393 | 6,508 |
| motorcycle | 360 | 5,019 | 222 | 5,601 |
| bag_box | 1,145 | 3,524 | 258 | 4,927 |
| animal | 417 | 431 | 2,244 | 3,092 |
| chair | 1,727 | 1,063 | 721 | 3,511 |
| fire_hydrant | 93 | 1,774 | 15 | 1,882 |
| table | 666 | 287 | 932 | 1,885 |
| potted_plant | 338 | 693 | 428 | 1,459 |
| bench | 375 | 809 | 9 | 1,193 |
| couch | 261 | 54 | 60 | 375 |
| door | 0 | 0 | 277 | 277 |
| pole_bollard | 0 | 0 | 86 | 86 |
| stairs | 0 | 0 | 43 | 43 |
| trash_bin | 0 | 0 | 29 | 29 |
| traffic_cone | 0 | 0 | 0 | **0** |

### What the failures mean

- The 15 % share limit is meant for the **combined training mix**, so single sources dominated by person and vehicle
  are expected to fail. The threshold was not changed to make them pass.
- The mobility-relevant classes are critically thin: stairs 43, trash_bin 29, pole_bollard 86, door 277, and no
  traffic_cone anywhere. COCO has none of these classes; Open Images validation is the only source so far. More data
  for them must come from Open Images train/test, Mapillary Vistas (poles, bollards) and MaxSight's own captures, not
  from duplicating examples.
- Splits were assigned per source. Duplicates **across** sources (for example COCO train vs val) are not checked yet;
  that needs the combined audit in the next slice.
- Neither selector keeps images without a mapped box, so there are no negative examples yet.

## Why nothing is ready

| Purpose | Blocked by |
|---|---|
| Detector pretraining | Combined audit not built or passed; rare classes too thin; no negative pool; no SSD-MobileNet loader to export for |
| Evaluation only (BDD100K) | Not downloaded; licence not read |
| Segmentation experiments (Vistas) | Not downloaded; no converter |
| Depth experiments (DIODE) | Not downloaded (needs S3 or space) |
| Temporal experiments (PIE) | Not downloaded; video terms unverified; no reader |
| Assistance necessity (Problem 2) | No public source has these labels; needs MaxSight's own reviewed recordings |
