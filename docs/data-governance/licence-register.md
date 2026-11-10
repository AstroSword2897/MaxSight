# Dataset licence register

- **Source of truth:** the `license` and `governance` blocks in [configs/data/sources.yaml](../../configs/data/sources.yaml).
  This page summarises them; if the two disagree, the registry wins and this page is wrong.
- **Last reviewed:** 2026-11-04 (ticket 302). This is an engineering record, not legal advice.
- **Approved (Y/N):** Y means cleared for the stated role under the owner's research-use decision (2026-11-03). Every dataset used in training must be Y. A dataset with unverified terms is never Y.

## Owner decision: research use (2026-11-03)

MaxSight is treated as a **research project**. Consequences:

1. NonCommercial (NC) data may be used for training, evaluation and publication, with attribution.
2. Per-image licences are still recorded, and `commercial_release_blockers()` still reports NC sources, so a
   commercially clean export remains possible if this decision changes.
3. "Research" does not make unverified terms acceptable. Sources whose terms have not been read (BDD100K, PIE videos)
   stay out of every export until their terms are reviewed.
4. ShareAlike and NoDerivs obligations still apply to anything shared outside the project (derived datasets,
   published images, released models).

Proposed for the decision register: a DR entry recording this decision and its revisit trigger (any plan to
distribute MaxSight commercially).

## What the licence terms mean here

| Term | Meaning for MaxSight |
|---|---|
| **Attribution (BY)** | Credit the dataset (and, for per-image licences, the author) in papers and anything shared |
| **NonCommercial (NC)** | Use must not be primarily aimed at commercial advantage or payment. Research use is fine; a paid or revenue-driven product is not |
| **ShareAlike (SA)** | Anything shared that is built from the material carries the same licence |
| **NoDerivs (ND)** | Modified versions (crops, resizes, re-encodes) are not shared |
| **Unverified** | The terms have not been read and confirmed from the source. Not used until reviewed |

Whether a model trained on NC images is itself restricted is unsettled. The cautious reading, which spec D7
already applies to Mapillary Vistas, is that such a model is not cleared for commercial release.

## Register

| Dataset | URL | Version | Date checked | Licence | Commercial use | Redistribution | Attribution | Role in MaxSight | Approved (Y/N) |
|---|---|---|---|---|---|---|---|---|---|
| COCO 2017 | https://cocodataset.org/#download | 2017 | 2026-11-03 | Annotations CC BY 4.0; images per-image Flickr licence (CC 2.0 family, read from the annotation file) | Per image: ~26 % allow it, ~69 % NonCommercial | Annotations with attribution; images and derived images not redistributed | Cite Lin et al. (ECCV 2014); per-image licence kept in each record | Detector pretraining | **Y** (research use) |
| Open Images V7 | https://storage.googleapis.com/openimages/web/download_v7.html | v7 | 2026-11-03 | Annotations CC BY 4.0; images listed as CC BY 2.0 | Listed as allowed; per-image status to confirm | Annotations with attribution; images not redistributed until per-image author is recorded | Credit Google; CC BY 2.0 needs each image's author | Detector pretraining (after its audit passes) | **Y** (research use; attribution needed before sharing images) |
| Mapillary Vistas v2.0 | https://www.mapillary.com/dataset/vistas | 2.0 | 2026-11-03 | CC BY-NC-SA 4.0 (registry; confirm at registration) | **No** | Non-commercial, same licence | Credit Mapillary; licence travels with derived records | Pretraining for poles and bollards (research only) | **N** (terms not yet accepted at registration) |
| BDD100K | https://bdd-data.berkeley.edu/ | detection-2020 | 2026-11-03 | BDD100K licence, unverified (read at registration) | Unverified | Unverified; treated as not redistributable | Cite Yu et al. (CVPR 2020) | Evaluation only | **N** (licence not read) |
| PIE (ticket "PEDESTRIAN", to confirm) | https://data.nvision2.eecs.yorku.ca/PIE_dataset/ | 2019 | 2026-11-03 | Code MIT; video terms unverified | Unverified | Unverified; treated as not redistributable | Cite Rasouli et al. (ICCV 2019) | Temporal research (not detector training) | **N** (video terms not stated) |
| RampNet | https://huggingface.co/datasets/projectsidewalk/rampnet-dataset | 1.0 | 2026-11-04 | Code and dataset card MIT; imagery is Google Street View, whose terms are unverified | Unverified (Street View terms) | Unverified (Street View terms) | Cite O'Meara et al. (ICCV 2025 Workshops) | Candidate: curb-ramp coverage for `curb_dropoff` | **N** (Street View terms not reviewed) |
| Waymo Open Dataset | https://waymo.com/open/licensing | Agreement of March 2025 | 2026-11-04 | Waymo Dataset License Agreement for Non-Commercial Use | **No** (separate licence needed) | Only to people registered at waymo.com/open; trained models non-commercial only | Citation required by the agreement | Candidate only (vehicle-mounted cameras) | **N** (not registered; not in the plan) |
| Ego4D | https://ego4d-data.org/docs/start-here/ | v1 | 2026-11-03 | Signed Ego4D licence | Unverified | Not outside the signed licence | Per the licence; cite Grauman et al. (CVPR 2022) | Context research | **N** (not acquired, owner decision) |
| DIODE | https://diode-dataset.org/ | val | 2026-11-03 | MIT | Yes | Yes, with the MIT notice | Cite Vasiljevic et al. (2019) | Depth-reference research (not detector training) | **Y** (research use) |
| NYU Depth V2 | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html | V2 | 2026-11-04 | Not stated on the official page; unverified (mirrors label it differently) | Unverified | Unverified | Cite Silberman et al. (ECCV 2012) | Candidate: indoor depth reference | **N** (no licence stated) |
| SANPO | https://github.com/google-research-datasets/sanpo_dataset | v0 | 2026-11-03 | CC BY 4.0 | Yes | Yes, with attribution | Cite the SANPO paper; credit Google Research | Research (egocentric context) | **N** (not acquired, owner decision) |

Only COCO 2017 and Open Images V7 are used in training today; both are Y. Mapillary Vistas becomes Y once its terms
are accepted at registration. RampNet, Waymo and NYU Depth V2 are candidates and are not in
`configs/data/sources.yaml`; adding one needs an owner decision (DR-112).

## COCO per-image licences (measured)

Every COCO image has its own Flickr licence. Counts from the annotation files on disk:

| Licence | train2017 (all 118,287) | val2017 downloaded (3,648) |
|---|---|---|
| Attribution-NonCommercial-ShareAlike | 27.8 % | 29.3 % |
| Attribution-NonCommercial-NoDerivs | 27.2 % | 27.5 % |
| Attribution | 16.5 % | 17.5 % |
| Attribution-NonCommercial | 13.9 % | 12.1 % |
| Attribution-ShareAlike | 9.2 % | 8.2 % |
| Attribution-NoDerivs | 5.1 % | 5.1 % |
| No known copyright restrictions | 0.4 % | 0.1 % |
| **NonCommercial, total** | **68.9 %** | **69.0 %** |
| **NoDerivs, total** | **32.3 %** | **32.6 %** |

Under the research-use decision all of these are usable for training and evaluation. A commercially clean export
would keep only Attribution, Attribution-ShareAlike and No-known-restrictions images (944 of the 3,648 val2017
images).

## Open actions

| Action | Owner | Blocks |
|---|---|---|
| Read and record the BDD100K licence at registration | Project owner | BDD100K evaluation |
| Confirm PIE video terms | Project owner | PIE acquisition |
| Download Open Images per-image licence and author metadata | Engineering (next slice) | Open Images attribution in any shared artifact |
| Add the research-use decision to the decision register | Project owner | — |
| Confirm whether ticket 302's "PEDESTRIAN" means PIE | Project owner | Register completeness |
| Review Google Street View terms before any RampNet use | Project owner | RampNet |
| Accept Mapillary Vistas terms at registration | Project owner | Vistas approval |
