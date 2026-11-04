# Dataset licence register

- **Source of truth:** the `license` and `governance` blocks in [configs/data/sources.yaml](../../configs/data/sources.yaml).
  This page summarises them; if the two disagree, the registry wins and this page is wrong.
- **Last reviewed:** 2026-11-03. This is an engineering record, not legal advice.

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

| Source | Licence | Research use | Commercial use | Attribution | Redistribution | Status of terms |
|---|---|---|---|---|---|---|
| COCO 2017 | Annotations CC BY 4.0; images per-image Flickr licence | Yes | Per image: ~26 % of images allow it | Cite Lin et al. (ECCV 2014); per-image licence in `extra.image_license` | Annotations yes; images and derived images no | Licence list read from the annotation JSON |
| Open Images V7 | Annotations CC BY 4.0; images listed as CC BY 2.0 | Yes | Listed as allowed; per-image status to confirm | Credit Google; CC BY 2.0 needs each image's author (not downloaded yet) | Annotations yes; images not until per-image author is recorded | Per-image licence and author not yet recorded |
| Mapillary Vistas v2.0 | CC BY-NC-SA 4.0 | Yes | **No** | Credit Mapillary; keep licence on derived records | Non-commercial, same licence | Registry; confirm on registration |
| BDD100K | BDD100K licence | **Unverified** | Unverified | Cite Yu et al. (CVPR 2020) | Treated as not redistributable | **Read at registration before any use** |
| SANPO | CC BY 4.0 | Yes | Yes | Cite the SANPO paper; credit Google Research | Yes, with attribution | Registry; not acquired (owner decision) |
| DIODE | MIT | Yes | Yes | Cite Vasiljevic et al. (2019); keep the MIT notice | Yes, with notice | Registry |
| PIE | Code MIT; video terms not stated | **Unverified** for video | Unverified | Cite Rasouli et al. (ICCV 2019) | Treated as not redistributable | **Video terms to be confirmed** |
| Ego4D | Signed Ego4D licence | Under the signed licence | Unverified | Per the licence; cite Grauman et al. (CVPR 2022) | Not outside the licence | Not acquired (owner decision) |

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
