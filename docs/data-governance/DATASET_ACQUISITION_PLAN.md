# Dataset acquisition plan

- **Registry:** [configs/data/sources.yaml](../../configs/data/sources.yaml). **Live status:**
  `python scripts/data_sources.py status` and [DATASET_STATUS.md](DATASET_STATUS.md).
- **Owner decisions (2026-11-03):** research use ([licence register](licence-register.md)); SANPO and Ego4D
  not acquired; full datasets wanted where they are needed; training through deployment will run in AWS on an S3
  bucket that has not been created yet.

## Storage constraint

The development Mac had **2.9 GB free** after this slice's downloads (460 GB disk, 99 % used). Acquisition keeps at
least 2 GB free. Everything larger waits for the S3 bucket. Raw archives are kept after extraction (raw data is
immutable), which roughly doubles the space an archive needs while it is extracted.

## Sizes

Figures marked *registry* come from `sources.yaml`; *measured* from this environment; the rest are approximate
public figures (±20 %).

| Source | Needed for MaxSight | Download | On disk after extraction | Where |
|---|---|---|---|---|
| COCO 2017 | Train + val images for the 12 mapped classes, annotations | ~19 GB full (registry) | ~19 GB + 0.7 GB annotations | Subset here; full set in S3 |
| Open Images V7 | 4,000 images per class (registry cap), all splits; train CSV 2.3 GB (measured) | ~30 GB capped | ~30 GB | Validation here; train/test in S3 |
| Mapillary Vistas v2.0 | Full archive (one download) | ~25 GB | ~50 GB during extraction | S3, after registration |
| BDD100K | 100K images + Detection 2020 labels (eval only) | ~7 GB | ~14 GB during extraction | S3, after registration |
| DIODE | Val first (2.6 GB, registry); train depth later (81 GB, registry) | 2.6 GB → 84 GB | ~6 GB → ~210 GB | S3 |
| PIE | Annotations first; clips per experiment | ~74 GB all clips (registry) | ~74 GB | Annotations here; clips in S3 |
| SANPO, Ego4D | Not acquired (owner decision) | — | — | — |

Planning total for the recommended set: about 260–290 GB in S3, plus about 80 GB of temporary space while the DIODE
train archive is extracted.

## Order

1. **Done in slice 1 (this machine):** COCO val2017 (every image with a mapped class), a 1 GB rare-class-first COCO
   train2017 subset, Open Images validation reselected with the current filters. All checked with
   `data_sources.py verify`.
2. **Next (this machine, small):** Open Images test split (CSV ~ size unverified; images capped), PIE annotations,
   Open Images per-image licence and author metadata.
3. **After the S3 bucket exists:** full COCO train2017, Open Images train (needs `--approve-gb 3` for the CSV),
   DIODE val then train depth, Mapillary Vistas and BDD100K after registration, PIE clips for a named experiment.

## Manual actions (project owner)

| Action | Unblocks |
|---|---|
| Create the S3 bucket; set `MAXSIGHT_S3_BUCKET` and the IAM role | Every full download and AWS training |
| Register at mapillary.com and accept the Vistas terms | Mapillary Vistas |
| Create a BDD100K account at bdd-data.berkeley.edu and read the licence | BDD100K (evaluation only) |
| Confirm the PIE video terms | PIE clips |
| Free disk space or attach storage, if more local work is wanted before S3 | Larger local subsets |

## Acquisition rules (enforced in code)

- Nothing downloads without an explicit command; `plan` and `--dry-run` never download.
- Byte budgets (`--max-gb`) and size approvals (`--approve-gb`) refuse oversized downloads.
- Every image gets a SHA-256 in its manifest; every split gets an entry in `ACQUIRED.json` with source URL, code
  revision, selection parameters and archive hashes; a re-run keeps `first_acquired_at`.
- A split is `RAW_VERIFIED` only after `data_sources.py verify` finds every image present, decodable and unchanged.
  COCO and Open Images publish no per-image checksums, so this is recorded as a local check.
- Processed outputs and audit reports are written under new, revision-tagged names; earlier ones are not overwritten.
