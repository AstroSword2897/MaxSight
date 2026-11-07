# Public datasets: acquisition and use

Source of truth: `configs/data/sources.yaml` (roles, licences, access). State is derived from marker files under
`datasets/` (git-ignored), so a configured dataset is never reported as downloaded.

```
python scripts/data_sources.py status
python scripts/data_sources.py plan <dataset> [...]     # always a dry run
python scripts/data_sources.py fetch <dataset> --approve-gb N [--execute]
```

States: CONFIGURED -> DOWNLOADED -> VERIFIED -> PROCESSED -> VALIDATED. Access kinds: automatic, registration,
license_acceptance, credentials, manual_download.

| Dataset | Role | Access | Licence | How |
|---|---|---|---|---|
| COCO 2017 | pretrain | automatic | CC BY 4.0 annotations | `scripts/acquire_data.py coco` (class-capped, byte budget) |
| Open Images V7 | pretrain | automatic | CC BY 4.0 / 2.0 | `scripts/acquire_data.py openimages --split validation --dry-run`, then without `--dry-run` (needs `pip install boto3`). Uses the official CSVs and the anonymous `open-images-dataset` S3 bucket, like the upstream `downloader.py`, plus per-class caps, split-qualified paths, no re-download or overwrite, retries and collected failures. The train bbox CSV is 2.3 GB and needs `--approve-gb 3`. **Pole and Traffic cone are not boxable classes.** |
| Mapillary Vistas v2.0 | pretrain | licence acceptance | CC BY-NC-SA 4.0 (non-commercial) | Register, accept terms, manual download to `datasets/raw/mapillary_vistas_v2/` |
| BDD100K | **eval only** | registration | research licence | Manual download of 100K images (val first) + Detection 2020 labels; convert with `ml/datasets/annotation/bdd100k.py` |
| SANPO | research | automatic (GCS) | CC BY 4.0 | `plan sanpo --session <id> [--segmentation] [--depth]`; explicit sessions only |
| DIODE | research | automatic (HTTP) | MIT | `fetch diode --approve-gb 3` (val 2.6 GB, MD5-checked); validate with `ml/datasets/annotation/diode.py` |
| PIE | research | manual | code MIT; video terms unverified | Selected clips + annotations from the PIE repo |
| Ego4D | research | credentials | Ego4D licence | `pip install ego4d`, AWS profile, then `plan ego4d [--video-uid-file F]` |

## Guardrails (enforced in `ml/datasets/sources.py` and the planners)

- Only `pretrain` sources may enter the detector mix; BDD100K is evaluation only; research sets stay separate.
- Vistas is non-commercial: `commercial_release_blockers()` must be empty before a commercial release.
- SANPO: explicit sessions, right stream and ZED depth excluded, IMU is never a dependency, real/synthetic kept apart,
  use the corrected `fixed_camera_poses.csv`, stereo depth is not evidence of monocular iPhone depth.
- Ego4D: video requires a UID file or benchmark filter. DIODE: train (81 GB) and normals are deferred.
- DIODE evaluation: RGB is the only model input; depth + validity mask are reference only.
- PIE crossing intention is not an assistance-necessity label; it must keep its own schema (not implemented yet).

## Not yet implemented (needs real data to verify formats)

Vistas v2 polygon -> pole_bollard converter, PIE annotation reader, SANPO session reader, Ego4D annotation linker,
PIE/BDD download automation. No data for any dataset has been downloaded.
