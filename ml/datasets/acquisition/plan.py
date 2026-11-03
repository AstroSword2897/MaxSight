"""Acquisition plans: what would be fetched, how large, and what a human must do first.

Planning never touches the network. ``Plan.argv`` steps are exact commands; ``manual`` lists
actions that cannot be automated (registration, licence acceptance, credentials).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml.datasets.sources import DEFAULT_DATA_ROOT, Source

SANPO_REGIONS = ("sanpo-real", "sanpo-synthetic")
# Stream folders named on the official SANPO page; the left-camera folder name is not documented there,
# so selection excludes streams instead of guessing an include name.
SANPO_ALWAYS_EXCLUDE = (r".*/right", r".*/zed_depth_maps")
SANPO_SEGMENTATION = r".*/segmentation_masks"
SANPO_DEPTH = r".*/depth_maps"


@dataclass(frozen=True)
class Plan:
    source_id: str
    est_gb: float | None
    argv: tuple[tuple[str, ...], ...] = ()
    manual: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def automatable(self) -> bool:
        return bool(self.argv) and not self.manual


def raw_dir(src: Source, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return data_root / "raw" / src.id


def sanpo_plan(
    src: Source, sessions: list[str], *, region: str = "sanpo-real", segmentation: bool = False,
    depth: bool = False, data_root: Path = DEFAULT_DATA_ROOT,
) -> Plan:
    """One ``gcloud storage rsync`` per explicitly named session. Never the whole bucket."""
    if region not in SANPO_REGIONS:
        raise ValueError(f"region must be one of {SANPO_REGIONS}")
    if not sessions:
        raise ValueError("SANPO requires explicit --session ids; refusing to plan a bucket-wide download")
    excludes = list(SANPO_ALWAYS_EXCLUDE)
    if not segmentation:
        excludes.append(SANPO_SEGMENTATION)
    if not depth:
        excludes.append(SANPO_DEPTH)
    steps = []
    for sid in sessions:
        if "/" in sid or ".." in sid:
            raise ValueError(f"invalid session id {sid!r}")
        dest = raw_dir(src, data_root) / region / sid
        steps.append(("gcloud", "storage", "rsync", "-r", *(a for e in excludes for a in ("--exclude", e)),
                      f"{src.raw['bucket']}/{region}/{sid}", str(dest)))
    return Plan(src.id, None, tuple(steps), notes=(
        "Size unknown until the session is listed: run each command with --dry-run first (gcloud supports it).",
        "IMU data is not requested explicitly; do not make it a dependency.",
        "SANPO-Real poses: use the corrected fixed_camera_poses.csv.",
        "Real and synthetic land in separate sanpo-real / sanpo-synthetic folders.",
    ))


def ego4d_plan(
    src: Source, *, video_uid_file: Path | None = None, benchmarks: list[str] | None = None,
    include_video: bool = False, aws_profile: str = "ego4d", datasets_flag: str = "--datasets",
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Plan:
    """Annotations first; video only with a UID file or benchmark filter."""
    out = str(raw_dir(src, data_root))
    base = ["ego4d", "--output_directory", out, "--aws_profile_name", aws_profile, "-y"]
    steps = [(*base, datasets_flag, "annotations", "--metadata")]
    if include_video:
        if video_uid_file is None and not benchmarks:
            raise ValueError("Ego4D video needs --video-uid-file or --benchmarks; refusing a full download")
        sel = ["--video_uid_file", str(video_uid_file)] if video_uid_file else ["--benchmarks", *(benchmarks or [])]
        steps.append((*base, datasets_flag, "full_scale", *sel, "--no-metadata"))
    return Plan(src.id, None, tuple(steps), manual=(
        "Sign the Ego4D licence and configure the emailed AWS credentials as a profile "
        f"named {aws_profile!r} in ~/.aws/credentials.",
    ), notes=(
        "The CLI documents both --dataset and --datasets; confirm with `ego4d --help` (override datasets_flag).",
        "The CLI has no dry-run option; size is not reported before download.",
    ))


def diode_plan(src: Source, data_root: Path = DEFAULT_DATA_ROOT, include_train: bool = False) -> Plan:
    if include_train:
        raise ValueError("DIODE train (81 GB) is deferred by the research plan")
    arch = src.raw["archives"]
    est = sum(a["gb"] for a in arch.values())
    return Plan(src.id, est, notes=(
        "val.tar.gz MD5 is published and enforced; data_list.zip has no published checksum (size/sha256 recorded).",
        "RGB is model input; depth + depth_mask are reference only.",
    ))


def manual_plan(src: Source) -> Plan:
    steps = {
        "coco2017": ("python scripts/acquire_data.py coco --dry-run  (then without --dry-run with --max-gb)",),
        "open_images_v7": ("pip install fiftyone; python scripts/acquire_data.py openimages",
                           "Pole and Traffic cone are not Open Images boxable classes; they yield no images."),
        "mapillary_vistas_v2": (f"Register and accept the licence at {src.homepage}, download v2.0, unpack to "
                                "datasets/raw/mapillary_vistas_v2/. Treat as CC BY-NC-SA 4.0 (non-commercial).",),
        "bdd100k": (f"Register at {src.homepage}; download '100K Images' (val first) and 'Detection 2020 Labels' "
                    "to datasets/raw/bdd100k/. Evaluation only.",),
        "pie": (f"Download selected clips from {src.homepage}PIE_clips/ plus annotations from github.com/aras62/PIE "
                "to datasets/raw/pie/. File names were not verifiable; confirm before scripting.",),
    }
    return Plan(src.id, None, manual=steps.get(src.id, (src.access_note,)))
