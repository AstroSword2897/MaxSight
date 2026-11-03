"""Open Images V7 subset via the official "download manually" route (no FiftyOne).

Uses the same sources as the official ``openimages/dataset`` ``downloader.py`` (Apache-2.0): the boxable
class list, the bbox annotation CSVs and the anonymous ``open-images-dataset`` S3 bucket. Differences that
matter for MaxSight: images are picked per class with a cap, output paths keep the split, finished files are
never re-downloaded or overwritten, transient errors are retried, and failures are collected instead of
``sys.exit`` ending the whole run. Pole and Traffic cone are not boxable classes and are reported as unavailable.
"""

from __future__ import annotations

import csv
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord

SOURCE = "openimages"
LICENSE = "annotations CC BY 4.0; images CC BY 2.0"
BUCKET = "open-images-dataset"
SPLITS = ("train", "validation", "test")
URLS = {
    "class_descriptions": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "bbox_train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "bbox_validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "bbox_test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
# Verified Content-Length of the annotation CSVs (bytes); test is not verified.
BBOX_BYTES = {"train": 2_258_447_590, "validation": 25_105_048}
LIST_LINE = re.compile(r"^(test|train|validation|challenge2018)/([0-9a-fA-F]+)$")


class S3Client(Protocol):
    def download_file(self, bucket: str, key: str, filename: str) -> None: ...


@dataclass(frozen=True)
class BoxRow:
    label_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    is_group_of: bool
    is_depiction: bool


@dataclass
class Selection:
    images: dict[str, list[BoxRow]] = field(default_factory=dict)
    class_counts: Counter = field(default_factory=Counter)


def parse_class_descriptions(text: str) -> dict[str, str]:
    """``LabelName,DisplayName`` CSV -> {display name: label id}. The v7 file has a header; older ones do not."""
    rows = [r for r in csv.reader(text.splitlines()) if len(r) >= 2]
    if rows and rows[0][0] == "LabelName":
        rows = rows[1:]
    return {name: label for label, name in rows}


def resolve_classes(display_to_id: dict[str, str], requested: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Return ({label id: display name} for boxable requests, requested names that are not boxable)."""
    lower = {name.lower(): (name, lid) for name, lid in display_to_id.items()}
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in requested:
        hit = lower.get(name.lower())
        if hit:
            found[hit[1]] = hit[0]
        else:
            missing.append(name)
    return found, missing


def iter_bbox_rows(path: Path, wanted: dict[str, str]) -> Iterable[tuple[str, BoxRow]]:
    """Stream the bbox CSV, yielding only rows of wanted labels (the train CSV is 2.3 GB)."""
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["LabelName"] in wanted:
                yield row["ImageID"], BoxRow(
                    row["LabelName"], float(row["XMin"]), float(row["XMax"]), float(row["YMin"]),
                    float(row["YMax"]), row["IsGroupOf"] == "1", row["IsDepiction"] == "1",
                )


def select_images(
    rows: Iterable[tuple[str, BoxRow]], per_class_cap: int, seed: int = 0, max_images: int | None = None,
    ignore_ids: frozenset[str] = frozenset(),
) -> Selection:
    """Pick images so each class reaches ``per_class_cap`` instances; rare classes are visited first.

    An accepted image keeps all its wanted boxes (dropping some would create false background). ``max_images``
    stops selection early in the same seeded, rare-first order, so class counts always match the images kept.
    Only trainable boxes count toward the cap: group-of, depiction and ``ignore_ids`` boxes become ignore
    regions later, so an image with no trainable box is never selected.
    """
    def trainable(b: BoxRow) -> bool:
        return not (b.is_group_of or b.is_depiction or b.label_id in ignore_ids)

    by_image: dict[str, list[BoxRow]] = defaultdict(list)
    for image_id, box in rows:
        by_image[image_id].append(box)
    counted = {i: [b for b in boxes if trainable(b)] for i, boxes in by_image.items()}
    counted = {i: c for i, c in counted.items() if c}
    freq: Counter = Counter()
    for boxes in counted.values():
        freq.update({b.label_id for b in boxes})
    ids = sorted(counted)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda i: min(freq[b.label_id] for b in counted[i]))
    sel = Selection()
    for image_id in ids:
        if all(sel.class_counts[b.label_id] >= per_class_cap for b in counted[image_id]):
            continue
        sel.class_counts.update(b.label_id for b in counted[image_id])
        sel.images[image_id] = by_image[image_id]
        if max_images is not None and len(sel.images) >= max_images:
            break
    return sel


def write_image_list(split: str, image_ids: Iterable[str], path: Path) -> int:
    """Write the official ``<split>/<image id>`` list (usable with the upstream downloader.py too)."""
    lines = [f"{split}/{i}" for i in sorted(image_ids)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def read_image_list(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip().removesuffix(".jpg")
        if not line:
            continue
        m = LIST_LINE.match(line)
        if not m:
            raise ValueError(f'line {n}: not a "<split>/<image id>" entry: {raw!r}')
        pairs.append((m.group(1), m.group(2)))
    return pairs


def make_client() -> S3Client:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Open Images download needs boto3: pip install boto3") from exc
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _fetch_one(client: S3Client, split: str, image_id: str, folder: Path, retries: int,
               sleep: Callable[[float], None]) -> str:
    dest = folder / split / f"{image_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".jpg.part")
    last = ""
    for attempt in range(retries):
        try:
            client.download_file(BUCKET, f"{split}/{image_id}.jpg", str(part))
            part.replace(dest)
            return ""
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = f"{type(exc).__name__}: {exc}"
            sleep(1.5 * (attempt + 1))
    return last


def download_images(
    pairs: list[tuple[str, str]], folder: Path, *, client: S3Client | None = None, workers: int = 5,
    retries: int = 3, sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Download ``pairs`` to ``folder/<split>/<id>.jpg``. Returns (ok, [(split/id, error)])."""
    client = client or make_client()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(lambda p: _fetch_one(client, p[0], p[1], folder, retries, sleep), pairs))
    ok = [p for p, e in zip(pairs, errors, strict=True) if not e]
    failed = [(f"{p[0]}/{p[1]}", e) for p, e in zip(pairs, errors, strict=True) if e]
    return ok, failed


def to_records(
    selection: Selection, wanted: dict[str, str], split: str, root: Path, label_map: LabelMap,
    images_subdir: str = "images",
) -> list[ImageRecord]:
    """Manifest records for downloaded images. Group-of, depiction and label-map ignored boxes become ignore regions."""
    out: list[ImageRecord] = []
    for image_id, boxes in selection.images.items():
        rel = f"{images_subdir}/{split}/{image_id}.jpg"
        if not (root / rel).exists():
            continue
        with Image.open(root / rel) as im:
            w, h = im.size
        recs: list[BoxRecord] = []
        ignores: list[tuple[float, float, float, float]] = []
        for b in boxes:
            xyxy = (b.xmin * w, b.ymin * h, b.xmax * w, b.ymax * h)
            name = wanted[b.label_id]
            if label_map.is_ignored(SOURCE, name):
                ignores.append(xyxy)
                continue
            target = label_map.map(SOURCE, name)
            if target is None:
                continue
            if b.is_group_of or b.is_depiction:
                ignores.append(xyxy)
            else:
                recs.append(BoxRecord(target, xyxy, name))
        if recs:
            out.append(ImageRecord(
                f"{SOURCE}:{image_id}", SOURCE, rel, w, h, f"{SOURCE}:{image_id}", tuple(recs),
                tuple(ignores), license=LICENSE, extra={"split": split},
            ))
    return out
