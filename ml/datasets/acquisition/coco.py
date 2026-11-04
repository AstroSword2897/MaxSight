"""COCO 2017 subset acquisition: select by class, cap per class, download only those images.

Avoids the 19 GB train2017 download. Annotations (~250 MB zip) are fetched once;
images come from each record's ``coco_url`` with a hard byte budget.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ml.datasets.acquisition.fetch import download_resumable, file_hash
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
SOURCE = "coco"
LICENSE = "annotations CC BY 4.0; images per Flickr terms"
_UA = {"User-Agent": "maxsight-data/1.0"}


def fetch_annotations(dest: Path, split: str = "train2017", url: str = ANNOTATIONS_URL) -> Path:
    """Extract instances_<split>.json, keeping the annotations zip on disk.

    The zip is downloaded once (resumable, never replaced) and read from disk, so a second split
    neither downloads ~250 MB again nor holds the archive in memory.
    """
    target = dest / f"instances_{split}.json"
    if target.exists():
        return target
    archive = download_resumable(url, dest / Path(url).name)
    with zipfile.ZipFile(archive) as zf:
        target.write_bytes(zf.read(f"annotations/instances_{split}.json"))
    return target


def select_records(
    annotations: dict,
    label_map: LabelMap,
    per_class_cap: int,
    seed: int = 0,
    min_box_px: float = 8.0,
) -> list[ImageRecord]:
    """Pick images so every class is represented up to ``per_class_cap`` instances.

    An image is accepted if at least one of its classes is still under the cap, and then keeps
    *all* of its mapped boxes (dropping boxes would create false background). Rare classes are
    visited first so common ones (person) cannot crowd them out.
    """
    cats = {c["id"]: c["name"] for c in annotations["categories"]}
    images = {im["id"]: im for im in annotations["images"]}
    # Each COCO image has its own Flickr licence (many are NonCommercial or NoDerivs).
    licence_names = {lic["id"]: lic["name"] for lic in annotations.get("licenses", [])}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations["annotations"]:
        by_image[ann["image_id"]].append(ann)

    mapped: dict[int, tuple[list[BoxRecord], list[tuple[float, float, float, float]], list[str]]] = {}
    freq: Counter[str] = Counter()
    for image_id, anns in by_image.items():
        boxes: list[BoxRecord] = []
        ignores: list[tuple[float, float, float, float]] = []
        box_annotation_ids: list[str] = []  # original COCO ids of the kept boxes, in order ("" if absent)
        for ann in anns:
            name = cats[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            xyxy = (float(x), float(y), float(x + w), float(y + h))
            target = label_map.map(SOURCE, name)
            if ann.get("iscrowd") or target is None:
                if ann.get("iscrowd") or label_map.is_ignored(SOURCE, name):
                    ignores.append(xyxy)
                continue
            if w < min_box_px or h < min_box_px:
                ignores.append(xyxy)  # real but too small to train on: ignore it, never leave it as background
                continue
            boxes.append(BoxRecord(target, xyxy, name))
            box_annotation_ids.append(str(ann.get("id", "")))
        if boxes:
            mapped[image_id] = (boxes, ignores, box_annotation_ids)
            freq.update({b.cls for b in boxes})

    rarity = {c: freq.get(c, 0) for c in label_map.classes}
    ids = sorted(mapped)
    random.Random(seed).shuffle(ids)
    # Visit images holding the rarest class first.
    ids.sort(key=lambda i: min(rarity[b.cls] for b in mapped[i][0]))

    counts: Counter[str] = Counter()
    out: list[ImageRecord] = []
    for image_id in ids:
        boxes, ignores, box_annotation_ids = mapped[image_id]
        if all(counts[b.cls] >= per_class_cap for b in boxes):
            continue
        counts.update(b.cls for b in boxes)
        im = images[image_id]
        provenance = {
            "image_license": licence_names.get(im.get("license"), "unknown"),
            "flickr_url": im.get("flickr_url", ""),
            "annotation_ids": ",".join(box_annotation_ids),
        }
        out.append(
            ImageRecord(
                image_id=f"coco:{image_id}",
                source=SOURCE,
                path=f"images/{im['file_name']}",
                width=int(im["width"]),
                height=int(im["height"]),
                group_id=f"coco:{image_id}",
                boxes=tuple(boxes),
                ignore_regions=tuple(ignores),
                url=im.get("coco_url", ""),
                license=LICENSE,
                extra=provenance,
            )
        )
    return out


def manifest_path(root: Path, split: str) -> Path:
    """One manifest per split, so acquiring val2017 never overwrites the train2017 records."""
    return root / f"manifest_{split}.jsonl"


def _download_one(rec: ImageRecord, root: Path, retries: int = 3) -> tuple[ImageRecord, int, str]:
    """Download one image. Returns (record with its sha256 filled in, bytes on disk, error or "")."""
    dest = root / rec.path

    # Resumed run: the file is already here. Hash it so the manifest still records its sha256.
    if dest.exists() and dest.stat().st_size > 0:
        return rec.with_(sha256=file_hash(dest, "sha256")), dest.stat().st_size, ""

    dest.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(rec.url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                data = resp.read()
            # Write to a .part file first, so an interrupted write never looks like a finished image.
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return rec.with_(sha256=hashlib.sha256(data).hexdigest()), len(data), ""
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return rec, 0, last


def download_images(
    records: list[ImageRecord], root: Path, max_gb: float, workers: int = 8
) -> tuple[list[ImageRecord], list[tuple[str, str]]]:
    """Download until the byte budget is reached. Returns (downloaded records, failures).

    Records left undownloaded when the budget runs out are returned as failures ("byte budget reached"),
    so a truncated subset never looks complete. The budget is checked per batch of ``workers * 4``
    images, so a run may overshoot it by at most one batch.
    """
    budget = int(max_gb * 1024**3)
    used = 0
    ok: list[ImageRecord] = []
    failed: list[tuple[str, str]] = []
    batch = workers * 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(records), batch):
            if used >= budget:
                failed.extend((rec.image_id, "byte budget reached") for rec in records[i:])
                break
            for rec, size, err in pool.map(lambda r: _download_one(r, root), records[i : i + batch]):
                if err:
                    failed.append((rec.image_id, err))
                else:
                    ok.append(rec)
                    used += size
    return ok, failed


def load_annotations(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
