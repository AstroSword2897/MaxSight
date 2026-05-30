"""Deterministic assistive labels: class priors, bbox proximity, and center bias.

Single source for gold adapters, MaxSightDataset, and generate_annotations so
training targets stay reproducible when COCO-style boxes are the only signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT / "ml" / "config" / "assistive_supervision.yaml"


@dataclass(frozen=True)
class AssistiveSupervisionSpec:
    """Frozen parameters for derived urgency and distance zones."""

    w_class: float
    w_proximity: float
    w_center: float
    urgency_bin_edges: Tuple[float, float, float]
    distance_area_thresholds: Tuple[float, float]
    default_class_prior: float
    class_prior_overrides: Mapping[str, float]


def _spec_from_flat_dict(data: Mapping[str, Any]) -> AssistiveSupervisionSpec:
    w = data.get("weights") or {}
    if not isinstance(w, dict):
        raise TypeError("weights must be a dict")
    w = cast(Dict[str, Any], w)
    edges = tuple(float(x) for x in (data.get("urgency_bin_edges") or [0.30, 0.50, 0.68]))
    if len(edges) != 3:
        raise ValueError("urgency_bin_edges must have exactly 3 values (4 urgency levels).")
    dth = tuple(float(x) for x in (data.get("distance_area_thresholds") or [0.10, 0.05]))
    if len(dth) != 2:
        raise ValueError("distance_area_thresholds must have exactly 2 values.")
    ov = data.get("class_prior_overrides") or {}
    if not isinstance(ov, dict):
        raise ValueError("class_prior_overrides must be a string->float mapping.")
    overrides = {str(k): float(v) for k, v in ov.items()}
    return AssistiveSupervisionSpec(
        w_class=float(w.get("class", 0.52)),
        w_proximity=float(w.get("proximity", 0.33)),
        w_center=float(w.get("center", 0.15)),
        urgency_bin_edges=(edges[0], edges[1], edges[2]),
        distance_area_thresholds=(dth[0], dth[1]),
        default_class_prior=float(data.get("default_class_prior", 0.38)),
        class_prior_overrides=overrides,
    )


def _default_yaml_dict() -> Dict:
    return {
        "weights": {"class": 0.52, "proximity": 0.33, "center": 0.15},
        "urgency_bin_edges": [0.30, 0.50, 0.68],
        "distance_area_thresholds": [0.10, 0.05],
        "default_class_prior": 0.38,
        "class_prior_overrides": {},
    }


def _merge_yaml_into_base(base: Dict, loaded: Mapping[str, object]) -> Dict:
    out = dict(base)
    for k, v in loaded.items():
        if k == "weights" and isinstance(v, dict) and isinstance(out.get("weights"), dict):
            out["weights"] = {**out["weights"], **v}
        elif k == "class_prior_overrides" and isinstance(v, dict) and isinstance(out.get("class_prior_overrides"), dict):
            out["class_prior_overrides"] = {**out["class_prior_overrides"], **v}
        else:
            out[k] = v
    return out


@lru_cache(maxsize=8)
def _load_assistive_spec_cached(resolved_key: str) -> AssistiveSupervisionSpec:
    base = _default_yaml_dict()
    if resolved_key == "__defaults__":
        return _spec_from_flat_dict(base)
    chosen = Path(resolved_key)
    if not chosen.is_file():
        return _spec_from_flat_dict(base)
    try:
        import yaml  # type: ignore
    except ImportError:
        return _spec_from_flat_dict(base)
    with chosen.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        return _spec_from_flat_dict(base)
    merged = _merge_yaml_into_base(base, loaded)
    return _spec_from_flat_dict(merged)


def load_assistive_spec(path: Optional[str] = None) -> AssistiveSupervisionSpec:
    """Load YAML spec; fall back to defaults. Cached per resolved file path."""
    env_path = os.environ.get("MAXSIGHT_ASSISTIVE_SPEC_PATH", "").strip()
    if path:
        key = str(Path(path).resolve())
    elif env_path:
        key = str(Path(env_path).resolve())
    elif _DEFAULT_YAML.is_file():
        key = str(_DEFAULT_YAML.resolve())
    else:
        key = "__defaults__"
    return _load_assistive_spec_cached(key)


@lru_cache(maxsize=1)
def _coco_base_priors() -> Dict[str, float]:
    """Static risk table for COCO-80 names (exact string keys as in maxsight_cnn)."""
    from ml.models.maxsight_cnn import COCO_BASE_CLASSES

    high = {
        "car",
        "bus",
        "truck",
        "train",
        "motorcycle",
        "traffic light",
        "fire hydrant",
        "stop sign",
    }
    med_high = {
        "person",
        "bicycle",
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "zebra",
        "giraffe",
        "bear",
        "airplane",
        "boat",
    }
    medium = {
        "chair",
        "couch",
        "bed",
        "dining table",
        "bench",
        "toilet",
        "sink",
    }
    out: Dict[str, float] = {}
    for c in COCO_BASE_CLASSES:
        if c in high:
            out[c] = 0.90
        elif c in med_high:
            out[c] = 0.72
        elif c in medium:
            out[c] = 0.52
        else:
            out[c] = 0.34
    return out


def _keyword_boost_prior(name: str) -> Optional[float]:
    """Fallback for extended class names not in COCO-80."""
    n = name.lower()
    if any(k in n for k in ("traffic", "crosswalk", "vehicle", "collision", "emergency", "fire_", "stair", "escalator")):
        return 0.78
    if any(k in n for k in ("sign", "exit", "warning", "hazard", "braille", "elevator", "door", "ramp", "curb")):
        return 0.58
    if any(k in n for k in ("food", "vase", "remote", "clock", "book", "toothbrush", "hair drier")):
        return 0.22
    return None


def class_prior(category_name: str, spec: AssistiveSupervisionSpec) -> float:
    """Return [0,1] class risk prior for a category string."""
    raw = (category_name or "unknown").strip()
    if not raw:
        return spec.default_class_prior
    low = raw.lower()
    for k, v in spec.class_prior_overrides.items():
        if k.lower() == low:
            return float(v)
    base = _coco_base_priors()
    if raw in base:
        return base[raw]
    for k, v in base.items():
        if k.lower() == low:
            return v
    bump = _keyword_boost_prior(raw)
    if bump is not None:
        return bump
    return spec.default_class_prior


def center_weight(cx: float, cy: float) -> float:
    """Higher when the box center is near the image center (navigation relevance)."""
    cx = max(0.0, min(1.0, float(cx)))
    cy = max(0.0, min(1.0, float(cy)))
    dist = abs(cx - 0.5) + abs(cy - 0.5)
    return max(0.0, min(1.0, 1.0 - dist))


def proximity_from_area(area: float) -> float:
    """Monotonic [0,1]: larger normalized bbox area reads as closer / more pressing."""
    a = max(1e-8, float(area))
    return max(0.0, min(1.0, sqrt(a / 0.25)))


def continuous_urgency_score(
    cx: float,
    cy: float,
    w: float,
    h: float,
    category_name: str,
    spec: AssistiveSupervisionSpec,
) -> float:
    """Weighted combination in [0,1] before binning to discrete urgency levels."""
    area = max(1e-8, float(w) * float(h))
    p = class_prior(category_name, spec)
    prox = proximity_from_area(area)
    ctr = center_weight(cx, cy)
    s = spec.w_class * p + spec.w_proximity * prox + spec.w_center * ctr
    return max(0.0, min(1.0, s))


def distance_zone_from_area(area: float, spec: AssistiveSupervisionSpec) -> int:
    """Near=0, medium=1, far=2 using normalized area thresholds."""
    t0, t1 = spec.distance_area_thresholds
    a = float(area)
    if a > t0:
        return 0
    if a > t1:
        return 1
    return 2


def urgency_level_from_score(score: float, spec: AssistiveSupervisionSpec) -> int:
    """Map continuous score to integer urgency 0..3."""
    e0, e1, e2 = spec.urgency_bin_edges
    level = 0
    if score > e0:
        level += 1
    if score > e1:
        level += 1
    if score > e2:
        level += 1
    return min(3, level)


def object_distance_and_urgency(
    cx: float,
    cy: float,
    w: float,
    h: float,
    category_name: str,
    spec: Optional[AssistiveSupervisionSpec] = None,
) -> Tuple[int, int]:
    """Return (distance_zone, urgency_level) for one normalized box."""
    sp = spec or load_assistive_spec()
    area = max(1e-8, float(w) * float(h))
    dz = distance_zone_from_area(area, sp)
    score = continuous_urgency_score(cx, cy, w, h, category_name, sp)
    ul = urgency_level_from_score(score, sp)
    return dz, ul


def scene_urgency_from_objects(object_urgencies: Iterable[int]) -> int:
    """Aggregate scene urgency as max over objects (existing convention)."""
    levels = [int(u) for u in object_urgencies]
    return max(levels) if levels else 0


def clear_spec_cache() -> None:
    """Drop cached specs (tests only)."""
    _load_assistive_spec_cached.cache_clear()
