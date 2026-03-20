"""Utilities for sequence-native video panoptic supervision.

This module is intentionally dataset-agnostic so open video sources can share
the same clip sampling, quality filtering, and temporal track construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Sampling settings for fixed-stride temporal windows."""

    temporal_window: int = 8
    temporal_stride: int = 1
    temporal_overlap: int = 0

    def validate(self) -> None:
        if self.temporal_window < 2:
            raise ValueError("temporal_window must be >= 2")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be >= 1")
        if self.temporal_overlap < 0:
            raise ValueError("temporal_overlap must be >= 0")
        if self.temporal_overlap >= self.temporal_window:
            raise ValueError("temporal_overlap must be < temporal_window")

    @property
    def step(self) -> int:
        return self.temporal_window - self.temporal_overlap


@dataclass(frozen=True)
class PseudoPanopticQualityConfig:
    """Quality gates for pseudo-panoptic segments."""

    min_confidence: float = 0.45
    min_area_pixels: float = 24.0
    min_bbox_width: float = 2.0
    min_bbox_height: float = 2.0

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_area_pixels < 0:
            raise ValueError("min_area_pixels must be >= 0")
        if self.min_bbox_width <= 0 or self.min_bbox_height <= 0:
            raise ValueError("bbox thresholds must be > 0")


@dataclass(frozen=True)
class AdaptiveTemporalConfig:
    """Adaptive temporal windowing configuration."""

    t_min: int = 4
    t_max: int = 16
    smooth_factor: float = 0.2
    alpha_iou: float = 0.5
    beta_displacement: float = 0.5
    image_size_norm: float = 256.0
    overlap_ratio: float = 0.25

    def validate(self) -> None:
        if self.t_min < 2:
            raise ValueError("t_min must be >= 2")
        if self.t_max < self.t_min:
            raise ValueError("t_max must be >= t_min")
        if not (0.0 <= self.smooth_factor <= 1.0):
            raise ValueError("smooth_factor must be in [0, 1]")
        if self.alpha_iou < 0 or self.beta_displacement < 0:
            raise ValueError("alpha_iou and beta_displacement must be >= 0")
        if (self.alpha_iou + self.beta_displacement) <= 0:
            raise ValueError("alpha_iou + beta_displacement must be > 0")
        if self.image_size_norm <= 0:
            raise ValueError("image_size_norm must be > 0")
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0, 1)")


def build_fixed_stride_windows(
    num_frames: int,
    config: VideoSamplingConfig,
) -> List[Tuple[int, int]]:
    """Build [start, end) frame windows with overlap support."""

    config.validate()
    if num_frames <= 0:
        return []
    if num_frames < config.temporal_window:
        return []

    windows: List[Tuple[int, int]] = []
    start = 0
    while start + config.temporal_window <= num_frames:
        end = start + config.temporal_window
        windows.append((start, end))
        start += config.step
    return windows


def iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    """Yield sequence slices for scalable chunked preprocessing."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def prune_pseudo_segments(
    segments_info: Sequence[Dict[str, Any]],
    config: PseudoPanopticQualityConfig,
) -> List[Dict[str, Any]]:
    """Drop low-quality pseudo-panoptic segments."""

    config.validate()
    kept: List[Dict[str, Any]] = []
    for seg in segments_info:
        conf = float(seg.get("score", seg.get("confidence", 1.0)))
        area = float(seg.get("area", 0.0))
        bbox = seg.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        _, _, w, h = [float(v) for v in bbox]

        if conf < config.min_confidence:
            continue
        if area < config.min_area_pixels:
            continue
        if w < config.min_bbox_width or h < config.min_bbox_height:
            continue
        kept.append(dict(seg))
    return kept


def _xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU for [x, y, w, h] boxes."""

    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _center_displacement_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return (dx * dx + dy * dy) ** 0.5


def compute_motion_score(
    prev_frame: Sequence[Dict[str, Any]],
    curr_frame: Sequence[Dict[str, Any]],
    alpha_iou: float = 0.5,
    beta_displacement: float = 0.5,
    image_size_norm: float = 256.0,
) -> float:
    """Compute normalized motion score from segment bboxes.

    Score is in [0, 1], where higher indicates more motion/instability.
    """
    if not prev_frame or not curr_frame:
        return 1.0

    # Greedy matching by best IoU for each current segment.
    ious: List[float] = []
    disps: List[float] = []
    for cur in curr_frame:
        cur_bbox = cur.get("bbox")
        if not isinstance(cur_bbox, (list, tuple)) or len(cur_bbox) != 4:
            continue
        best_iou = 0.0
        best_prev_bbox = None
        for prev in prev_frame:
            prev_bbox = prev.get("bbox")
            if not isinstance(prev_bbox, (list, tuple)) or len(prev_bbox) != 4:
                continue
            score = bbox_iou_xywh(prev_bbox, cur_bbox)
            if score > best_iou:
                best_iou = score
                best_prev_bbox = prev_bbox
        if best_prev_bbox is not None:
            ious.append(best_iou)
            disps.append(_center_displacement_xywh(best_prev_bbox, cur_bbox))

    if not ious:
        return 1.0

    mean_iou = sum(ious) / len(ious)
    mean_disp = sum(disps) / len(disps) if disps else image_size_norm
    norm_disp = min(1.0, max(0.0, mean_disp / max(1e-6, image_size_norm)))
    total = max(1e-6, alpha_iou + beta_displacement)
    score = (alpha_iou * (1.0 - mean_iou) + beta_displacement * norm_disp) / total
    return max(0.0, min(1.0, float(score)))


def motion_to_temporal_window(
    motion_score: float,
    config: AdaptiveTemporalConfig,
    t_prev: int | None = None,
) -> int:
    """Map motion score to temporal window length with smoothing."""
    config.validate()
    ms = max(0.0, min(1.0, float(motion_score)))
    t_new = int(round(config.t_max - (config.t_max - config.t_min) * ms))
    if t_prev is None:
        return max(config.t_min, min(config.t_max, t_new))
    t_smooth = int(round((1.0 - config.smooth_factor) * float(t_prev) + config.smooth_factor * float(t_new)))
    return max(config.t_min, min(config.t_max, t_smooth))


def build_adaptive_windows(
    frames_segments: Sequence[Sequence[Dict[str, Any]]],
    config: AdaptiveTemporalConfig,
) -> List[Tuple[int, int]]:
    """Build adaptive [start, end) windows from per-frame motion."""
    config.validate()
    n = len(frames_segments)
    if n <= 0:
        return []
    windows: List[Tuple[int, int]] = []
    i = 0
    t_prev: int | None = None
    while i < n:
        if i == 0:
            motion_score = 0.0
        else:
            motion_score = compute_motion_score(
                frames_segments[i - 1],
                frames_segments[i],
                alpha_iou=config.alpha_iou,
                beta_displacement=config.beta_displacement,
                image_size_norm=config.image_size_norm,
            )
        t_cur = motion_to_temporal_window(motion_score, config, t_prev=t_prev)
        t_prev = t_cur
        end = min(n, i + t_cur)
        windows.append((i, end))
        overlap = int(round(config.overlap_ratio * t_cur))
        step = max(1, t_cur - overlap)
        i += step
    return windows


def associate_tracks_multi_frame(
    frames_segments: Sequence[Sequence[Dict[str, Any]]],
    lookback: int = 2,
    iou_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    """Assign robust temporal track ids using multi-frame IoU association.

    Input is per-frame segment dicts where each segment has `bbox`.
    Output preserves structure and adds `track_proxy_id` to each segment.
    """

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    next_track_id = 1
    history: List[List[Dict[str, Any]]] = []
    out: List[List[Dict[str, Any]]] = []

    for frame_idx, segments in enumerate(frames_segments):
        frame_out: List[Dict[str, Any]] = []
        for seg in segments:
            cur = dict(seg)
            bbox = cur.get("bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                cur["track_proxy_id"] = next_track_id
                next_track_id += 1
                frame_out.append(cur)
                continue

            best_iou = 0.0
            best_track_id = None
            for back in range(1, lookback + 1):
                prev_idx = frame_idx - back
                if prev_idx < 0 or prev_idx >= len(history):
                    continue
                for prev in history[prev_idx]:
                    prev_bbox = prev.get("bbox")
                    if not isinstance(prev_bbox, (list, tuple)) or len(prev_bbox) != 4:
                        continue
                    score = bbox_iou_xywh(bbox, prev_bbox)
                    if score > best_iou:
                        best_iou = score
                        best_track_id = prev.get("track_proxy_id")

            if best_track_id is not None and best_iou >= iou_threshold:
                cur["track_proxy_id"] = int(best_track_id)
            else:
                cur["track_proxy_id"] = next_track_id
                next_track_id += 1
            frame_out.append(cur)

        out.append(frame_out)
        history.append(frame_out)
    return out

