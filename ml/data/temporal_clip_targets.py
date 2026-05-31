"""Temporal supervision proxies from per-frame pseudo-panoptic segments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml.data.video_panoptic import bbox_iou_xywh


@dataclass(frozen=True)
class TemporalClipTargets:
    """Scalar targets for a single clip, aligned with therapy/simulator-style signals."""

    temporal_consistency: float
    flicker_proxy: float


def derive_temporal_clip_targets(
    frames_segments: Sequence[Sequence[dict[str, Any]]],
) -> TemporalClipTargets:
    """Derive stability and flicker proxies from track-linked segments.

    Higher temporal_consistency means matched tracks stay spatially aligned (mean IoU).
    Higher flicker_proxy mixes IoU drop and relative change in segment counts between frames.
    """

    T = len(frames_segments)
    if T < 2:
        return TemporalClipTargets(1.0, 0.0)

    ious: list[float] = []
    count_ratios: list[float] = []

    for t in range(1, T):
        prev = frames_segments[t - 1]
        cur = frames_segments[t]
        if not isinstance(prev, (list, tuple)) or not isinstance(cur, (list, tuple)):
            continue

        n_prev, n_cur = len(prev), len(cur)
        denom = max(1, n_prev + n_cur)
        count_ratios.append(abs(n_cur - n_prev) / denom)

        prev_by_track: dict[Any, dict[str, Any]] = {}
        for s in prev:
            if isinstance(s, dict) and "track_proxy_id" in s:
                prev_by_track[s["track_proxy_id"]] = s

        for s in cur:
            if not isinstance(s, dict):
                continue
            tid = s.get("track_proxy_id")
            pb = prev_by_track.get(tid) if tid is not None else None
            if pb is None:
                continue
            ba = pb.get("bbox")
            bb = s.get("bbox")
            if (
                isinstance(ba, (list, tuple))
                and len(ba) == 4
                and isinstance(bb, (list, tuple))
                and len(bb) == 4
            ):
                ious.append(bbox_iou_xywh(ba, bb))

    mean_iou = sum(ious) / len(ious) if ious else 1.0
    mean_count_jump = sum(count_ratios) / len(count_ratios) if count_ratios else 0.0
    flicker_proxy = max(0.0, min(1.0, (1.0 - mean_iou) * 0.5 + mean_count_jump * 0.5))
    temporal_consistency = max(0.0, min(1.0, float(mean_iou)))

    return TemporalClipTargets(
        temporal_consistency=temporal_consistency,
        flicker_proxy=flicker_proxy,
    )
