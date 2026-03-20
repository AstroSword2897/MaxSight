"""Validate video panoptic clip manifests (v1 schema)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

MANIFEST_SCHEMA_VERSION = "1.0"
CONTRACT_FIXED_STRIDE_T8 = "fixed_stride_t8"


def validate_manifest_v1(
    data: Dict[str, Any],
    *,
    require_fixed_t8: bool = False,
) -> List[str]:
    """Return human-readable errors; empty list means the manifest is usable."""

    errors: List[str] = []
    if not isinstance(data, dict):
        return ["root must be an object"]
    ver = data.get("schema_version")
    if ver != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, got {ver!r}")
    clips = data.get("clips")
    if not isinstance(clips, list):
        errors.append("clips must be an array")
        return errors

    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"clips[{i}] must be an object")
            continue
        errs = _validate_clip(clip, index=i)
        errors.extend(errs)

    if require_fixed_t8:
        for i, clip in enumerate(clips):
            if not isinstance(clip, dict):
                continue
            tw = clip.get("temporal_window")
            paths = clip.get("frame_paths")
            segs = clip.get("frames_segments")
            start = clip.get("start_frame")
            end = clip.get("end_frame")
            if tw != 8:
                errors.append(f"clips[{i}]: fixed_stride_t8 requires temporal_window==8, got {tw!r}")
            if isinstance(paths, list) and len(paths) != 8:
                errors.append(f"clips[{i}]: fixed_stride_t8 requires len(frame_paths)==8, got {len(paths)}")
            if isinstance(segs, list) and len(segs) != 8:
                errors.append(f"clips[{i}]: fixed_stride_t8 requires len(frames_segments)==8, got {len(segs)}")
            if isinstance(start, int) and isinstance(end, int) and end - start != 8:
                errors.append(
                    f"clips[{i}]: fixed_stride_t8 requires end_frame - start_frame == 8, got {end}-{start}"
                )

    return errors


def _validate_clip(clip: Dict[str, Any], index: int) -> List[str]:
    errors: List[str] = []
    prefix = f"clips[{index}]"
    required = (
        "clip_id",
        "video_id",
        "start_frame",
        "end_frame",
        "temporal_window",
        "temporal_stride",
        "temporal_overlap",
        "frame_paths",
        "frames_segments",
    )
    for key in required:
        if key not in clip:
            errors.append(f"{prefix}: missing {key!r}")

    start = clip.get("start_frame")
    end = clip.get("end_frame")
    tw = clip.get("temporal_window")
    paths = clip.get("frame_paths")
    segs = clip.get("frames_segments")

    if isinstance(start, int) and isinstance(end, int):
        if end <= start:
            errors.append(f"{prefix}: end_frame must be > start_frame")
        elif isinstance(tw, int) and tw != end - start:
            errors.append(
                f"{prefix}: temporal_window ({tw}) must equal end_frame - start_frame ({end - start})"
            )

    if isinstance(paths, list) and isinstance(tw, int) and len(paths) != tw:
        errors.append(f"{prefix}: len(frame_paths) ({len(paths)}) must equal temporal_window ({tw})")

    if isinstance(segs, list) and isinstance(tw, int) and len(segs) != tw:
        errors.append(
            f"{prefix}: len(frames_segments) ({len(segs)}) must equal temporal_window ({tw})"
        )

    if isinstance(segs, list):
        for fi, frame_segs in enumerate(segs):
            if not isinstance(frame_segs, list):
                errors.append(f"{prefix}: frames_segments[{fi}] must be an array")
                continue
            for si, seg in enumerate(frame_segs):
                if not isinstance(seg, dict):
                    errors.append(f"{prefix}: frames_segments[{fi}][{si}] must be an object")
                    continue
                bbox = seg.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    errors.append(f"{prefix}: frames_segments[{fi}][{si}].bbox must be length-4 array")

    return errors


def clip_spans(clip: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return (start_frame, end_frame) if valid integers are present."""

    s, e = clip.get("start_frame"), clip.get("end_frame")
    if isinstance(s, int) and isinstance(e, int):
        return s, e
    return None


def iter_clip_frame_paths(
    clip: Dict[str, Any],
    manifest_root: Optional[Any] = None,
) -> Sequence[str]:
    """Return frame_paths list from a clip (for callers that resolve paths themselves)."""

    _ = manifest_root
    paths = clip.get("frame_paths")
    if isinstance(paths, list):
        return [str(p) for p in paths]
    return []
