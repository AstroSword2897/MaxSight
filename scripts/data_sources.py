#!/usr/bin/env python3
"""Status, plans and guarded fetch for the eight public datasets (registry: configs/data/sources.yaml).

  python scripts/data_sources.py status
  python scripts/data_sources.py plan sanpo --session <id> [--segmentation] [--depth]
  python scripts/data_sources.py plan ego4d [--video-uid-file F | --benchmarks EM] [--video]
  python scripts/data_sources.py fetch diode --approve-gb 3        # the only HTTP fetch implemented
  python scripts/data_sources.py fetch sanpo --session <id> --execute --approve-gb 5

Nothing is downloaded unless ``fetch`` is given ``--approve-gb`` large enough for the plan (and
``--execute`` for subprocess-based sources). The default is always a dry run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.datasets.acquisition import fetch  # noqa: E402
from ml.datasets.acquisition import plan as acq_plan  # noqa: E402
from ml.datasets.integrity import verify_raw  # noqa: E402
from ml.datasets.sources import (  # noqa: E402
    ACQUIRED_MARKER,
    DEFAULT_DATA_ROOT,
    load_registry,
    status_label,
    write_acquired_marker,
)


def _plan_for(src, args) -> acq_plan.Plan:
    root = Path(args.data_root)
    if src.id == "sanpo":
        return acq_plan.sanpo_plan(src, args.session or [], segmentation=args.segmentation, depth=args.depth, data_root=root)
    if src.id == "ego4d":
        uid = Path(args.video_uid_file) if args.video_uid_file else None
        return acq_plan.ego4d_plan(src, video_uid_file=uid, benchmarks=args.benchmarks, include_video=args.video, data_root=root)
    if src.id == "diode":
        return acq_plan.diode_plan(src, root)
    return acq_plan.manual_plan(src)


def cmd_status(reg, args) -> int:
    print(f"{'dataset':22s} {'role':10s} {'access':20s} {'commercial':11s} status")
    for s in reg.values():
        print(f"{s.id:22s} {s.role:10s} {s.access_kind:20s} {s.commercial_use:11s} {status_label(s, Path(args.data_root))}")
    return 0


def cmd_verify(reg, args) -> int:
    """Check every image of every acquired split, then record the result in the ACQUIRED marker."""
    raw = Path(args.data_root) / "raw" / args.dataset
    marker = raw / ACQUIRED_MARKER
    if not marker.exists():
        print(f"{args.dataset}: nothing acquired (no {marker})")
        return 1

    info = json.loads(marker.read_text(encoding="utf-8"))
    splits = info.get("splits") or ({info["split"]: info} if "split" in info else {})
    if not splits:
        print(f"{args.dataset}: marker has no per-split manifests to verify")
        return 1

    all_ok = True
    for split, entry in splits.items():
        manifest = Path(entry["manifest"])
        if not manifest.is_absolute():
            manifest = REPO / manifest
        report = verify_raw(manifest, raw)
        all_ok &= report.ok
        print(f"{args.dataset}/{split}: checked {report.checked}, passed {report.passed}, failed {len(report.failures)}")
        for path, reason in report.failures[:10]:
            print(f"  FAILED {path}: {reason}")

        # Keep everything already recorded for the split; update only the verification result.
        kept = {key: value for key, value in entry.items() if key not in ("verified", "split", "integrity")}
        write_acquired_marker(raw, verified=report.ok, split=split, integrity=report.as_dict(), **kept)

    print("RAW_VERIFIED" if all_ok else "verification FAILED")
    return 0 if all_ok else 1


def cmd_plan(reg, args) -> int:
    pl = _plan_for(reg[args.dataset], args)
    print(f"{args.dataset}: est_gb={pl.est_gb}  automatable={pl.automatable}")
    for a in pl.manual:
        print("  MANUAL:", a)
    for n in pl.notes:
        print("  NOTE:", n)
    return fetch.run_plan(pl, dry_run=True)


def cmd_fetch(reg, args) -> int:
    src = reg[args.dataset]
    pl = _plan_for(src, args)
    if pl.manual:
        print("Manual steps required first:")
        for a in pl.manual:
            print("  -", a)
    if pl.est_gb is not None and args.approve_gb < pl.est_gb:
        print(f"Refusing: plan needs ~{pl.est_gb} GB, approved {args.approve_gb} GB.")
        return 2
    if src.id == "diode":
        raw = Path(args.data_root) / "raw" / src.id
        info = fetch.fetch_diode(src.raw, raw)
        print("DIODE acquired; md5-verified:", info["verified"])
        return 0
    if src.id in ("sanpo", "ego4d"):
        if args.approve_gb <= 0 or not args.execute:
            print("Dry run only (needs --execute and --approve-gb > 0).")
            return fetch.run_plan(pl, dry_run=True)
        return fetch.run_plan(pl, dry_run=False)
    print("No automated fetch for this source; follow the manual steps above.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("verify", help="check every acquired image: present, decodes, sha256 unchanged").add_argument("dataset")
    for name in ("plan", "fetch"):
        sp = sub.add_parser(name)
        sp.add_argument("dataset")
        sp.add_argument("--session", action="append")
        sp.add_argument("--segmentation", action="store_true")
        sp.add_argument("--depth", action="store_true")
        sp.add_argument("--video-uid-file")
        sp.add_argument("--benchmarks", nargs="*")
        sp.add_argument("--video", action="store_true")
        if name == "fetch":
            sp.add_argument("--approve-gb", type=float, default=0.0)
            sp.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = load_registry()
    if args.cmd != "status" and args.dataset not in reg:
        ap.error(f"unknown dataset {args.dataset!r}; choose from {sorted(reg)}")
    try:
        commands = {"status": cmd_status, "verify": cmd_verify, "plan": cmd_plan, "fetch": cmd_fetch}
        return commands[args.cmd](reg, args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
