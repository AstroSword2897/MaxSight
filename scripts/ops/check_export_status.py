#!/usr/bin/env python3
"""Check whether all top 7 condition models are exported (JIT/PTE and/or CoreML). Verifies files on disk and manifest."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOP7 = [
    "amblyopia",
    "amd",
    "color_blindness",
    "cvi",
    "glaucoma",
    "retinitis_pigmentosa",
    "strabismus",
]


def main():
    parser = argparse.ArgumentParser(description="Check if all top 7 models are exported.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "exports" / "top7",
        help="Export root (default: exports/top7)",
    )
    parser.add_argument(
        "--coreml-only", action="store_true", help="Success = all 7 have CoreML (ignore JIT/PTE)"
    )
    parser.add_argument("--quiet", action="store_true", help="Only print summary and exit code")
    args = parser.parse_args()

    export_root = Path(args.output_dir).resolve()
    manifest_path = export_root / "manifest.json"
    checkpoints_base = REPO / "checkpoints"

    if not export_root.exists():
        if not args.quiet:
            print("Export directory missing:", export_root)
            print("Run: python scripts/export_top7_to_xcode.py")
            print("\nCheckpoint status:")
            for c in TOP7:
                ckpt = checkpoints_base / f"checkpoints_{c}" / "best_model.pt"
                print(f"  {c}: {'exists' if ckpt.exists() else 'MISSING'}")
        return 1

    if not manifest_path.exists():
        if not args.quiet:
            print("Manifest missing:", manifest_path)
            print("Export may have failed before writing manifest. Re-run export.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ok = True
    for c in TOP7:
        info = manifest.get("conditions", {}).get(c, {})
        cond_dir = export_root / c
        has_jit = (cond_dir / "maxsight_traced.pt").exists() or (cond_dir / "maxsight.pte").exists()
        coreml_path = info.get("coreml_path") or str(cond_dir / f"{c}.mlpackage")
        has_coreml = (
            Path(coreml_path).exists() if coreml_path else (cond_dir / f"{c}.mlpackage").exists()
        )
        inference_ok = info.get("inference_ok", False)
        err = info.get("error")

        ok = has_coreml and (has_jit or getattr(args, "coreml_only", False))
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISSING"
        if not args.quiet:
            detail = f"  jit={has_jit} coreml={has_coreml}"
            if err:
                detail += f"  error={err}"
            print(f"  {c}: {status}{detail}")

    if not args.quiet:
        count = sum(
            1
            for c in TOP7
            if (export_root / c / "maxsight_traced.pt").exists()
            or (export_root / c / "maxsight.pte").exists()
        )
        coreml_count = sum(1 for c in TOP7 if (export_root / c / f"{c}.mlpackage").exists())
        mode = "CoreML-only" if getattr(args, "coreml_only", False) else "JIT/PTE + CoreML"
        print(f"\nExported: {count}/7 JIT/PTE, {coreml_count}/7 CoreML  ({mode}, all OK: {all_ok})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
