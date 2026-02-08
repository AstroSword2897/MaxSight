#!/usr/bin/env python3
"""Single entry point: export the top 7 condition models to Xcode-ready bundles. Uses JIT-only and CPU to reduce crashes."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if __name__ == "__main__":
    # Run deploy_top7 with safe defaults so one run completes.
    sys.argv = [
        sys.argv[0],
        "--output-dir", str(REPO / "exports" / "top7"),
        "--device", "cpu",
        "--quick",
    ]
    import scripts.deploy_top7 as deploy
    deploy.main()

