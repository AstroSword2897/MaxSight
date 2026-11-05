"""Session-wide test setup.

On macOS, torch and faiss-cpu each bundle their own OpenMP runtime. With more than one OpenMP thread the two
runtimes crash the process (segmentation fault in faiss index building, reproduced with torch 2.14.1 and
faiss-cpu 1.15.1). Limiting OpenMP to one thread avoids it. Linux CI images share one runtime and are unaffected.
The variable must be set before torch or faiss is imported, which is why it lives here.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
