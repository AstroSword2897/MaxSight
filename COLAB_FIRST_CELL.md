# Colab: run this first

If you see **"not a git repository"** or **"can't open file '/content/scripts/...'"**, the notebook is using `/content` and the repo is not cloned yet. Run the cells below **in order** before any script that uses `scripts/` or `$REPO`.

## Cell 1 – Mount Drive (optional) and clone repo

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone -q -b feature/multimodal_refactor https://github.com/AstroSword2897/2026-Prototype.git
%cd /content/2026-Prototype
```

## Cell 2 – Run your script from inside the repo

Always use `%cd /content/2026-Prototype` (or run from a cell that already did that), then run scripts by name:

```python
%cd /content/2026-Prototype
!python scripts/inference_and_deploy_top7.py \
  --checkpoints-base /content/drive/MyDrive/MaxSight \
  --output-dir /content/drive/MyDrive/MaxSight/exports_top7
```

**Why:** The repo and `.git` live in `/content/2026-Prototype`. Scripts like `inference_and_deploy_top7.py` must be run as `scripts/inference_and_deploy_top7.py` with current directory `/content/2026-Prototype`, not from `/content`.

More commands: **TOP7_REFERENCE.md** §4, **COLAB_RUNBOOK.md**.
