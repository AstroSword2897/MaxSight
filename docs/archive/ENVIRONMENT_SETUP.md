# Environment Setup Guide

## Overview

This repository supports two environment setups:

1. **Virtual Environment (venv)** - Recommended for development
   - Location: `/Users/nani/2026/venv`
   - Python: 3.10 (Homebrew)
   - Has all dependencies including `torchaudio`

2. **Conda Environment** - Alternative setup
   - Location: `/Users/nani/miniforge3`
   - Python: 3.12
   - Defined in `environment.yml`

## Current Active Environment

To check which Python environment is active:

```bash
python -c "import sys; print('executable:', sys.executable); print('prefix:', sys.prefix); print('base_prefix:', sys.base_prefix)"
```

**Interpretation:**
- If `sys.prefix != sys.base_prefix` → you're inside a venv
- If `sys.prefix` points to `/Users/nani/2026/venv` → using the venv
- If `sys.prefix` points to `/Users/nani/miniforge3` → using conda

## Activating the Virtual Environment

```bash
source /Users/nani/2026/venv/bin/activate
```

After activation, verify:
```bash
which python
# Should show: /Users/nani/2026/venv/bin/python

python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
# Should print the version without errors
```

## Running Scripts

### Option 1: Manual Activation (Recommended)
```bash
source /Users/nani/2026/venv/bin/activate
python scripts/setup_coco_splits.py [args...]
```

### Option 2: Using Helper Script
```bash
./scripts/run_with_venv.sh scripts/setup_coco_splits.py [args...]
```

### Option 3: Direct venv Python
```bash
/Users/nani/2026/venv/bin/python scripts/setup_coco_splits.py [args...]
```

## Troubleshooting

### Issue: Script fails with "ModuleNotFoundError: No module named 'torchaudio'"

**Cause:** Script is running with a different Python interpreter (likely conda).

**Solution:**
1. Activate the venv: `source /Users/nani/2026/venv/bin/activate`
2. Verify: `which python` should show `/Users/nani/2026/venv/bin/python`
3. Re-run the script

### Issue: Shell prompt shows `(base)` but want to use venv

**Explanation:** Conda's base environment is loaded in your shell, but you can still activate the venv. The `(base)` prompt doesn't prevent venv activation.

**Solution:**
```bash
# Deactivate conda if needed (optional)
conda deactivate

# Activate venv
source /Users/nani/2026/venv/bin/activate

# Your prompt may still show (base), but Python will use venv
which python  # Verify it's the venv Python
```

## Environment Verification

Run this diagnostic to confirm everything is set up correctly:

```bash
source /Users/nani/2026/venv/bin/activate
python -c "
import sys
print('='*60)
print('Environment Diagnostic')
print('='*60)
print(f'Python executable: {sys.executable}')
print(f'Python prefix: {sys.prefix}')
print(f'Base prefix: {sys.base_prefix}')
print(f'Is venv: {sys.prefix != sys.base_prefix}')
print()

# Check critical packages
try:
    import torch
    print(f'✅ torch: {torch.__version__}')
except ImportError as e:
    print(f'❌ torch: {e}')

try:
    import torchaudio
    print(f'✅ torchaudio: {torchaudio.__version__}')
except ImportError as e:
    print(f'❌ torchaudio: {e}')

try:
    import torchvision
    print(f'✅ torchvision: {torchvision.__version__}')
except ImportError as e:
    print(f'❌ torchvision: {e}')

print('='*60)
"
```

## Notes

- The `.gitignore` ignores `venv/` directories, which is standard
- The `pyproject.toml` references the venv at `../2026/venv`
- The `environment.yml` defines a conda environment but is optional
- Scripts should work with either environment, but venv is the primary setup

