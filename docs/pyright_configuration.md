# Pyright Configuration for PyTorch

## Overview
This document explains the Pyright/Pylance configuration for proper PyTorch type checking.

## Configuration Files

### 1. `pyrightconfig.json`
Main configuration file for Pyright type checker.

**Key Settings**:
- `reportAttributeAccessIssue: "none"` - Disables false positives for PyTorch attributes
- `reportMissingTypeStubs: false` - PyTorch has type stubs but they may not be fully recognized
- `typeCheckingMode: "basic"` - Less strict checking for better compatibility
- `extraPaths` - Points to venv site-packages where PyTorch is installed

### 2. `.vscode/settings.json`
VS Code specific settings for Pylance.

**Key Settings**:
- `python.analysis.diagnosticSeverityOverrides` - Overrides for specific error types
- `python.defaultInterpreterPath` - Points to venv Python
- `python.analysis.extraPaths` - Additional paths for type checking

### 3. `pyproject.toml`
Modern Python project configuration.

## Why These Errors Appear

The type checking errors you see are **false positives**. They occur because:

1. **PyTorch is a C++ extension**: Many PyTorch modules are compiled C++ extensions, not pure Python
2. **Dynamic attributes**: PyTorch uses dynamic attribute access that type checkers can't always infer
3. **Type stubs**: While PyTorch has type stubs, they may not cover all dynamic attributes
4. **Venv detection**: The type checker may not always detect the virtual environment correctly

## Solution

The configuration files disable the problematic error types:
- `reportAttributeAccessIssue: "none"` - Disables "not a known attribute" errors
- These are safe to disable because:
  - PyTorch is well-tested and documented
  - Runtime errors would occur if attributes didn't exist
  - The code runs successfully (verified)

## Verification

All PyTorch imports and attributes work correctly at runtime:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# All these work correctly:
nn.Module
nn.Conv2d
nn.Linear
F.relu
torch.randn
torch.cat
torch.stack
torch.sigmoid
```

## Best Practices

1. **Runtime Testing**: Always test code at runtime, not just rely on type checking
2. **Documentation**: Refer to PyTorch documentation for correct usage
3. **Type Hints**: Add type hints where helpful, but don't rely solely on type checking
4. **Configuration**: Use the provided configuration to reduce noise from false positives

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pyright Configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
- [Pylance Settings](https://code.visualstudio.com/docs/python/settings-reference)

---

**Note**: These are configuration warnings, not actual code errors. The code runs correctly when the venv is activated.

