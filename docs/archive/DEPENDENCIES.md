# MaxSight Dependencies Documentation

**Last Updated:** January 2026  
**Python Version:** 3.12+ (Homebrew - Native arm64)  
**Platform:** macOS (darwin 25.1.0+)

---

## 📦 Core Dependencies

### ML Framework (Required)

```txt
torch>=2.9.1              # Core PyTorch (with MPS support)
torchvision>=0.24.1      # Computer vision utilities
torchaudio>=2.9.1        # Audio processing
```

**Installation:**
```bash
pip install torch torchvision torchaudio
```

**Verification:**
```python
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
```

### Data Processing (Required)

```txt
numpy>=2.2.6             # Numerical computing
pandas>=2.3.3            # Data manipulation
pillow>=12.0.0            # Image processing
opencv-python>=4.8.0     # Computer vision
scipy>=1.11.0            # Scientific computing (Hungarian matching)
scikit-learn>=1.3.0      # Machine learning utilities (clustering)
```

**Installation:**
```bash
pip install numpy pandas pillow opencv-python scipy scikit-learn
```

### Model Optimization (Required)

```txt
torchao>=0.14.1          # Model quantization and optimization
```

**Installation:**
```bash
pip install torchao
```

### Mobile Export (Optional - For iOS Deployment)

```txt
# ExecuTorch - Missing, needs installation
# CoreML - Missing, needs installation
```

**Installation (When Available):**
```bash
# ExecuTorch
pip install executorch

# CoreML
pip install coremltools
```

**Status:** ⚠️ Code ready in `ml/training/export.py`, but dependencies not installed

### Testing (Required for Development)

```txt
pytest>=9.0.1            # Testing framework
```

**Installation:**
```bash
pip install pytest
```

**Run Tests:**
```bash
pytest tests/
```

### Development Tools (Optional)

```txt
matplotlib>=3.10.7        # Visualization
tqdm>=4.66.0             # Progress bars
```

**Installation:**
```bash
pip install matplotlib tqdm
```

### Web Simulator (Optional)

```txt
flask>=3.0.0             # Web framework
flask-cors>=4.0.0        # CORS support
```

**Installation:**
```bash
pip install flask flask-cors
```

**Run Simulator:**
```bash
python tools/simulation/web_simulator.py
```

---

## 🖥️ System Dependencies

### macOS Requirements

- **OS Version**: macOS 25.1.0+ (darwin)
- **Architecture**: Apple Silicon (M1+) required for MPS support
- **Xcode**: 16.1+ (for iOS development)
- **Command Line Tools**: Required

**Verify System:**
```bash
uname -a                    # Should show arm64
xcode-select --version      # Should show Xcode 16.1+
python3 --version           # Should show Python 3.12+
```

### Python Environment

- **Python Version**: 3.12+
- **Package Manager**: pip (comes with Python)
- **Virtual Environment**: Recommended

**Setup Virtual Environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📋 Complete Requirements File

### `requirements.txt` (Production)

```txt
# Core ML Framework
torch>=2.9.1
torchvision>=0.24.1
torchaudio>=2.9.1

# Data Processing
numpy>=2.2.6
pandas>=2.3.3
pillow>=12.0.0
opencv-python>=4.8.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Model Optimization
torchao>=0.14.1

# Testing
pytest>=9.0.1

# Development Tools
matplotlib>=3.10.7
tqdm>=4.66.0

# Web Simulator (Optional)
flask>=3.0.0
flask-cors>=4.0.0
```

### `requirements-dev.txt` (Development)

```txt
# Include production requirements
-r requirements.txt

# Development Tools
black>=23.0.0              # Code formatting
flake8>=6.0.0              # Linting
mypy>=1.0.0                # Type checking
pytest-cov>=4.0.0          # Coverage reporting
```

### `requirements-mobile.txt` (iOS Deployment)

```txt
# Include production requirements
-r requirements.txt

# Mobile Export (When Available)
executorch                 # ExecuTorch export
coremltools                # CoreML export
```

---

## 🔍 Dependency Verification

### Check All Dependencies

```python
# verify_dependencies.py
import sys

dependencies = {
    'torch': '2.9.1',
    'torchvision': '0.24.1',
    'torchaudio': '2.9.1',
    'numpy': '2.2.6',
    'pandas': '2.3.3',
    'pillow': '12.0.0',
    'opencv-python': '4.8.0',
    'scipy': '1.11.0',
    'scikit-learn': '1.3.0',
    'torchao': '0.14.1',
    'pytest': '9.0.1',
    'matplotlib': '3.10.7',
    'tqdm': '4.66.0',
    'flask': '3.0.0',
    'flask-cors': '4.0.0',
}

missing = []
for package, min_version in dependencies.items():
    try:
        mod = __import__(package.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f'✅ {package}: {version}')
    except ImportError:
        print(f'❌ {package}: MISSING')
        missing.append(package)

if missing:
    print(f'\n⚠️ Missing packages: {", ".join(missing)}')
    print('Install with: pip install ' + ' '.join(missing))
else:
    print('\n✅ All dependencies installed!')
```

**Run Verification:**
```bash
python verify_dependencies.py
```

---

## 🚨 Known Dependency Issues

### Missing Mobile Export Dependencies

**Issue**: ExecuTorch and CoreML dependencies not installed  
**Impact**: Cannot export models for iOS deployment  
**Status**: Code ready, dependencies missing  
**Fix**: Install when available (see Mobile Export section)

### MPS Support Requirements

**Issue**: MPS (Metal Performance Shaders) requires Apple Silicon  
**Impact**: Cannot use GPU acceleration on Intel Macs  
**Status**: By design - MPS only works on Apple Silicon  
**Fix**: Use CPU mode on Intel Macs (slower but functional)

### Python Version Compatibility

**Issue**: Python 3.12+ required  
**Impact**: Older Python versions may not work  
**Status**: Documented requirement  
**Fix**: Upgrade Python to 3.12+

---

## 📦 Installation Instructions

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd 2026-Prototype

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"

# 6. Run tests
pytest tests/
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 ml/ tests/

# Run type checker
mypy ml/

# Run tests with coverage
pytest tests/ --cov=ml --cov-report=html
```

### Mobile Export Setup (When Available)

```bash
# Install mobile export dependencies
pip install -r requirements-mobile.txt

# Verify ExecuTorch
python -c "import executorch; print('ExecuTorch installed')"

# Verify CoreML
python -c "import coremltools; print('CoreML installed')"
```

---

## 🔄 Dependency Updates

### Update Strategy

1. **Pin Major Versions**: Use `>=` for minor/patch updates
2. **Test After Updates**: Run full test suite after updates
3. **Document Breaking Changes**: Note any breaking changes
4. **Update Requirements**: Keep `requirements.txt` current

### Update Commands

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade torch

# Check for outdated packages
pip list --outdated
```

### Security Updates

```bash
# Check for security vulnerabilities
pip-audit --requirement requirements.txt

# Update vulnerable packages
pip install --upgrade <vulnerable-package>
```

---

## 📝 Dependency Notes

### Version Compatibility

- **PyTorch 2.9.1+**: Required for MPS support on Apple Silicon
- **Python 3.12+**: Required for latest PyTorch features
- **macOS 25.1.0+**: Required for MPS support

### Optional Dependencies

Some features are optional and can be disabled if dependencies are missing:

- **Audio Processing**: Requires `torchaudio` (optional)
- **Web Simulator**: Requires `flask` and `flask-cors` (optional)
- **Visualization**: Requires `matplotlib` (optional)
- **Mobile Export**: Requires `executorch` and `coremltools` (optional)

### Platform-Specific Notes

- **Apple Silicon**: Full MPS support, best performance
- **Intel Macs**: CPU-only mode, slower but functional
- **Linux**: Not tested, may require modifications
- **Windows**: Not supported (macOS/iOS focus)

---

**Last Updated**: January 2026  
**Maintainer**: Development Team

