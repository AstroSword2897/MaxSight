# MaxSight Onboarding Guide

**Welcome to MaxSight!** This guide will help you get started with the codebase.

**Estimated Time**: 2-4 hours  
**Prerequisites**: Python 3.12+, macOS, basic ML knowledge

---

## 🎯 Quick Start (30 minutes)

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd 2026-Prototype

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### 2. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=ml --cov-report=html
```

### 3. Create Your First Model

```python
# test_model.py
from ml.models.maxsight_cnn import create_model
import torch

# Create model
model = create_model(num_classes=80)

# Test forward pass
images = torch.randn(1, 3, 224, 224)
outputs = model(images)

print("Model created successfully!")
print(f"Output keys: {list(outputs.keys())}")
```

**Run:**
```bash
python test_model.py
```

---

## 📚 Understanding the Architecture (1-2 hours)

### Core Concepts

#### 1. Two-Stage Inference Pipeline

MaxSight uses a **two-stage inference pipeline**:

- **Stage A (Safety Pass)**: <150ms, Tier 1 heads only
  - Objectness, Classification, Box Regression
  - Distance, Urgency, Uncertainty
  - Answers: "Is the user safe right now?"

- **Stage B (Context Pass)**: Opportunistic, Tier 2/3 heads
  - Motion, ROI Priority, Scene Complexity
  - Scene Description, Retrieval, Therapy, Fatigue
  - Provides rich context (can be skipped if latency high)

**Key File**: `ml/models/maxsight_cnn.py` (lines 1079-1535)

#### 2. Tiered Head Architecture

Heads are organized into **3 tiers** by safety criticality:

- **Tier 1 (Safety-Critical)**: Never disabled, highest priority
  - Objectness, Classification, Box Regression
  - Distance, Urgency, Uncertainty

- **Tier 2 (Navigation & Context)**: Can degrade gracefully
  - Motion, ROI Priority, Scene Complexity
  - Spatial Memory, Path Planning

- **Tier 3 (Enhancement)**: Optional, advisory only
  - Scene Description, Retrieval, Therapy, Fatigue

**Key File**: `ml/models/maxsight_cnn.py` (TierConfig, CapabilityTier)

#### 3. Condition-Specific Adaptations

MaxSight adapts to **10+ vision conditions**:

- Glaucoma (peripheral loss)
- AMD (central loss)
- Cataracts (blur)
- Color blindness
- Retinitis pigmentosa (night blindness)
- And more...

**Key File**: `ml/utils/preprocessing.py`

### Architecture Diagram

See `README.md` for complete architecture diagram (lines 92-148).

---

## 🔍 Exploring the Codebase (1 hour)

### Directory Structure

```
2026-Prototype/
├── ml/                          # Core ML code
│   ├── models/                  # Model architectures
│   │   ├── maxsight_cnn.py      # Main model (2378 lines)
│   │   ├── heads/               # 20 specialized heads
│   │   ├── backbone/            # Backbone architectures
│   │   └── ...
│   ├── training/                # Training infrastructure
│   ├── data/                     # Dataset utilities
│   ├── retrieval/               # Retrieval system
│   ├── therapy/                  # Therapy system
│   └── utils/                   # Utilities
├── app/                         # Application code
├── tools/                       # Development tools
├── scripts/                     # Training scripts
├── tests/                       # Test suite
└── docs/                        # Documentation
```

### Key Files to Read

1. **`README.md`** (1348 lines)
   - Start here! Complete overview
   - Architecture explanation
   - Design decisions

2. **`ml/models/maxsight_cnn.py`** (2378 lines)
   - Core model implementation
   - Two-stage inference
   - All heads

3. **`docs/SYSTEM_ARCHITECTURE.md`**
   - Detailed architecture
   - Component interactions

4. **`docs/CRITICAL_LIMITATIONS.md`**
   - Known issues
   - Limitations
   - What needs fixing

5. **`docs/MAINTENANCE_SURVIVAL_MAP.md`**
   - Maintenance guide
   - Health checks
   - Troubleshooting

### Code Reading Order

1. **Start**: `README.md` (overview)
2. **Then**: `ml/models/maxsight_cnn.py` (core model)
3. **Then**: `ml/models/heads/` (individual heads)
4. **Then**: `ml/training/train_loop.py` (training)
5. **Finally**: `tools/simulation/` (application)

---

## 🛠️ Common Tasks

### Training a Model

```bash
# Basic training
python scripts/train_maxsight.py \
    --data-dir datasets/coco \
    --epochs 100 \
    --batch-size 32 \
    --device mps

# With GradNorm (recommended)
python scripts/train_maxsight.py \
    --data-dir datasets/coco \
    --epochs 100 \
    --batch-size 32 \
    --device mps \
    --use-gradnorm
```

**Key File**: `scripts/train_maxsight.py`

### Running Inference

```python
from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
import torch
from PIL import Image

# Load model
model = create_model(num_classes=80)
model.eval()

# Preprocess image
preprocessor = ImagePreprocessor(condition_mode='normal')
image = Image.open('test_image.jpg')
tensor = preprocessor(image)  # [1, 3, 224, 224]

# Run inference
with torch.no_grad():
    outputs = model(tensor)

# Access outputs
detections = outputs['classifications']
boxes = outputs['boxes']
urgency = outputs['urgency_scores']
```

### Exporting for iOS

```python
from ml.training.export import export_to_executorch, export_to_coreml
from ml.models.maxsight_cnn import create_model

# Create model
model = create_model(num_classes=80)
model.eval()

# Export to ExecuTorch
export_to_executorch(
    model, 
    "model.pte", 
    input_size=(1, 3, 224, 224)
)

# Export to CoreML
export_to_coreml(
    model, 
    "model.mlpackage"
)
```

**Note**: Requires ExecuTorch and CoreML dependencies (see `docs/DEPENDENCIES.md`)

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_model.py

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=ml --cov-report=html

# Run stress tests
python scripts/run_stress_tests.py --checkpoint checkpoints/model.pt
```

### Using the Web Simulator

```bash
# Start simulator
python tools/simulation/web_simulator.py

# Access at http://localhost:5000
# Upload images, see detections in real-time
```

---

## 🐛 Debugging Tips

### Common Issues

#### 1. MPS Not Available

**Error**: `torch.backends.mps.is_available()` returns `False`

**Solutions**:
- Verify Apple Silicon (M1+)
- Check macOS version (25.1.0+)
- Verify PyTorch version (2.9.1+)
- Use CPU mode: `device='cpu'`

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'ml'`

**Solutions**:
- Ensure you're in the repository root
- Check Python path: `python -c "import sys; print(sys.path)"`
- Install dependencies: `pip install -r requirements.txt`

#### 3. Model Output Shape Errors

**Error**: Shape mismatch in model outputs

**Solutions**:
- Check input size: Must be `[B, 3, 224, 224]`
- Verify model was created correctly
- Check head configurations

#### 4. Training Issues

**Error**: Loss not decreasing or NaN values

**Solutions**:
- Check learning rate (may be too high)
- Verify data loading (check dataset)
- Enable GradNorm: `--use-gradnorm`
- Check for gradient clipping

### Debugging Tools

#### Enable Debug Logging

```python
import logging
from ml.utils.logging_config import setup_logging

# Setup debug logging
setup_logging(log_level="DEBUG", log_dir=Path("logs"))
logger = logging.getLogger(__name__)

# Use logger
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Check Model Outputs

```python
# Print all output keys
print(f"Output keys: {list(outputs.keys())}")

# Check output shapes
for key, value in outputs.items():
    if torch.is_tensor(value):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")
```

#### Profile Performance

```python
import torch

# Profile forward pass
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True
) as prof:
    outputs = model(images)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

---

## 📖 Learning Resources

### Internal Documentation

1. **`README.md`**: Complete overview
2. **`docs/SYSTEM_ARCHITECTURE.md`**: Detailed architecture
3. **`docs/CRITICAL_LIMITATIONS.md`**: Known issues
4. **`docs/MAINTENANCE_SURVIVAL_MAP.md`**: Maintenance guide
5. **`docs/HANDOFF_MIGRATION_PLAN.md`**: Migration guide

### External Resources

1. **PyTorch Documentation**: https://pytorch.org/docs/
2. **Multi-Task Learning**: Research papers on GradNorm, PCGrad
3. **Object Detection**: FCOS, FPN papers
4. **Accessibility**: WCAG guidelines, accessibility research

### Code Examples

- **Model Creation**: `tests/test_model.py`
- **Training**: `scripts/train_maxsight.py`
- **Inference**: `tools/simulation/inference_engine.py`
- **Preprocessing**: `ml/utils/preprocessing.py`

---

## ✅ Onboarding Checklist

### Day 1: Setup and Overview

- [ ] Clone repository
- [ ] Set up development environment
- [ ] Install dependencies
- [ ] Run tests (all passing)
- [ ] Read `README.md`
- [ ] Create first model
- [ ] Run inference on test image

### Day 2: Architecture Deep Dive

- [ ] Read `docs/SYSTEM_ARCHITECTURE.md`
- [ ] Understand two-stage inference
- [ ] Understand tiered head architecture
- [ ] Explore `ml/models/maxsight_cnn.py`
- [ ] Review head implementations
- [ ] Understand condition adaptations

### Day 3: Training and Development

- [ ] Review training script
- [ ] Understand loss functions
- [ ] Understand GradNorm (if integrated)
- [ ] Run training on small dataset
- [ ] Export model for iOS
- [ ] Use web simulator

### Day 4: Advanced Topics

- [ ] Understand retrieval system
- [ ] Understand therapy system
- [ ] Review error handling
- [ ] Review output scheduler
- [ ] Read `docs/CRITICAL_LIMITATIONS.md`
- [ ] Understand known issues

### Week 2: Contribution

- [ ] Fix a small bug
- [ ] Add a test
- [ ] Improve documentation
- [ ] Review a pull request
- [ ] Contribute to codebase

---

## 🎓 Key Concepts to Master

### 1. Multi-Task Learning

MaxSight uses **multi-task learning** with 20+ heads sharing features. This is more efficient than separate models but requires careful balancing.

**Key Concepts**:
- Shared backbone (ResNet50 + FPN)
- Task-specific heads
- Gradient balancing (GradNorm)
- Loss weighting

### 2. Safety-First Architecture

MaxSight prioritizes **safety** over features:

- Tier 1 heads never disabled
- Stage A always runs (<150ms)
- Stage B is opportunistic
- Uncertainty suppression (>0.7 threshold)

**Key Concepts**:
- Tiered criticality
- Fail-silent modes
- Graceful degradation

### 3. Condition Adaptations

MaxSight adapts to **user's vision condition**:

- Learnable FiLM adapters (not hard-coded)
- Condition-specific preprocessing
- Adaptive attention

**Key Concepts**:
- FiLM (Feature-wise Linear Modulation)
- Condition embeddings
- Adaptive processing

---

## 🚀 Next Steps

After completing onboarding:

1. **Pick a Component**: Choose a component to specialize in
2. **Read the Code**: Deep dive into that component
3. **Run Experiments**: Try modifications and see results
4. **Contribute**: Fix bugs, add features, improve docs
5. **Ask Questions**: Use team channels for help

---

## 📞 Getting Help

### Documentation

- **Architecture**: `docs/SYSTEM_ARCHITECTURE.md`
- **Limitations**: `docs/CRITICAL_LIMITATIONS.md`
- **Maintenance**: `docs/MAINTENANCE_SURVIVAL_MAP.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md` (to be created)

### Code References

- **Model**: `ml/models/maxsight_cnn.py`
- **Training**: `ml/training/train_loop.py`
- **Utils**: `ml/utils/`
- **Tests**: `tests/`

### Common Questions

**Q: How do I add a new head?**  
A: See `ml/models/heads/` for examples. Add head class, integrate into `maxsight_cnn.py`, add loss function.

**Q: How do I modify the architecture?**  
A: Start with `ml/models/maxsight_cnn.py`. Understand two-stage inference first.

**Q: How do I train on my own data?**  
A: See `ml/data/dataset.py`. Create dataset loader, update annotations format.

**Q: How do I deploy to iOS?**  
A: See `ml/training/export.py`. Install ExecuTorch/CoreML dependencies first.

---

**Welcome to MaxSight!** 🎉

**Last Updated**: January 2026  
**Maintainer**: Development Team

