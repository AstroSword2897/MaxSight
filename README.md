# MaxSight - Removing Barriers for Vision & Hearing Disabilities

**Visual Focus First** | **60-Day Implementation Plan**

---

## 🎯 Project Overview

MaxSight is an accessibility application that helps users with vision and hearing disabilities navigate their environment through:

- **Environmental Structuring**: Label surroundings in ways the recipient can understand
- **Clear Multimodal Communication**: Visual, audio, and haptic feedback
- **Skill Development Across Senses**: Address different senses for information input
- **Routine Workflow**: Adapts tasks to usage patterns and needs

---

## 👁️ Vision Conditions Supported (10 Types)

1. **Refractive Errors** (myopia, hyperopia, astigmatism, presbyopia)
2. **Cataracts** (reduced acuity)
3. **Glaucoma** (peripheral vision loss)
4. **AMD** (central vision damage)
5. **Diabetic Retinopathy** (retinal damage, floaters)
6. **Retinitis Pigmentosa** (night blindness, tunnel vision)
7. **Color Blindness** (color confusion)
8. **CVI** (cortical visual impairment)
9. **Amblyopia** (lazy eye)
10. **Strabismus** (crossed eyes)

---

## 🔧 Core Functions

### Environmental Reading
- **Object Detection**: Identifies objects, obstacles, people, vehicles
- **OCR**: Reads text from signs, labels, documents
- **Scene Descriptions**: Natural language descriptions of surroundings
- **Distance Estimation**: Near/medium/far zones

### Sound Detection & Alerts
- **Environmental Sound Classification**: Alarms, sirens, vehicles, speech
- **Sound Prioritization**: Urgent vs. general sounds
- **Multimodal Alerts**: Visual, audio, haptic notifications

### Multimodal Communication
- **Text-to-Speech**: Reads environment descriptions aloud
- **Speech-to-Text**: Captions for deaf users
- **Haptic Feedback**: Directional vibration patterns
- **Visual Overlays**: Subtle highlighting (max 10% screen)

### Personal Mode
- **Custom Labels**: User-defined object names
- **Verbosity Adjustment**: Brief/normal/detailed descriptions
- **Routine Adaptation**: Learns user patterns

---

## 🏗️ Technical Stack

- **ML Framework**: Custom PyTorch CNN (ResNet50 + FPN)
- **Platform**: iOS (camera/microphone, ExecuTorch export)
- **Architecture**: Meta AI-style (pure PyTorch, GPU-friendly, differentiable)
- **Deployment**: Quantized models for mobile (<50MB, <500ms latency)

---

## 📁 Repository Structure

```
2026-Prototype/
├── ml/                          # Core ML code
│   ├── models/                  # Model architectures
│   │   ├── maxsight_cnn.py      # Main CNN (object detection, scene understanding)
│   │   ├── heads/               # Output heads (detection, urgency, distance)
│   │   ├── temporal/            # Temporal encoder (motion tracking)
│   │   └── eye_model/           # Eye/face micro-model
│   ├── training/                # Training infrastructure
│   │   ├── train_loop.py        # Production training loop
│   │   ├── losses.py            # Multi-task losses
│   │   ├── metrics.py           # Evaluation metrics (mAP, precision, recall)
│   │   ├── matching.py          # Hungarian matching for detection
│   │   ├── scene_metrics.py     # Scene-level metrics
│   │   ├── evaluation.py        # Evaluation reports
│   │   ├── benchmark.py         # Inference latency benchmarking
│   │   ├── quantization.py      # INT8 quantization
│   │   └── export.py            # Model export (CoreML, ExecuTorch, JIT, ONNX)
│   ├── data/                    # Dataset utilities
│   │   ├── dataset.py           # MaxSightDataset (COCO, audio, environmental)
│   │   ├── create_accessibility_dataset.py  # Therapy-focused dataset
│   │   ├── download_datasets.py # Dataset downloaders
│   │   └── generate_annotations.py  # Annotation generation
│   ├── therapy/                 # Therapy system
│   │   ├── task_generator.py    # Task generation logic
│   │   └── session_manager.py   # Session tracking
│   └── utils/                   # Utilities
│       ├── preprocessing.py     # Meta AI-style preprocessing
│       └── output_scheduler.py  # Output scheduling
├── app/                         # Application code
│   ├── overlays/                # Overlay engine
│   ├── session_manager/         # Session management
│   └── ui/                      # UI components
│       ├── voice_feedback.py    # Voice prompts
│       └── haptic_feedback.py   # Haptic feedback
├── ios/                         # iOS app (Sprint 2)
├── tools/                       # Development tools
│   ├── quantization/           # Quantization tools
│   └── simulation/             # Simulation harness
├── scripts/                     # Training scripts
├── tests/                       # Test suite
├── checkpoints/                 # Model checkpoints
└── datasets/                    # Training data
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PyTorch 2.5.0+ (with MPS support for Apple Silicon)
- macOS with Apple Silicon M1+ (for iOS development)
- Xcode 16.1+ (for iOS app)

### Installation

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

### Training

```bash
# Train model
python scripts/train_maxsight.py \
    --data-dir datasets/coco \
    --epochs 100 \
    --batch-size 32 \
    --device mps
```

### Export for iOS

```python
from ml.training import export_to_executorch
from ml.models.maxsight_cnn import create_model

model = create_model()
export_to_executorch(model, "model.pte", input_size=(1, 3, 224, 224))
```

---

## 📅 60-Day Implementation Timeline

### **Sprint 1: Custom CNN for Environmental Reading** (Days 1-14)
- ✅ CNN architecture (ResNet50 + FPN)
- ✅ Object detection (48 environmental classes)
- ✅ Scene understanding
- ✅ Multi-task training
- ✅ Model quantization & export

### **Sprint 2: iOS App - "Reads Environment"** (Days 8-21)
- 📅 Camera integration
- 📅 Real-time detection
- 📅 OCR integration
- 📅 Text-to-speech
- 📅 Haptic feedback
- 📅 Sound detection

### **Sprint 3: Advanced Features & Polish** (Days 22-35)
- 📅 Distance estimation
- 📅 Navigation assistance
- 📅 Condition-specific adaptations
- 📅 Personal labeling
- 📅 Performance optimization

### **Sprint 4: Deployment & Iteration** (Days 36-60)
- 📅 TestFlight beta
- 📅 User testing
- 📅 App Store submission
- 📅 Launch & support

---

## 🎯 Key Features by Function

### Environmental Reading
- **Object Detection**: 48 classes (doors, stairs, vehicles, people, obstacles)
- **OCR**: Text detection and reading via iOS Vision framework
- **Scene Descriptions**: Natural language ("Door 2 meters ahead, handle on left")
- **Distance Zones**: Near/medium/far estimation

### Sound Detection
- **15 Sound Categories**: Alarms, sirens, vehicles, footsteps, speech
- **Prioritization**: Urgent sounds interrupt low-priority
- **Directional Audio**: Left/right/front/back when available

### Multimodal Output
- **Audio**: Text-to-speech descriptions, sound alerts
- **Visual**: Bounding boxes, text highlighting, urgency colors
- **Haptic**: Directional vibration patterns (danger/warning/caution)

### Condition Adaptations
- **Glaucoma**: Peripheral obstacle priority
- **AMD**: Central magnification, edge priority
- **Color Blindness**: Explicit color announcements
- **Retinitis Pigmentosa**: Low-light enhancement
- **CVI**: Simplified descriptions, consistent format

---

## 📊 Performance Targets

- **Inference Latency**: <500ms (target: <400ms)
- **Model Size**: <50MB (quantized)
- **Battery Drain**: <12% per hour normal use
- **Detection Accuracy**: >85% in varied environments
- **OCR Accuracy**: >90% text recognition
- **Sound Classification**: >80% accuracy

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test model
python tests/test_model.py

# Test preprocessing
python -m ml.utils.preprocessing
```

---

## 📝 Usage Examples

### Training

```python
from ml.training import ProductionTrainLoop, MaxSightLoss
from ml.data import MaxSightDataset
from ml.models.maxsight_cnn import create_model

# Create model
model = create_model(num_classes=48)

# Load dataset
train_dataset = MaxSightDataset(
    data_dir="datasets/coco",
    annotation_file="datasets/coco/annotations/instances_train2017.json"
)

# Train
trainer = ProductionTrainLoop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=MaxSightLoss(num_classes=48),
    num_epochs=100,
    device="mps"
)
results = trainer.train()
```

### Export

```python
from ml.training import export_to_executorch, export_to_coreml

# Export for iOS
export_to_executorch(model, "model.pte")
export_to_coreml(model, "model.mlpackage")
```

---

## 🔗 Key Modules

| Module | Purpose |
|--------|---------|
| `ml.models.maxsight_cnn` | Main CNN architecture |
| `ml.training.train_loop` | Production training loop |
| `ml.training.export` | Model export (iOS-ready) |
| `ml.data.dataset` | Dataset loading |
| `ml.utils.preprocessing` | Image preprocessing |

---

## 📄 License

See [LICENSE](LICENSE) file.

---

**Status**: 🟢 Active Development  
**Timeline**: 60 days (Nov 15, 2025 - Jan 14, 2026)  
**Platform**: iOS (iOS 17+)  
**Tech Stack**: PyTorch, ExecuTorch, CoreML
