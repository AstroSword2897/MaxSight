# MaxSight Product Simulator - Quick Start

## 🚀 Quick Start

### Option 1: Using the startup script
```bash
cd tools/simulation
./start_simulator.sh
```

### Option 2: Manual start
```bash
# Install dependencies (if not already installed)
pip install flask flask-cors

# Run simulator
cd tools/simulation
python web_simulator.py
```

### Option 3: Direct Python
```bash
python tools/simulation/web_simulator.py
```

## 🌐 Access the Simulator

Once running, open your browser to:
**http://localhost:8002**

## 📋 What the Simulator Does

The simulator is a **complete product simulation** that:

1. **Runs on a local web server** (port 8002)
2. **Integrates ALL MaxSight components**:
   - Model inference
   - Preprocessing (condition-specific)
   - OCR text detection
   - Description generation
   - Spatial memory
   - Path planning
   - Output scheduling
   - Therapy system
   - Visual overlays
   - Voice feedback
   - Haptic feedback

3. **Shows what users actually see**:
   - Visual overlays with bounding boxes
   - Color-coded urgency levels
   - Text detection highlights
   - Scene descriptions
   - Voice announcements
   - Performance metrics

4. **Supports multiple scenarios**:
   - General environment reading
   - Navigation assistance
   - Text reading
   - Vision therapy
   - Safety alerts
   - Accessibility features

5. **Tests all visual conditions**:
   - 13 different vision impairments
   - Condition-specific adaptations
   - Real-time preprocessing

## 🎮 How to Use

1. **Select Visual Condition**: Choose from dropdown (glaucoma, cataracts, etc.)
2. **Select Scenario**: Pick a test scenario
3. **Start Session** (optional): Begin therapy session tracking
4. **Upload Image**: Drag & drop or click to select an image
5. **View Results**: See complete processing results

## 📊 What You'll See

For each processed image:
- ✅ **Visual Overlay**: Bounding boxes, urgency colors, text regions
- ✅ **Detections**: All objects with confidence scores
- ✅ **OCR Results**: Detected text
- ✅ **Scene Description**: Natural language description
- ✅ **Voice Announcements**: What would be spoken
- ✅ **Haptic Patterns**: Vibration patterns
- ✅ **Path Info**: Navigation guidance (if navigation scenario)
- ✅ **Therapy Feedback**: Progress and tasks (if session active)
- ✅ **Performance Stats**: Latency, FPS, detection counts

## 🔧 Components Integrated

Every single component is integrated:
- `MaxSightCNN` - Core model
- `ImagePreprocessor` - Condition preprocessing
- `OutputScheduler` - Output prioritization
- `OCRIntegration` - Text detection
- `DescriptionGenerator` - Scene descriptions
- `SpatialMemory` - Object tracking
- `PathPlanner` - Navigation
- `SessionManager` - Therapy sessions
- `TaskGenerator` - Therapy tasks
- `TherapyIntegration` - Complete therapy system
- `OverlayEngine` - Visual overlays
- `VoiceFeedback` - Audio announcements
- `HapticFeedback` - Haptic patterns

## 🎯 This is the Complete Product

The simulator shows **exactly** what users would experience:
- Real-time processing
- Visual feedback
- Audio announcements
- Haptic patterns
- Therapy tracking
- Performance metrics

**It's a complete product demo running locally!**

