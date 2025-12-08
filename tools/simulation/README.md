# MaxSight Product Simulator

Complete web-based simulator that runs the entire MaxSight system on a local port, simulating the actual product experience.

## Features

- **Complete Integration**: Uses every component in the MaxSight system
- **Web Interface**: Accessible via browser at http://localhost:5001
- **Multiple Scenarios**: Test different use cases (navigation, text reading, therapy, etc.)
- **Visual Conditions**: Simulate different vision impairments
- **Real-time Processing**: Process images through the complete pipeline
- **Visual Feedback**: See overlays, detections, and results
- **Session Management**: Track therapy sessions
- **Statistics**: Monitor performance metrics

## Installation

```bash
# Install Flask and dependencies
pip install flask flask-cors

# The simulator will use all existing MaxSight components
```

## Running the Simulator

```bash
cd tools/simulation
python web_simulator.py
```

Then open your browser to: **http://localhost:5001**

## Usage

1. **Select Visual Condition**: Choose from 13 different vision conditions
2. **Select Scenario**: Pick a test scenario (general, navigation, text reading, etc.)
3. **Start Session** (optional): Begin a therapy session for tracking
4. **Upload Image**: Drag & drop or select an image file
5. **View Results**: See detections, OCR results, descriptions, and feedback

## Components Integrated

The simulator integrates ALL MaxSight components:

- ✅ `MaxSightCNN` - Core model
- ✅ `ImagePreprocessor` - Condition-specific preprocessing
- ✅ `OutputScheduler` - Output prioritization
- ✅ `OCRIntegration` - Text detection and reading
- ✅ `DescriptionGenerator` - Scene descriptions
- ✅ `SpatialMemory` - Object tracking
- ✅ `PathPlanner` - Navigation planning
- ✅ `SessionManager` - Therapy session tracking
- ✅ `TaskGenerator` - Therapy task generation
- ✅ `TherapyIntegration` - Complete therapy system
- ✅ `OverlayEngine` - Visual overlays
- ✅ `VoiceFeedback` - Audio announcements
- ✅ `HapticFeedback` - Haptic patterns

## API Endpoints

- `GET /` - Main simulator interface
- `POST /api/init` - Initialize simulator with settings
- `POST /api/process` - Process image through pipeline
- `GET /api/scenarios` - Get available scenarios
- `GET /api/conditions` - Get available conditions
- `GET /api/stats` - Get current statistics
- `POST /api/session/start` - Start therapy session
- `POST /api/session/stop` - Stop therapy session
- `GET /api/session/status` - Get session status

## Scenarios

1. **General Environment** - Standard object detection
2. **Navigation** - Path planning and obstacle avoidance
3. **Text Reading** - OCR and text-to-speech focus
4. **Vision Therapy** - Therapy session with tasks
5. **Safety Alerts** - Urgency detection and warnings
6. **Accessibility Features** - Condition-specific adaptations

## Visual Conditions

All 13 supported conditions:
- Normal Vision
- Myopia, Hyperopia, Astigmatism
- Cataracts, Glaucoma, AMD
- Diabetic Retinopathy
- Retinitis Pigmentosa
- Color Blindness
- CVI, Amblyopia, Strabismus

## What You'll See

For each processed image:
- **Visual Overlay**: Bounding boxes, urgency colors, text regions
- **Detections**: All detected objects with confidence scores
- **OCR Results**: Detected text from signs/labels
- **Scene Description**: Natural language description
- **Voice Announcements**: What would be spoken
- **Haptic Patterns**: Vibration patterns (visualized)
- **Path Info**: Navigation guidance (if navigation scenario)
- **Therapy Feedback**: Progress and tasks (if session active)
- **Performance Stats**: Latency, FPS, detection counts

This is a complete product simulation - everything the user would experience!

