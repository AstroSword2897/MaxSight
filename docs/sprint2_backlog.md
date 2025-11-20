# Sprint 2 Backlog: iOS Integration & Real-Time Inference

## Sprint Goal
Integrate MaxSight CNN into iOS app with real-time camera inference, text reading (OCR), and audio output.

## Duration
14 days (Days 8-21)

---

## Week 1: iOS Foundation & Model Integration (Days 8-11)

### Day 8: iOS Project Setup

**Morning Session (3 hours)**
- Task 2.1: Create iOS Xcode Project (90 min)
  - Create new SwiftUI project
  - Set minimum iOS version to 15.0
  - Configure project settings
  - Add MaxSight branding/assets
  - **Testing**: Project builds successfully
  - **Acceptance**: Xcode project created, builds without errors

**Afternoon Session (3 hours)**
- Task 2.2: CoreML Model Integration (2 hours)
  - Import exported .mlpackage model
  - Create `MaxSightModel` wrapper class
  - Test model loading and inference
  - **Testing**: Model loads, runs inference on dummy data
  - **Acceptance**: Model successfully loaded, inference works

---

### Day 9: Camera Integration

**Morning Session (3 hours)**
- Task 2.3: Camera Setup (2 hours)
  - Implement `AVCaptureSession` for camera access
  - Create camera preview view
  - Handle permissions (camera, microphone)
  - **Testing**: Camera preview displays, permissions work
  - **Acceptance**: Camera captures frames successfully

**Afternoon Session (3 hours)**
- Task 2.4: Real-Time Inference Pipeline (2 hours)
  - Connect camera frames to model inference
  - Implement frame processing queue
  - Add inference rate limiting (target: 2 FPS)
  - **Testing**: Inference runs on camera frames, <500ms latency
  - **Acceptance**: Real-time inference working, meets latency target

---

### Day 10: Object Detection Display

**Morning Session (3 hours)**
- Task 2.5: Detection Visualization (2 hours)
  - Draw bounding boxes on camera preview
  - Display class labels and confidence scores
  - Add visual feedback for detections
  - **Testing**: Bounding boxes render correctly
  - **Acceptance**: Detections visualized on screen

**Afternoon Session (3 hours)**
- Task 2.6: Detection Filtering & Prioritization (2 hours)
  - Implement NMS in Swift (or use model output)
  - Filter by confidence threshold
  - Prioritize detections by urgency
  - **Testing**: Only high-confidence detections shown
  - **Acceptance**: Detection filtering working

---

### Day 11: Audio Output Foundation

**Morning Session (3 hours)**
- Task 2.7: Text-to-Speech Integration (2 hours)
  - Integrate `AVSpeechSynthesizer`
  - Create `AudioOutputService` class
  - Implement basic object announcement
  - **Testing**: Objects announced via TTS
  - **Acceptance**: TTS working, announces detections

**Afternoon Session (3 hours)**
- Task 2.8: Audio Controls & Settings (2 hours)
  - Add speech rate controls
  - Add verbosity levels (low/medium/high)
  - Persist settings with UserDefaults
  - **Testing**: Settings persist, audio controls work
  - **Acceptance**: Audio controls functional

---

## Week 2: OCR & Advanced Features (Days 12-15)

### Day 12: OCR Integration

**Morning Session (3 hours)**
- Task 2.9: Vision Framework OCR (2 hours)
  - Integrate `VNRecognizeTextRequest`
  - Process text regions from CNN
  - Extract text content
  - **Testing**: OCR extracts text from images
  - **Acceptance**: OCR working, extracts text accurately

**Afternoon Session (3 hours)**
- Task 2.10: Text Reading UI (2 hours)
  - Add "Read Text" button
  - Display detected text regions
  - Show extracted text
  - **Testing**: Text reading UI functional
  - **Acceptance**: Users can read text from camera

---

### Day 13: Condition-Specific Modes

**Morning Session (3 hours)**
- Task 2.11: Condition Selection UI (2 hours)
  - Create condition selection screen
  - Map conditions to model modes
  - Persist condition selection
  - **Testing**: Condition selection works
  - **Acceptance**: Users can select their condition

**Afternoon Session (3 hours)**
- Task 2.12: Condition-Specific Adaptations (2 hours)
  - Implement condition-specific preprocessing
  - Adjust detection thresholds per condition
  - Test with different conditions
  - **Testing**: Condition modes work correctly
  - **Acceptance**: Adaptations applied based on condition

---

### Day 14: Performance & Polish

**Morning Session (3 hours)**
- Task 2.13: Performance Optimization (2 hours)
  - Optimize inference pipeline
  - Add frame skipping for performance
  - Monitor battery usage
  - **Testing**: Performance meets targets (<500ms, <12%/hour battery)
  - **Acceptance**: Performance optimized

**Afternoon Session (3 hours)**
- Task 2.14: UI/UX Polish (2 hours)
  - Improve visual design
  - Add accessibility labels
  - Test with VoiceOver
  - **Testing**: App accessible, UI polished
  - **Acceptance**: App ready for user testing

---

### Day 15: Sprint Review & Planning

**Full Day (6 hours)**
- Task 2.15: Sprint 2 Demo (3 hours)
  - Demonstrate real-time object detection
  - Show OCR text reading
  - Present condition-specific modes
  - **Testing**: All features working
  - **Acceptance**: Demo successful

- Task 2.16: Retrospective & Backlog Refinement (3 hours)
  - Review Sprint 2 achievements
  - Identify improvements
  - Plan Sprint 3 (Sound Detection, Navigation)
  - **Acceptance**: Sprint 3 backlog ready

---

## User Stories

### Must Have (Sprint 2)
1. **As a user**, I want to point my phone at objects and hear what they are
2. **As a user**, I want to read text from signs and labels
3. **As a user**, I want to select my vision condition for better detection
4. **As a user**, I want to adjust how much information I hear

### Should Have (Sprint 2)
5. **As a user**, I want to see bounding boxes on detected objects
6. **As a user**, I want to pause/resume audio announcements
7. **As a user**, I want the app to work in different lighting conditions

### Could Have (Sprint 3+)
8. **As a user**, I want to detect sounds (alarms, vehicles)
9. **As a user**, I want navigation assistance
10. **As a user**, I want to label custom objects

---

## Technical Dependencies

- **iOS 15.0+**: Required for Vision framework improvements
- **CoreML**: For model inference
- **AVFoundation**: For camera and TTS
- **Vision Framework**: For OCR
- **SwiftUI**: For UI (or UIKit if preferred)

---

## Performance Targets

- **Inference Latency**: <500ms per frame
- **Frame Rate**: 2 FPS (inference), 30 FPS (camera preview)
- **Battery Usage**: <12% per hour
- **Memory**: <250MB RAM
- **Model Size**: <50MB

---

## Risk Mitigation

1. **Model Export Issues**
   - Risk: CoreML conversion fails
   - Mitigation: Test export early, have JIT fallback

2. **Performance Issues**
   - Risk: Inference too slow
   - Mitigation: Optimize model, use frame skipping

3. **Battery Drain**
   - Risk: App drains battery quickly
   - Mitigation: Optimize inference rate, add power-saving mode

4. **OCR Accuracy**
   - Risk: OCR not accurate enough
   - Mitigation: Use Vision framework, add confidence filtering

---

## Definition of Done

- [ ] Model integrated into iOS app
- [ ] Real-time camera inference working
- [ ] Object detection displayed on screen
- [ ] TTS announces detected objects
- [ ] OCR reads text from camera
- [ ] Condition-specific modes functional
- [ ] Performance meets targets
- [ ] App tested with VoiceOver
- [ ] Demo completed successfully

