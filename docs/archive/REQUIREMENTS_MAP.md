# MaxSight Requirements Map
## Visual Awareness & Accessibility Goals → Code Implementation

**Purpose:** Map user-focused, practical goals to specific code modules and features.

---

## 1. Environmental Awareness Goals

### 1.1 Detect and Label Objects
**Goal:** Detect and label objects in the environment (doors, stairs, furniture, signs)

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Object detection (48 classes) | ✅ Implemented | `ml/models/maxsight_cnn.py` - `cls_head`, `box_head` |
| Environmental classes | ✅ Implemented | `ml/models/maxsight_cnn.py` - `COCO_CLASSES` (48 classes) |
| Object categorization | ✅ Implemented | `ml/models/maxsight_cnn.py` - Classification head |
| Personal object labeling | ✅ Implemented | `ml/therapy/session_manager.py` - Custom labels support |
| Object tracking over time | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion tracking exists, needs integration |

### 1.2 Distance and Direction Cues
**Goal:** Provide distance and direction cues ("Stairs 3 meters ahead, slightly left")

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Distance estimation (near/medium/far) | ✅ Implemented | `ml/models/maxsight_cnn.py` - `distance_head` |
| Directional cues | ⚠️ Partial | `ml/models/maxsight_cnn.py` - Bounding boxes provide position, needs explicit direction |
| Spatial localization | ✅ Implemented | `ml/models/maxsight_cnn.py` - Bounding boxes (x, y, w, h) |
| Relative height & size | ⚠️ Partial | Bounding box size provides relative size, height needs enhancement |

### 1.3 Hazard Detection
**Goal:** Identify hazards or obstacles early (moving objects, drop-offs, clutter)

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Urgency scoring | ✅ Implemented | `ml/models/maxsight_cnn.py` - `urgency_head` (4 levels) |
| Moving object detection | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion features exist |
| Obstacle prioritization | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head prioritizes hazards |
| Dynamic change detection | ⚠️ Partial | Temporal encoder tracks motion, needs integration with detection |

### 1.4 Gradually Reduced Assistance
**Goal:** Provide gradually reduced assistance to allow users to practice using their own vision

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Adjustable verbosity | ✅ Implemented | `ml/therapy/session_manager.py` - Verbosity levels |
| Skill tracking | ✅ Implemented | `ml/therapy/session_manager.py` - Performance tracking |
| Adaptive assistance | ⚠️ Partial | `ml/therapy/task_generator.py` - Task difficulty scaling exists, needs integration |

---

## 2. Visual Assistance & Training Goals

### 2.1 Highlight Key Objects
**Goal:** Highlight key objects or areas to train the user's attention

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Visual overlays | ✅ Implemented | `app/overlays/overlay_engine.py` - Overlay system |
| ROI prioritization | ✅ Implemented | `ml/models/heads/roi_priority_head.py` - Priority scoring |
| Attention mechanisms | ✅ Implemented | `ml/models/maxsight_cnn.py` - Multi-head attention |

### 2.2 Adjustable Assistance Levels
**Goal:** Offer adjustable assistance levels (detailed vs. minimal cues)

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Verbosity settings | ✅ Implemented | `ml/therapy/session_manager.py` - Brief/normal/detailed |
| Output frequency control | ✅ Implemented | `ml/utils/output_scheduler.py` - Frequency-based scheduling |
| Customizable alerts | ✅ Implemented | `ml/utils/output_scheduler.py` - Channel selection (audio/visual/haptic) |

### 2.3 Independent Recognition Training
**Goal:** Enable users to recognize objects independently over time

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Session tracking | ✅ Implemented | `ml/therapy/session_manager.py` - Session management |
| Performance metrics | ✅ Implemented | `ml/therapy/session_manager.py` - Skill curves |
| Progress tracking | ✅ Implemented | `ml/therapy/session_manager.py` - Task attempt logging |

### 2.4 Fine-Detail Detection
**Goal:** Support fine-detail detection (text, small objects, facial recognition)

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| OCR integration | ✅ Implemented | `ml/utils/ocr_integration.py` - Text detection and extraction |
| Text region detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - `text_head` |
| Small object detection | ✅ Implemented | `ml/training/metrics.py` - Per-size metrics (small/medium/large) |
| Multi-scale detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - FPN for multi-scale features |

### 2.5 User Progress Tracking
**Goal:** Track user progress in recognizing objects with less assistance

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Skill curves | ✅ Implemented | `ml/therapy/session_manager.py` - `_generate_skill_curve()` |
| Performance history | ✅ Implemented | `ml/therapy/session_manager.py` - Session history |
| Improvement metrics | ✅ Implemented | `ml/therapy/session_manager.py` - Performance summaries |

---

## 3. Text & Speech Interaction Goals

### 3.1 OCR and Text Reading
**Goal:** OCR to read signs, labels, and printed text aloud

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Text detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - `text_head` |
| OCR extraction | ✅ Implemented | `ml/utils/ocr_integration.py` - `extract_text_from_region()` |
| iOS Vision integration | ⚠️ Placeholder | `ml/utils/ocr_integration.py` - iOS Vision framework ready |
| Text-to-speech | ⚠️ Placeholder | `app/ui/voice_feedback.py` - TTS ready for iOS integration |

### 3.2 Speech-to-Text Captioning
**Goal:** Convert speech → text for live captioning

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Speech recognition | ⚠️ Placeholder | iOS Speech framework integration needed |
| Live captions | ⚠️ Placeholder | iOS app integration needed |
| Caption display | ⚠️ Placeholder | iOS UI component needed |

### 3.3 Conversation Summarization
**Goal:** Summarize conversations in actionable phrases ("John said to meet at 5 PM")

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Summarization logic | ⚠️ Partial | OpenAI API integration ready, needs iOS app integration |
| Action item extraction | ⚠️ Placeholder | Needs implementation |

### 3.4 Multimodal Text Output
**Goal:** Provide TTS for blind users, on-screen captions for deaf users

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| TTS integration | ⚠️ Placeholder | `app/ui/voice_feedback.py` - Ready for iOS AVSpeechSynthesizer |
| Caption overlay | ⚠️ Placeholder | iOS UI component needed |
| Multimodal selection | ✅ Implemented | `ml/utils/output_scheduler.py` - Channel selection |

### 3.5 Custom Labels
**Goal:** Allow custom labels for personal objects ("my fridge," "meds box")

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Personal labeling | ✅ Implemented | `ml/therapy/session_manager.py` - Custom label support |
| Label recognition | ⚠️ Partial | Feature matching needed for personal object recognition |

---

## 4. Audio & Sound Awareness Goals

### 4.1 Sound Detection
**Goal:** Detect alarms, sirens, vehicle sounds, and other urgent cues

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Audio fusion | ✅ Implemented | `ml/models/maxsight_cnn.py` - Audio processing branch |
| Sound classification | ✅ Implemented | `ml/models/maxsight_cnn.py` - 15 sound classes |
| MFCC extraction | ✅ Implemented | `ml/data/dataset.py` - Audio feature extraction |

### 4.2 Sound Prioritization
**Goal:** Distinguish important sounds from background noise and prioritize by urgency

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Sound categorization | ✅ Implemented | `ml/data/download_datasets.py` - 15 sound classes defined |
| Urgency mapping | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head applies to sounds |
| Background noise filtering | ⚠️ Partial | Audio processing exists, noise reduction needs enhancement |

### 4.3 Multimodal Alerts
**Goal:** Provide audio, visual, or haptic alerts depending on user preference

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Haptic feedback | ✅ Implemented | `app/ui/haptic_feedback.py` - Haptic patterns |
| Visual alerts | ✅ Implemented | `app/overlays/overlay_engine.py` - Visual overlays |
| Audio alerts | ⚠️ Placeholder | iOS TTS integration needed |
| Channel selection | ✅ Implemented | `ml/utils/output_scheduler.py` - Cross-modal scheduling |

---

## 5. Integration & Prioritization Goals

### 5.1 Priority-Based Alerts
**Goal:** Combine visual and audio data into priority-based alerts

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Cross-modal integration | ✅ Implemented | `ml/utils/output_scheduler.py` - `CrossModalScheduler` |
| Priority calculation | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head + ROI priority |
| Alert filtering | ✅ Implemented | `ml/utils/output_scheduler.py` - Priority threshold filtering |

### 5.2 AI Summarization
**Goal:** Use AI summarization for complex scenes

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Scene description | ✅ Implemented | `ml/models/maxsight_cnn.py` - Scene embedding (512-d) |
| Natural language generation | ⚠️ Partial | Scene embedding ready, NLG needs iOS app integration |
| OpenAI integration | ⚠️ Placeholder | API integration ready, needs iOS app implementation |

### 5.3 Clear, Concise Outputs
**Goal:** Ensure alerts are clear, concise, and non-overwhelming

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Output frequency control | ✅ Implemented | `ml/utils/output_scheduler.py` - Rate limiting |
| Verbosity adjustment | ✅ Implemented | `ml/therapy/session_manager.py` - Brief/normal/detailed |
| Uncertainty suppression | ✅ Implemented | `ml/utils/output_scheduler.py` - Uncertainty threshold |

### 5.4 Customization
**Goal:** Support customization of alerts and feedback modality

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Alert frequency settings | ✅ Implemented | `ml/utils/output_scheduler.py` - `OutputConfig` |
| Channel preferences | ✅ Implemented | `ml/utils/output_scheduler.py` - Channel selection |
| User preferences | ⚠️ Partial | Settings structure exists, needs iOS app integration |

---

## 6. Practical Usability & Safety Goals

### 6.1 Real-Time Performance
**Goal:** Run in real-time on mobile device without lag

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Model quantization | ✅ Implemented | `ml/training/quantization.py` - INT8 quantization |
| Latency benchmarking | ✅ Implemented | `ml/training/benchmark.py` - Inference latency tracking |
| Performance targets | ✅ Documented | README.md - <500ms latency target |
| Model size optimization | ✅ Implemented | `ml/training/quantization.py` - <50MB target |

### 6.2 Accuracy & Reliability
**Goal:** Minimize false positives or missed detections

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Detection metrics | ✅ Implemented | `ml/training/metrics.py` - mAP, precision, recall, F1 |
| Confidence thresholds | ✅ Implemented | `ml/models/maxsight_cnn.py` - `detection_threshold` |
| Per-class metrics | ✅ Implemented | `ml/training/metrics.py` - Per-class AP |
| Per-size metrics | ✅ Implemented | `ml/training/metrics.py` - Small/medium/large object metrics |

### 6.3 User Feedback
**Goal:** Allow user feedback to correct misidentified objects or sounds

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Feedback collection | ⚠️ Placeholder | Needs iOS app implementation |
| Label correction | ⚠️ Placeholder | Needs implementation |
| Model fine-tuning | ⚠️ Placeholder | Training loop supports, needs user feedback integration |

### 6.4 Privacy & Security
**Goal:** Maintain privacy and security, no sensitive data leaving device without consent

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| On-device processing | ✅ Implemented | Model runs on-device (ExecuTorch/CoreML) |
| Data encryption | ⚠️ Placeholder | iOS app security needed |
| Consent management | ⚠️ Placeholder | iOS app privacy settings needed |

### 6.5 Usage History & Settings
**Goal:** Save usage history and user settings for continuity

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Session history | ✅ Implemented | `ml/therapy/session_manager.py` - Session tracking |
| Settings persistence | ⚠️ Placeholder | iOS UserDefaults integration needed |
| Export functionality | ⚠️ Placeholder | Session export needs implementation |

---

## 7. Optional Vision Enhancement Goals

### 7.1 Vision Training Exercises
**Goal:** Provide exercises that strengthen recognition of shapes, edges, and spatial relationships

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Task generation | ✅ Implemented | `ml/therapy/task_generator.py` - Task types |
| Therapy tasks | ✅ Implemented | `ml/therapy/task_generator.py` - Contrast, motion, depth, gaze tasks |
| Exercise tracking | ✅ Implemented | `ml/therapy/session_manager.py` - Task attempt logging |

### 7.2 Gradual Independence
**Goal:** Gradually reduce dependency on app hints as user improves

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Difficulty scaling | ✅ Implemented | `ml/therapy/task_generator.py` - Task difficulty adjustment |
| Skill-based adaptation | ✅ Implemented | `ml/therapy/session_manager.py` - Performance-based adaptation |
| Progress tracking | ✅ Implemented | `ml/therapy/session_manager.py` - Skill curves |

### 7.3 Daily Training Support
**Goal:** Encourage daily use for incremental training without causing fatigue

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Fatigue detection | ✅ Implemented | `ml/models/heads/fatigue_head.py` - Fatigue scoring |
| Session management | ✅ Implemented | `ml/therapy/session_manager.py` - Session limits |
| Rest recommendations | ✅ Implemented | `ml/therapy/task_generator.py` - `FATIGUE_REST` task type |

---

## 8. Visual Awareness: Comprehensive Goals

### 8.1 Object Detection & Recognition
**Goal:** Make user aware of all relevant items in immediate environment

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Stationary object detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - 48 environmental classes |
| Dynamic object detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - People, vehicles, moving objects |
| Object categorization | ✅ Implemented | `ml/models/maxsight_cnn.py` - Classification head |
| Importance ranking | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head + ROI priority |
| Personal object tracking | ⚠️ Partial | Custom labels exist, tracking needs enhancement |
| Temporal object tracking | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion features exist |

### 8.2 Spatial Awareness & Localization
**Goal:** Help user understand where things are in 3D space relative to them

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Distance estimation | ✅ Implemented | `ml/models/maxsight_cnn.py` - `distance_head` (near/medium/far) |
| Directional cues | ⚠️ Partial | Bounding boxes provide position, explicit direction needs enhancement |
| Relative height & size | ⚠️ Partial | Bounding box size provides relative size, height needs enhancement |
| Movement pattern detection | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion tracking exists |
| Navigation assistance | ⚠️ Partial | Obstacle detection exists, path planning needs implementation |

### 8.3 Detail Awareness
**Goal:** Help users perceive fine visual features they might otherwise miss

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Contrast enhancement | ✅ Implemented | `ml/utils/preprocessing.py` - CLAHE, contrast enhancement |
| Text reading (OCR) | ✅ Implemented | `ml/utils/ocr_integration.py` - Text detection and extraction |
| Surface feature detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - Edge detection via FPN |
| Anomaly detection | ⚠️ Partial | Object detection exists, anomaly detection needs enhancement |
| Temporal change detection | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion tracking exists |

### 8.4 Scene Context & Semantic Understanding
**Goal:** Turn raw visual data into meaningful, interpretable information

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Scene descriptions | ✅ Implemented | `ml/models/maxsight_cnn.py` - Scene embedding (512-d) |
| Scene classification | ⚠️ Partial | Scene embedding exists, classification needs enhancement |
| Hazard flagging | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head flags hazards |
| Usable feature highlighting | ✅ Implemented | `ml/models/heads/roi_priority_head.py` - ROI utility scoring |
| Semantic grouping | ⚠️ Partial | Object detection exists, semantic grouping needs implementation |

### 8.5 Visual Attention & Prioritization
**Goal:** Focus user attention on what matters most without overwhelming them

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Priority-based filtering | ✅ Implemented | `ml/utils/output_scheduler.py` - Priority threshold filtering |
| Irrelevant item suppression | ✅ Implemented | `ml/utils/output_scheduler.py` - Frequency-based filtering |
| Customizable alert levels | ✅ Implemented | `ml/therapy/session_manager.py` - Verbosity settings |
| Multimodal cues | ✅ Implemented | `ml/utils/output_scheduler.py` - Cross-modal scheduling |

### 8.6 Visual Memory & Cognitive Mapping
**Goal:** Train user to build mental models of their environment

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Position tracking | ⚠️ Partial | Object detection exists, position memory needs implementation |
| Spatial memory | ⚠️ Placeholder | Needs implementation |
| Contextual reminders | ⚠️ Placeholder | Needs implementation |
| User labeling | ✅ Implemented | `ml/therapy/session_manager.py` - Custom labels |

### 8.7 Fine-Grained Visual Features
**Goal:** Support vision training by emphasizing details users struggle with

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Edge detection | ✅ Implemented | `ml/models/maxsight_cnn.py` - FPN edge features |
| Texture patterns | ⚠️ Partial | CNN features capture texture, explicit texture detection needs enhancement |
| Color differentiation | ✅ Implemented | `ml/models/maxsight_cnn.py` - Color head for color blindness |
| Motion tracking | ✅ Implemented | `ml/models/temporal/temporal_encoder.py` - Motion/flow head |
| Light adaptation | ✅ Implemented | `ml/utils/preprocessing.py` - Low-light enhancement |

### 8.8 Adaptive Visual Assistance
**Goal:** Make support graduated and customizable

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Adjustable detail level | ✅ Implemented | `ml/therapy/session_manager.py` - Verbosity levels |
| Gradual assistance reduction | ✅ Implemented | `ml/therapy/task_generator.py` - Difficulty scaling |
| Performance feedback | ✅ Implemented | `ml/therapy/session_manager.py` - Performance summaries |
| Visual performance metrics | ✅ Implemented | `ml/therapy/session_manager.py` - Skill curves |

### 8.9 Safety-Oriented Visual Awareness
**Goal:** Ensure users are aware of risks in their environment

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Hazard alerts | ✅ Implemented | `ml/models/maxsight_cnn.py` - Urgency head (4 levels) |
| Dynamic change monitoring | ⚠️ Partial | `ml/models/temporal/temporal_encoder.py` - Motion tracking exists |
| Path safety | ⚠️ Partial | Obstacle detection exists, path planning needs implementation |
| Emergency notifications | ⚠️ Partial | Sound detection exists, emergency integration needs enhancement |

### 8.10 Practical UX Integration
**Goal:** Deliver visual awareness in usable, non-overwhelming way

| Requirement | Implementation Status | Code Location |
|------------|----------------------|---------------|
| Concise outputs | ✅ Implemented | `ml/utils/output_scheduler.py` - Content generation |
| Multimodal cues | ✅ Implemented | `ml/utils/output_scheduler.py` - Cross-modal scheduling |
| Clutter reduction | ✅ Implemented | `ml/utils/output_scheduler.py` - Frequency-based filtering |
| User customization | ✅ Implemented | `ml/therapy/session_manager.py` - Settings support |

---

## Implementation Status Summary

### ✅ Fully Implemented (Core ML Infrastructure)
- Object detection (48 classes)
- Distance estimation
- Urgency scoring
- OCR integration
- Audio fusion
- Session management
- Task generation
- Output scheduling
- Model quantization
- Performance metrics

### ⚠️ Partially Implemented (Needs Enhancement)
- Temporal object tracking (motion features exist, needs integration)
- Directional cues (position exists, explicit direction needed)
- Height estimation (size exists, height needed)
- Background noise filtering
- Spatial memory
- Semantic grouping
- Path planning
- Anomaly detection

### 📅 Placeholder (Needs iOS App Integration)
- TTS (iOS AVSpeechSynthesizer)
- Speech-to-text (iOS Speech framework)
- Live captions (iOS UI)
- User feedback collection
- Settings persistence (iOS UserDefaults)
- Privacy/security settings
- Conversation summarization UI

---

## Next Steps

1. **Enhance Partial Implementations**: Integrate temporal tracking, add directional cues, implement spatial memory
2. **iOS App Development**: Integrate TTS, STT, captions, user feedback, settings
3. **Advanced Features**: Path planning, semantic grouping, anomaly detection
4. **Testing & Validation**: User testing with actual vision/hearing impaired users
5. **Performance Optimization**: Ensure <500ms latency, <50MB model size

---

**Last Updated:** Based on comprehensive visual awareness goals and problem statement requirements.

