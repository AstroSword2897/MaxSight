# OCR Integration Plan for MaxSight iOS App

## Overview
This document outlines the plan for integrating Optical Character Recognition (OCR) into the MaxSight iOS app, enabling text reading capabilities for users with vision impairments.

## Architecture

### 1. Text Detection (CNN-based)
**Status**: ✅ Implemented in `maxsight_cnn.py`

- Model includes `text_head` that outputs text probability scores per detection
- Text regions are identified during object detection
- Output: Bounding boxes with text confidence scores

**Usage**:
```python
outputs = model(image)
text_scores = outputs['text_regions']  # [B, N] text probabilities
detections = model.get_detections(outputs, confidence_threshold=0.5)
# Filter detections where text_scores > threshold
```

### 2. OCR Processing (iOS Vision Framework)
**Status**: ⚠️ To be implemented in iOS app

**iOS Vision Framework Integration**:
- Use `VNRecognizeTextRequest` from Vision framework
- Process detected text regions from CNN
- Extract text content and confidence scores

**Swift Implementation Plan**:
```swift
import Vision

func recognizeText(in image: UIImage, regions: [CGRect]) -> [String] {
    guard let cgImage = image.cgImage else { return [] }
    
    let request = VNRecognizeTextRequest { request, error in
        // Process results
    }
    
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])
    
    return extractedTexts
}
```

### 3. Text-to-Speech (TTS)
**Status**: ⚠️ To be implemented in iOS app

**iOS AVSpeechSynthesizer Integration**:
- Use `AVSpeechSynthesizer` for voice output
- Support multiple languages
- Adjustable speech rate and voice

**Swift Implementation Plan**:
```swift
import AVFoundation

func speakText(_ text: String) {
    let synthesizer = AVSpeechSynthesizer()
    let utterance = AVSpeechUtterance(string: text)
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    utterance.rate = 0.5  // Adjustable
    synthesizer.speak(utterance)
}
```

## Workflow

### End-to-End Pipeline

1. **Image Capture** (iOS Camera)
   - Capture frame from camera
   - Preprocess: resize to 224x224, normalize

2. **Text Detection** (MaxSight CNN)
   - Run model inference
   - Extract text regions from `text_head` output
   - Filter by confidence threshold (>0.5)
   - Get bounding boxes for text regions

3. **OCR Processing** (Vision Framework)
   - For each detected text region:
     - Crop region from original image
     - Run `VNRecognizeTextRequest`
     - Extract text content

4. **Text-to-Speech** (AVSpeechSynthesizer)
   - Combine all extracted text
   - Announce: "Text detected: [content]"
   - Or announce per-region: "Top left: [text]"

## Implementation Tasks

### Phase 1: Basic OCR Integration (Sprint 2, Week 1)
- [ ] Integrate Vision framework in iOS app
- [ ] Create `TextRecognitionService` class
- [ ] Connect CNN text detection to Vision OCR
- [ ] Test with sample images

### Phase 2: TTS Integration (Sprint 2, Week 1)
- [ ] Integrate AVSpeechSynthesizer
- [ ] Create `TextToSpeechService` class
- [ ] Add speech rate controls
- [ ] Test voice output

### Phase 3: UI Integration (Sprint 2, Week 2)
- [ ] Add "Read Text" button to UI
- [ ] Show detected text regions visually
- [ ] Display extracted text
- [ ] Add speech controls (play/pause/stop)

### Phase 4: Optimization (Sprint 2, Week 2)
- [ ] Batch OCR processing for multiple regions
- [ ] Cache OCR results
- [ ] Optimize for real-time performance

## Performance Targets

- **Text Detection Latency**: <100ms (CNN inference)
- **OCR Processing**: <200ms per region
- **Total Pipeline**: <500ms for single text region
- **Battery Impact**: <5% per hour for continuous use

## Accessibility Features

1. **Verbosity Control**
   - Low: Only announce detected text
   - Medium: Announce text + location
   - High: Announce text + location + confidence

2. **Language Support**
   - English (primary)
   - Spanish, French, German (future)

3. **Reading Modes**
   - Automatic: Read all detected text
   - Manual: User selects regions to read
   - Continuous: Read text as it appears in camera

## Testing Strategy

1. **Unit Tests**
   - Test text detection accuracy
   - Test OCR extraction accuracy
   - Test TTS output

2. **Integration Tests**
   - End-to-end pipeline test
   - Performance benchmarks
   - Battery usage tests

3. **User Testing**
   - Test with users with vision impairments
   - Gather feedback on verbosity and timing
   - Iterate based on feedback

## Dependencies

- **iOS Version**: iOS 15.0+ (for Vision framework improvements)
- **Vision Framework**: Built-in (no external dependencies)
- **AVFoundation**: Built-in (for TTS)
- **MaxSight CNN**: Requires text_head output

## Notes

- Vision framework OCR is highly accurate for printed text
- Handwritten text recognition may require additional training
- Consider using Core ML for on-device OCR if Vision framework is insufficient
- Text detection from CNN can be used to prioritize OCR regions (faster than full-image OCR)

