# Planning & Tech Stack Decisions
## Items for Discussion

This document outlines planning and tech stack decisions that we need to make together. Code-related items are already complete.

---

## Tech Stack Decisions Needed

### 1. Model Export Strategy ⚠️ **NEEDS DECISION**

**Current Status**: ExecuTorch mentioned but not fully integrated

**Options**:
- **Option A**: ExecuTorch (as planned)
  - Pros: Official PyTorch mobile solution, good for iOS
  - Cons: Complex setup, requires source build
  - Status: Installation attempted but needs completion
  
- **Option B**: CoreML (Apple's native solution)
  - Pros: Native iOS integration, excellent performance
  - Cons: Less flexible than ExecuTorch
  - Status: Not yet implemented
  
- **Option C**: ONNX → CoreML
  - Pros: Cross-platform, good tooling
  - Cons: Additional conversion step
  - Status: Not yet implemented

**Recommendation**: Start with CoreML for MVP, add ExecuTorch later if needed

**Decision Needed**: Which export format for Sprint 2?

---

### 2. Dataset Strategy ⚠️ **NEEDS DECISION**

**Current Status**: Code prepared, but actual dataset download/processing needed

**Questions**:
- **COCO Dataset**: 
  - Full COCO (80 classes) or filtered subset (48 environmental classes)?
  - Download size: ~25GB for full dataset
  - Processing time: Several hours for full dataset
  
- **Custom Dataset**:
  - Do we need custom environmental images?
  - Synthetic data generation strategy?
  
- **AudioSet**:
  - Full AudioSet or filtered subset?
  - Download size: ~1TB for full dataset
  - Processing: Extract only relevant sound classes?

**Recommendation**: Start with COCO subset (48 classes) for MVP

**Decision Needed**: 
- Which datasets to download?
- How much data to process initially?

---

### 3. Training Infrastructure ⚠️ **NEEDS DECISION**

**Current Status**: Training code ready, but needs actual data

**Questions**:
- **Training Hardware**:
  - Local (M1 Mac) or cloud (AWS/GCP)?
  - Estimated training time: 10-20 hours for full COCO subset
  
- **Training Schedule**:
  - When to start training?
  - How many epochs?
  - Validation strategy?
  
- **Model Checkpointing**:
  - Where to store checkpoints?
  - How often to save?
  - Best model selection strategy?

**Recommendation**: Start with local training on M1, move to cloud if needed

**Decision Needed**: Training hardware and schedule

---

### 4. iOS Integration Strategy ⚠️ **NEEDS DECISION**

**Current Status**: iOS folder exists but empty

**Questions**:
- **iOS Framework**:
  - SwiftUI or UIKit?
  - Minimum iOS version?
  
- **Model Integration**:
  - Real-time inference or batch processing?
  - Camera integration strategy?
  - Audio capture strategy?
  
- **UI/UX**:
  - Voice output (TTS)?
  - Haptic feedback?
  - Accessibility features priority?

**Recommendation**: SwiftUI + CoreML for MVP

**Decision Needed**: iOS architecture and features for Sprint 2

---

### 5. Performance Targets ⚠️ **NEEDS DECISION**

**Current Status**: Targets defined but not validated

**Questions**:
- **Latency**:
  - Target: <500ms per inference
  - Acceptable for real-time use?
  - Need optimization?
  
- **Model Size**:
  - Current: ~138MB (FP32)
  - Target: <50MB (quantized)
  - Quantization strategy?
  
- **Battery Usage**:
  - Target: <12%/hour
  - Acceptable for daily use?
  - Need power optimization?

**Recommendation**: Validate targets with actual testing

**Decision Needed**: Are current targets acceptable or need adjustment?

---

### 6. Deployment Strategy ⚠️ **NEEDS DECISION**

**Current Status**: Not yet planned

**Questions**:
- **App Distribution**:
  - App Store or TestFlight?
  - Beta testing strategy?
  
- **Model Updates**:
  - OTA model updates?
  - Version management?
  
- **Analytics**:
  - Usage tracking?
  - Performance monitoring?
  - Privacy considerations?

**Recommendation**: Start with TestFlight for beta testing

**Decision Needed**: Deployment and update strategy

---

## Sprint 2 Planning Questions

### Code-Related (Ready to Implement)
- ✅ Model export code structure
- ✅ iOS app skeleton
- ✅ CoreML conversion script
- ✅ Camera integration code
- ✅ Audio capture code

### Planning Needed (Your Input)
1. **Feature Priority**: Which features first?
   - Real-time object detection?
   - Voice descriptions?
   - Haptic feedback?
   - Offline mode?

2. **User Testing**: 
   - When to start user testing?
   - Who are the test users?
   - What feedback do we need?

3. **Accessibility Compliance**:
   - WCAG guidelines?
   - iOS accessibility standards?
   - Specific requirements?

---

## Next Steps

### Immediate (Code - I can do)
1. ✅ Complete Sprint 1 code review
2. ✅ Verify all tests pass
3. ✅ Update documentation

### Planning (We decide together)
1. ⚠️ Choose model export format
2. ⚠️ Decide on dataset strategy
3. ⚠️ Plan training schedule
4. ⚠️ Define iOS architecture
5. ⚠️ Set performance targets
6. ⚠️ Plan deployment strategy

---

## Questions for You

1. **Model Export**: ExecuTorch, CoreML, or both?
2. **Dataset**: Full COCO or subset? Custom data needed?
3. **Training**: When to start? Local or cloud?
4. **iOS**: SwiftUI or UIKit? Feature priorities?
5. **Performance**: Are current targets acceptable?
6. **Deployment**: App Store timeline? Beta testing plan?

---

**Status**: All code for Sprint 1 is complete! ✅  
**Next**: Planning decisions needed for Sprint 2

