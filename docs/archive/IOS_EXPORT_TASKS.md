# MaxSight iOS Export & Integration Tasks

**Last Updated:** 2025-12-06  
**Status:** Planning Phase

---

## 🎯 Goal

Export complete MaxSight system as minimal iOS bundle for Xcode integration:
- Single `.pte` model file
- Minimal configs (2 JSON files)
- Single processing reference file
- Integration documentation

---

## ✅ Completed Tasks

### Export System Implementation ✅
- [x] Design minimal export structure (4-5 files)
- [x] Implement `export_ios_bundle()` function
- [x] Implement `_extract_processing_reference()` function
- [x] Add to `ml/training/export.py`
- [x] Test export with real model
- [x] Verify all files created correctly

### Phase 1: Export System Refinement ✅
- [x] Test `_extract_processing_reference()` with actual files
- [x] Verify all essential functions are extracted correctly (6/6 functions found)
- [x] Ensure extracted code is syntactically valid (662 lines, 24KB)
- [x] Review `model_config.json` - ensure all needed params (✅ complete)
- [x] Review `runtime_config.json` - verify all runtime toggles (✅ complete)
- [x] Add version info to configs (✅ added)
- [x] Add export timestamp to configs (✅ added)
- [x] Complete `README_XCODE.md` with full Swift examples (✅ complete)
- [x] Add error handling examples (✅ added)
- [x] Add troubleshooting section (✅ added)

---

## 📋 Remaining Tasks

### Phase 1: Export System Refinement

#### 1.1 Function Extraction
- [ ] Test `_extract_processing_reference()` with actual files
- [ ] Verify all essential functions are extracted correctly
- [ ] Ensure extracted code is syntactically valid
- [ ] Test that extracted functions match source behavior
- [ ] Handle edge cases (nested functions, decorators, etc.)

#### 1.2 Config Refinement
- [ ] Review `model_config.json` - ensure all needed params
- [ ] Review `runtime_config.json` - verify all runtime toggles
- [ ] Add version info to configs
- [ ] Add export timestamp to configs
- [ ] Validate configs can be loaded in Swift

#### 1.3 Documentation
- [ ] Complete `README_XCODE.md` with full Swift examples
- [ ] Add error handling examples
- [ ] Add performance optimization tips
- [ ] Document expected output tensor shapes
- [ ] Add troubleshooting section

### Phase 2: Testing & Validation

#### 2.1 Export Testing
- [ ] Test export with quantized model (INT8)
- [ ] Test export with FP32 model
- [ ] Verify PTE file loads in ExecuTorch runtime
- [ ] Validate config JSON files are valid
- [ ] Test processing_reference.py imports correctly

#### 2.2 Integration Testing
- [ ] Test PTE loading in Swift (mock/test environment)
- [ ] Verify config loading in Swift
- [ ] Test preprocessing functions ported to Swift
- [ ] Test postprocessing functions ported to Swift
- [ ] Validate end-to-end: image → model → outputs

### Phase 3: iOS Implementation

#### 3.1 Xcode Project Setup
- [ ] Create Xcode project structure
- [ ] Add `maxsight.pte` to Xcode project
- [ ] Add ExecuTorch framework dependency
- [ ] Set up bundle resources (configs)
- [ ] Configure build settings

#### 3.2 Swift Model Wrapper
- [ ] Create Swift class for model loading
- [ ] Implement PTE loading with ExecuTorch
- [ ] Implement input preprocessing (port from Python)
- [ ] Implement output postprocessing (port from Python)
- [ ] Add error handling and validation

#### 3.3 Preprocessing Port
- [ ] Port condition-specific transforms to Swift
- [ ] Port image normalization to Swift
- [ ] Port CLAHE to Swift (or use iOS equivalent)
- [ ] Test preprocessing matches Python output
- [ ] Optimize for iOS performance

#### 3.4 Postprocessing Port
- [ ] Port NMS to Swift
- [ ] Port IoU calculation to Swift
- [ ] Port detection filtering to Swift
- [ ] Port priority calculation to Swift
- [ ] Test postprocessing matches Python output

#### 3.5 Integration
- [ ] Integrate model with iOS camera
- [ ] Integrate with iOS Vision framework (OCR)
- [ ] Integrate with AVFoundation (TTS)
- [ ] Integrate with Core Haptics (haptic feedback)
- [ ] Test full pipeline on device

### Phase 4: Optimization & Polish

#### 4.1 Performance
- [ ] Profile model inference latency
- [ ] Optimize preprocessing pipeline
- [ ] Optimize postprocessing pipeline
- [ ] Memory usage optimization
- [ ] Battery usage optimization

#### 4.2 Error Handling
- [ ] Add comprehensive error handling
- [ ] Add fallback mechanisms
- [ ] Add logging/monitoring
- [ ] Add user-friendly error messages
- [ ] Test error scenarios

#### 4.3 Documentation
- [ ] Complete iOS integration guide
- [ ] Document Swift API
- [ ] Add code examples
- [ ] Add troubleshooting guide
- [ ] Create video/walkthrough

---

## 🔧 Technical Tasks

### Export System
1. **Function Extractor**
   - Improve regex/parsing for function extraction
   - Handle class methods vs standalone functions
   - Preserve docstrings and type hints
   - Handle imports and dependencies

2. **Config Generation**
   - Auto-detect model parameters
   - Generate runtime config from model state
   - Add validation for config values
   - Version configs for compatibility

3. **Reference File**
   - Ensure all extracted functions are complete
   - Add comments explaining Swift porting
   - Include input/output examples
   - Add performance notes

### iOS Implementation
1. **ExecuTorch Integration**
   - Set up ExecuTorch framework
   - Create model loader wrapper
   - Handle tensor conversions (Swift ↔ ExecuTorch)
   - Implement batch processing

2. **Preprocessing**
   - Port image transforms to Swift
   - Use iOS-native APIs where possible (Core Image, vImage)
   - Optimize for real-time performance
   - Test accuracy vs Python version

3. **Postprocessing**
   - Port NMS algorithm to Swift
   - Port IoU calculation to Swift
   - Optimize for performance
   - Test accuracy vs Python version

4. **Integration**
   - Camera integration (AVFoundation)
   - Vision framework for OCR
   - AVSpeechSynthesizer for TTS
   - Core Haptics for haptic feedback
   - UI integration

---

## 📊 Priority Order

### High Priority (Must Have)
1. ✅ Export system implementation
2. Function extraction testing
3. Config validation
4. Basic Swift model wrapper
5. Preprocessing port (core functions)

### Medium Priority (Should Have)
6. Postprocessing port
7. Full iOS integration
8. Performance optimization
9. Error handling
10. Documentation

### Low Priority (Nice to Have)
11. Advanced optimizations
12. Extended documentation
13. Video tutorials
14. Performance benchmarking tools

---

## 🎯 Success Criteria

### Export System
- [ ] Single command exports complete bundle
- [ ] Bundle contains exactly 4-5 files
- [ ] All files are valid and usable
- [ ] Processing reference is complete and accurate

### iOS Integration
- [ ] Model loads successfully in Xcode
- [ ] Inference runs on device
- [ ] Preprocessing matches Python output
- [ ] Postprocessing matches Python output
- [ ] Full pipeline works end-to-end
- [ ] Performance meets targets (<500ms latency)

---

## 📝 Notes

- **Minimal Approach**: Keep bundle small, focused, and essential
- **Reference File**: Single file with all processing logic (not multiple files)
- **Configs**: Minimal JSON files with only essential parameters
- **Documentation**: Clear, actionable Swift integration guide
- **Testing**: Validate at each step before moving forward

---

## 🚀 Next Steps

1. **Immediate**: Test export system with real model
2. **Short-term**: Refine function extraction
3. **Medium-term**: Begin Swift porting
4. **Long-term**: Full iOS app integration

---

**Estimated Timeline:**
- Export System: 1-2 days
- iOS Implementation: 2-3 weeks
- Testing & Polish: 1 week
- **Total: 3-4 weeks**

