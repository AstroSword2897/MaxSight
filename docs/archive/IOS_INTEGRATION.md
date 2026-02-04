# iOS Integration Guide

## Overview

This guide covers integrating MaxSight models into iOS applications.

## Prerequisites

- macOS with Xcode
- iOS device or simulator
- CoreML Tools installed

## Model Export

### Export to CoreML

```bash
python -m ml.training.export \
  --checkpoint runs/experiment1/checkpoint_best.pth \
  --format coreml \
  --output ios_bundle/MaxSight.mlmodel
```

### Verify Export

```python
import coremltools as ct

model = ct.models.MLModel("ios_bundle/MaxSight.mlmodel")
print(model)
```

## iOS Integration

### Add Model to Xcode Project

1. Drag `MaxSight.mlmodel` into your Xcode project
2. Ensure "Copy items if needed" is checked
3. Add to target

### Swift Integration

```swift
import CoreML
import Vision

class MaxSightModel {
    private var model: MaxSight?
    
    init() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU
            self.model = try MaxSight(configuration: config)
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func predict(image: CVPixelBuffer) -> [String: Any]? {
        guard let model = model else { return nil }
        
        do {
            let input = MaxSightInput(image: image)
            let output = try model.prediction(input: input)
            
            // Process output
            return [
                "detections": output.detections,
                "urgency": output.urgency,
                "distance": output.distance
            ]
        } catch {
            print("Prediction error: \(error)")
            return nil
        }
    }
}
```

### Image Preprocessing

Ensure input images match training preprocessing:

```swift
func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
    // Resize to 224x224 (or model input size)
    let size = CGSize(width: 224, height: 224)
    UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
    image.draw(in: CGRect(origin: .zero, size: size))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    // Convert to CVPixelBuffer
    // (Use CVPixelBuffer creation utilities)
    return pixelBuffer
}
```

## Performance Optimization

### Neural Engine Usage

```swift
let config = MLModelConfiguration()
config.computeUnits = .neuralEngine  // Use Neural Engine only
```

### Batch Processing

Process multiple images in batch for efficiency.

### Memory Management

- Release model between uses if memory constrained
- Use autoreleasepool for batch processing

## On-Device Testing

### Latency Measurement

```swift
let startTime = CFAbsoluteTimeGetCurrent()
let output = try model.prediction(input: input)
let latency = CFAbsoluteTimeGetCurrent() - startTime
print("Inference latency: \(latency * 1000)ms")
```

### Memory Profiling

Use Xcode Instruments:
1. Product > Profile
2. Select "Allocations" template
3. Run app and capture memory usage

## Troubleshooting

### Model Loading Errors

- Verify model file is in bundle
- Check model input/output shapes match expectations
- Ensure iOS version supports CoreML version

### Performance Issues

- Use Neural Engine for best performance
- Optimize image preprocessing
- Consider quantized model (INT8)

### Accuracy Issues

- Verify preprocessing matches training
- Check input normalization (mean/std)
- Ensure RGB vs BGR color order matches

## Best Practices

1. **Test on real devices**: Simulator performance differs
2. **Measure latency**: Ensure <150ms for Stage A
3. **Handle errors gracefully**: Model may fail on edge cases
4. **Cache predictions**: Avoid redundant inference
5. **Monitor memory**: Watch for memory leaks

## Example App

See `ios_bundle/ExampleApp/` for complete example (if available).

