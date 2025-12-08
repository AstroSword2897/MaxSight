# MaxSight iOS Bundle

**Export Date:** 2025-12-07 22:23:56  
**Version:** 1.0.0

## Files

- `maxsight.pte` - ExecuTorch model (add to Xcode project)
- `model_config.json` - Model parameters and thresholds
- `runtime_config.json` - Runtime settings and toggles
- `processing_reference.py` - Reference implementation (port to Swift)

## Xcode Integration

### 1. Add Model to Project

1. Drag `maxsight.pte` into your Xcode project
2. Ensure it's added to your app target
3. Add to "Copy Bundle Resources" in Build Phases

### 2. Install ExecuTorch Framework

Add ExecuTorch to your project:

```swift
// Package.swift or Xcode Package Manager
dependencies: [
    .package(url: "https://github.com/pytorch/executorch", from: "0.4.0")
]
```

### 3. Load Model

```swift
import Executorch

class MaxSightModel {
    private var program: Program?
    private var method: Method?
    
    func load() throws {
        guard let modelPath = Bundle.main.path(forResource: "maxsight", ofType: "pte") else {
            throw MaxSightError.modelNotFound
        }
        
        program = try Program.load(fromPath: modelPath)
        method = program?.loadMethod("forward")
    }
    
    func predict(image: Tensor) throws -> [String: Tensor] {
        guard let method = method else {
            throw MaxSightError.modelNotLoaded
        }
        
        let outputs = try method.execute(inputs: [image])
        return processOutputs(outputs)
    }
}
```

### 4. Preprocess Input

Reference `processing_reference.py` for preprocessing logic. Port to Swift:

```swift
func preprocessImage(_ image: UIImage, condition: VisionCondition) -> Tensor {
    // 1. Resize to model input size (224x224)
    let resized = image.resized(to: CGSize(width: 224, height: 224))
    
    // 2. Apply condition-specific transform
    let transformed = applyConditionTransform(resized, condition: condition)
    
    // 3. Normalize to [0, 1] and convert to tensor
    let normalized = transformed.normalized()
    let tensor = Tensor.fromImage(normalized)
    
    // 4. Add batch dimension
    return tensor.unsqueeze(0)  // [1, 3, H, W]
}

func applyConditionTransform(_ image: UIImage, condition: VisionCondition) -> UIImage {
    switch condition {
    case .glaucoma:
        return applyGlaucomaVignette(image)  // See processing_reference.py
    case .amd:
        return applyAMDCentralDarkening(image)
    case .cataracts:
        return applyCataractContrast(image)
    default:
        return image
    }
}
```

### 5. Run Inference

```swift
let model = MaxSightModel()
try model.load()

let inputTensor = preprocessImage(cameraFrame, condition: .glaucoma)
let outputs = try model.predict(image: inputTensor)

// Outputs contain:
// - classifications: [B, 80] - class logits
// - boxes: [B, 100, 4] - bounding boxes (center format: x, y, w, h)
// - objectness: [B, 100] - object confidence scores
// - urgency_scores: [B, 100, 4] - urgency level scores
// - distance_zones: [B, 100, 3] - distance zone probabilities
```

### 6. Postprocess Outputs

Reference `processing_reference.py` for postprocessing:

```swift
func postprocessDetections(
    boxes: Tensor,
    scores: Tensor,
    classifications: Tensor,
    config: ModelConfig
) -> [Detection] {
    // 1. Filter by detection threshold
    let validIndices = scores > config.detectionThreshold
    
    // 2. Apply NMS (Non-Maximum Suppression)
    // See processing_reference.py _nms() function
    let nmsIndices = applyNMS(
        boxes: boxes[validIndices],
        scores: scores[validIndices],
        threshold: config.nmsThreshold
    )
    
    // 3. Convert to detections
    var detections: [Detection] = []
    for idx in nmsIndices {
        let box = boxes[idx]
        let score = scores[idx]
        let classId = classifications[idx].argmax()
        
        detections.append(Detection(
            box: box,
            score: score,
            classId: classId
        ))
    }
    
    return detections
}
```

### 7. Load Configs

```swift
struct ModelConfig: Codable {
    let inputSize: [Int]
    let numClasses: Int
    let detectionThreshold: Double
    let nmsThreshold: Double
}

func loadModelConfig() throws -> ModelConfig {
    guard let url = Bundle.main.url(forResource: "model_config", withExtension: "json"),
          let data = try? Data(contentsOf: url) else {
        throw MaxSightError.configNotFound
    }
    
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(ModelConfig.self, from: data)
}
```

## Model Information

- **Input Size:** (1, 3, 224, 224)
- **Parameters:** 32,978,627
- **Model Size:** 126.6 MB
- **Classes:** 80
- **Urgency Levels:** 4
- **Distance Zones:** 3
- **Quantization:** FP32

## Output Tensor Shapes

See `model_config.json` for exact shapes. Typical outputs:

- `classifications`: [1, 80] - Class logits
- `boxes`: [1, 100, 4] - Bounding boxes (center format)
- `objectness`: [1, 100] - Object confidence
- `urgency_scores`: [1, 100, 4] - Urgency level scores
- `distance_zones`: [1, 100, 3] - Distance zone probabilities

## Reference Implementation

See `processing_reference.py` for complete reference:

- **Preprocessing**: Condition-specific transforms (glaucoma, AMD, cataracts, etc.)
- **Postprocessing**: NMS, IoU calculation, detection filtering
- **Scheduling**: Priority calculation, intensity, frequency, channel selection
- **OCR**: Text region clustering and grouping

## Performance Targets

- **Latency**: <500ms per frame (target: <400ms)
- **Memory**: <50MB model size
- **Battery**: <12% per hour normal use

## Troubleshooting

### Model won't load
- Verify `maxsight.pte` is in bundle resources
- Check ExecuTorch framework is properly linked
- Ensure iOS deployment target is 15.0+

### Inference fails
- Verify input tensor shape matches `input_size` in config
- Check tensor dtype is Float32
- Ensure tensor is on CPU (ExecuTorch requirement)

### Outputs are wrong
- Verify preprocessing matches Python reference
- Check postprocessing (NMS, filtering) is correct
- Compare with `processing_reference.py` implementation
