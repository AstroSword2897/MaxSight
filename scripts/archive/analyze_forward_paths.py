"""
Forward Pass Path Analysis

Maps all possible forward pass paths through MaxSightCNN
to understand computational flow and dependencies.
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import MaxSightCNN, create_model, TierConfig, CapabilityTier


class ForwardPathMapper:
    """Maps all forward pass paths through the model."""
    
    def __init__(self):
        self.paths = []
        self.components = defaultdict(list)
        
    def analyze_model_structure(self, model: MaxSightCNN) -> Dict:
        """Analyze model structure to understand components."""
        structure = {
            'backbone': [],
            'fpn': [],
            'heads': [],
            'fusion': [],
            'temporal': [],
            'retrieval': [],
        }
        
        # Analyze backbone
        if hasattr(model, 'hybrid_backbone') and model.hybrid_backbone is not None:
            structure['backbone'].append('hybrid_cnn_vit')
        else:
            structure['backbone'].append('resnet50_fpn')
        
        # Analyze FPN
        if hasattr(model, 'fpn'):
            structure['fpn'].append('standard_fpn')
        if hasattr(model, 'fpn_attention'):
            structure['fpn'].append('attention_enhanced')
        
        # Analyze heads
        head_attrs = [
            'objectness_head', 'classification_head', 'box_head',
            'distance_head_module', 'urgency_head', 'uncertainty_head',
            'motion_head', 'roi_priority_head', 'contrast_head',
            'scene_description_head', 'sound_event_head', 'ocr_head',
            'personalization_head', 'predictive_alert_head', 'fatigue_head',
        ]
        
        for attr in head_attrs:
            if hasattr(model, attr):
                head = getattr(model, attr)
                if head is not None:
                    head_name = attr.replace('_head', '').replace('_module', '')
                    structure['heads'].append(head_name)
        
        # Analyze fusion
        if hasattr(model, 'audio_encoder') and model.audio_encoder is not None:
            structure['fusion'].append('audio_visual')
        if hasattr(model, 'spatial_sound') and model.spatial_sound is not None:
            structure['fusion'].append('spatial_sound_mapping')
        
        # Analyze temporal
        if hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
            structure['temporal'].append('temporal_encoder')
        
        # Analyze retrieval
        if hasattr(model, 'tier_config'):
            if model.tier_config.use_retrieval:
                structure['retrieval'].append('multi_vector_retrieval')
        
        return structure
    
    def map_forward_path(
        self,
        model: MaxSightCNN,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        use_temporal: bool = False
    ) -> Dict:
        """Map the forward pass path for given inputs."""
        path = {
            'input_shape': list(images.shape),
            'has_audio': audio_features is not None,
            'temporal_mode': use_temporal,
            'stages': [],
            'components_used': [],
        }
        
        # Trace through forward pass
        B = images.shape[0]
        
        # Stage 1: Backbone
        path['stages'].append({
            'name': 'backbone',
            'type': 'resnet50_fpn' if not hasattr(model, 'hybrid_backbone') or model.hybrid_backbone is None else 'hybrid_cnn_vit',
        })
        
        # Stage 2: Audio Fusion (if enabled)
        if audio_features is not None and model.use_audio:
            path['stages'].append({
                'name': 'audio_fusion',
                'type': 'enhanced_audio_encoder',
            })
        
        # Stage 3: Temporal Processing (if enabled)
        if use_temporal and hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
            path['stages'].append({
                'name': 'temporal_processing',
                'type': 'temporal_encoder',
            })
        
        # Stage 4: Detection Heads (Tier 1 - Always runs)
        tier1_heads = ['objectness', 'classification', 'box', 'distance', 'urgency', 'uncertainty']
        path['stages'].append({
            'name': 'stage_a_tier1',
            'heads': tier1_heads,
        })
        
        # Stage 5: Context Heads (Tier 2/3 - Conditional)
        if hasattr(model, 'tier_config'):
            tier2_heads = []
            tier3_heads = []
            
            if hasattr(model, 'motion_head') and model.motion_head is not None:
                tier2_heads.append('motion')
            if hasattr(model, 'roi_priority_head') and model.roi_priority_head is not None:
                tier2_heads.append('roi_priority')
            if hasattr(model, 'contrast_head') and model.contrast_head is not None:
                tier2_heads.append('contrast')
            
            if hasattr(model, 'scene_description_head') and model.scene_description_head is not None:
                tier3_heads.append('scene_description')
            if hasattr(model, 'ocr_head') and model.ocr_head is not None:
                tier3_heads.append('ocr')
            if hasattr(model, 'personalization_head') and model.personalization_head is not None:
                tier3_heads.append('personalization')
            
            if tier2_heads or tier3_heads:
                path['stages'].append({
                    'name': 'stage_b_tier2_3',
                    'tier2_heads': tier2_heads,
                    'tier3_heads': tier3_heads,
                })
        
        return path
    
    def generate_path_documentation(self, output_path: str = "forward_pass_paths.md"):
        """Generate documentation of all forward pass paths."""
        doc = """# MaxSight 3.0 Forward Pass Paths

## Overview

This document maps all possible forward pass paths through MaxSightCNN,
showing how different input configurations and tier settings affect the
computational flow.

## Forward Pass Stages

### Stage 1: Input Processing
- **Single Image**: [B, 3, 224, 224]
- **Temporal Sequence**: [B, T, 3, 224, 224] → flattened to [B*T, 3, 224, 224]
- **Audio Features**: [B, 128] (optional)

### Stage 2: Backbone Feature Extraction
- **ResNet50 + FPN**: Standard CNN backbone
  - Output: FPN features at 4 scales (P2, P3, P4, P5)
- **Hybrid CNN-ViT**: Advanced backbone (T2+)
  - Output: Fused CNN + ViT features

### Stage 3: Audio-Visual Fusion (if audio enabled)
- **Enhanced Audio Encoder**: Processes audio features
- **Spatial Sound Mapping**: Maps audio to visual attention
- **Multiplicative Gating**: Applies audio attention to visual features

### Stage 4: Temporal Processing (if temporal enabled)
- **ConvLSTM**: Motion tracking
- **TimeSformer**: Long-range temporal dependencies
- **Temporal Consistency**: Flicker detection

### Stage 5: Stage A - Safety Heads (Tier 1, Always Runs)
These heads MUST run for safety:
- Objectness Head
- Classification Head
- Box Regression Head
- Distance Head
- Urgency Head
- Uncertainty Head

**Target Latency**: <150ms

### Stage 6: Stage B - Context Heads (Tier 2/3, Conditional)
These heads run opportunistically:
- Motion Head (Tier 2)
- ROI Priority Head (Tier 2)
- Contrast Head (Tier 2)
- Scene Description Head (Tier 3)
- OCR Head (Tier 3)
- Personalization Head (Tier 3)
- Predictive Alert Head (Tier 3)
- Fatigue Head (Tier 3)

**Target Latency**: <500ms total (Stage A + B)

## Path Variations

### Path 1: Baseline (T0)
- Backbone: ResNet50 + FPN
- Stage A only
- No audio, no temporal

### Path 2: Enhanced (T1)
- Backbone: ResNet50 + FPN + Attention
- Stage A + Tier 2 heads
- Optional audio

### Path 3: Hybrid (T2)
- Backbone: Hybrid CNN-ViT
- Stage A + Tier 2 heads
- Audio + Temporal support

### Path 4: Advanced (T3)
- Backbone: Hybrid CNN-ViT
- Stage A + Tier 2 + Tier 3 heads
- Full audio-visual fusion
- Cross-modal attention

### Path 5: Retrieval (T4)
- All T3 features
- Multi-vector retrieval system
- Knowledge-augmented retrieval

### Path 6: Full System (T5)
- All features enabled
- Full temporal processing
- Complete retrieval system
- All heads active

## Input Scenarios

### Scenario 1: Single Image, No Audio
```
Input: [1, 3, 224, 224]
Path: Backbone → Stage A → (Stage B if tier allows)
```

### Scenario 2: Batch Processing
```
Input: [4, 3, 224, 224]
Path: Same as Scenario 1, but batched
```

### Scenario 3: Temporal Sequence
```
Input: [1, 8, 3, 224, 224]
Path: Backbone → Temporal → Stage A → Stage B
```

### Scenario 4: With Audio
```
Input: [1, 3, 224, 224] + [1, 128]
Path: Backbone → Audio Fusion → Stage A → Stage B
```

### Scenario 5: Full Multi-Modal
```
Input: [1, 8, 3, 224, 224] + [1, 128]
Path: Backbone → Audio Fusion → Temporal → Stage A → Stage B
```

## Computational Complexity

### Backbone
- ResNet50: ~4.1 GFLOPs per image
- FPN: ~0.5 GFLOPs
- Hybrid CNN-ViT: ~8-12 GFLOPs (depends on ViT size)

### Heads (per head)
- Detection heads: ~0.1-0.5 GFLOPs
- Context heads: ~0.2-1.0 GFLOPs
- Retrieval: ~1-5 GFLOPs (depends on index size)

### Total Complexity
- T0 Baseline: ~5 GFLOPs
- T5 Full System: ~20-30 GFLOPs

## Memory Requirements

### Per Image
- Backbone features: ~50-100 MB
- Head outputs: ~10-50 MB
- Temporal state: ~20-50 MB (if enabled)
- Retrieval cache: ~100-500 MB (if enabled)

### Total (T5 Full System)
- Model weights: ~200-300 MB
- Runtime memory: ~200-700 MB per batch

## Latency Targets

- **Stage A (Safety)**: <150ms (must meet)
- **Stage B (Context)**: <500ms total (opportunistic)
- **Retrieval**: <100ms (advisory only, doesn't block)

## Notes

- Stage A heads always run (safety critical)
- Stage B heads run if tier allows and time permits
- Retrieval is advisory and doesn't block inference
- Temporal processing adds ~50-100ms overhead
- Audio fusion adds ~10-20ms overhead
"""
        
        with open(output_path, 'w') as f:
            f.write(doc)
        
        print(f"Forward pass path documentation generated: {output_path}")


def main():
    """Generate forward pass path documentation."""
    mapper = ForwardPathMapper()
    mapper.generate_path_documentation()
    
    # Test a few scenarios
    print("\nTesting forward pass paths...")
    
    scenarios = [
        {'name': 'T0_Baseline', 'tier': CapabilityTier.T0_BASELINE_CNN, 'audio': False, 'temporal': False},
        {'name': 'T2_Hybrid', 'tier': CapabilityTier.T2_HYBRID_VIT, 'audio': True, 'temporal': False},
        {'name': 'T5_Temporal', 'tier': CapabilityTier.T5_TEMPORAL, 'audio': True, 'temporal': True},
    ]
    
    device = 'cpu'  # Use CPU for path mapping
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    
    for scenario in scenarios:
        print(f"\n  Testing: {scenario['name']}")
        tier_config = TierConfig.for_tier(scenario['tier'])
        model = create_model(
            num_classes=91,
            use_audio=scenario['audio'],
            tier_config=tier_config
        )
        model.eval()
        model = model.to(device)
        
        # Create inputs
        if scenario['temporal']:
            images = torch.randn(1, 8, 3, 224, 224, device=device)
        else:
            images = torch.randn(1, 3, 224, 224, device=device)
        
        audio_features = torch.randn(1, 128, device=device) if scenario['audio'] else None
        
        # Map path
        path = mapper.map_forward_path(model, images, audio_features, scenario['temporal'])
        structure = mapper.analyze_model_structure(model)
        
        print(f"    Stages: {len(path['stages'])}")
        print(f"    Components: {list(structure.keys())}")


if __name__ == "__main__":
    main()

