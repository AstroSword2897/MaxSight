"""Forward Pass Path Analysis

Maps all possible forward pass paths through MaxSightCNN
to understand computational flow and dependencies."""

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
        doc = """# MaxSight 3.0 Forward Pass Paths..."""
        
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

