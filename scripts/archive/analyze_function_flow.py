#!/usr/bin/env python3
"""Analyze Function Flow: MaxSightCNN Forward Pass

Traces the complete execution path from input to output,
identifying all function calls, data transformations, and decision points."""

import torch
import sys
from pathlib import Path
import inspect
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, CapabilityTier, TierConfig


class FunctionFlowTracer:
    """Traces function calls and data flow through the model."""
    
    def __init__(self):
        self.call_stack = []
        self.data_shapes = {}
        self.decision_points = []
        self.stage_transitions = []
    
    def trace_forward(self, model, images: torch.Tensor):
        """Trace the complete forward pass."""
        print("="*80)
        print("FUNCTION FLOW ANALYSIS: MaxSightCNN Forward Pass")
        print("="*80)
        
        batch_size = images.shape[0]
        print(f"\nInput: images.shape = {images.shape}")
        print(f"Batch size: {batch_size}")
        
        # Trace Stage A.
        print("\n" + "="*80)
        print("STAGE A: Fast Safety Pass (<150ms target)")
        print("="*80)
        
        # Stage A backbone.
        print("\n1. _forward_stage_a_backbone()")
        print("   - Input: images [B, 3, 224, 224]")
        print("   - Backbone: ResNet50 + FPN (ALWAYS, regardless of tier)")
        print("   - Output: fpn_features (p2, p3, p4, p5), fused_features, scene_context")
        
        # Stage A heads.
        print("\n2. Stage A Heads (Safety-Critical):")
        print("   - Objectness head")
        print("   - Classification head")
        print("   - Box regression head")
        print("   - Distance zone head")
        print("   - Urgency head")
        print("   - Uncertainty head")
        
        # Decision point.
        print("\n3. Stage A → Stage B Decision:")
        print("   - Check: stage_a_latency_ms > 200ms? → skip_stage_b = True")
        print("   - Check: uncertainty > 0.7? → skip_stage_b = True")
        print("   - Check: scene_graph_invalid? → skip_stage_b = True")
        print("   - If skip_stage_b: Return Stage A outputs only")
        
        # Trace Stage B (if not skipped)
        print("\n" + "="*80)
        print("STAGE B: Context Pass (Opportunistic, Tier-Dependent)")
        print("="*80)
        
        print("\n4. _forward_stage_b_backbone()")
        print("   - Input: images [B, 3, 224, 224] (raw input, not Stage A features)")
        print("   - Backbone: Hybrid CNN-ViT (if tier >= T2)")
        print("   - Temporal: ConvLSTM/TimeSformer (if tier >= T5)")
        print("   - Output: stage_b_features")
        
        print("\n5. Stage B Heads (Context-Rich):")
        print("   - Motion head (T2+)")
        print("   - Therapy state head (fatigue, depth, contrast)")
        print("   - Scene graph encoder (spatial/semantic relations)")
        print("   - OCR head (T3+)")
        print("   - Scene description head (T3+)")
        print("   - Sound event head (T4+)")
        print("   - Personalization head (T6+)")
        print("   - Predictive alert head (T2+)")
        
        print("\n6. Retrieval System (T4+, Async, Non-Blocking):")
        print("   - Global encoder (CLIP/DINOv2)")
        print("   - Region extractor")
        print("   - Patch extractor")
        print("   - Depth extractor")
        print("   - OCR encoder")
        print("   - Audio encoder")
        print("   - Scene graph encoder")
        
        print("\n" + "="*80)
        print("OUTPUT ASSEMBLY")
        print("="*80)
        
        print("\n7. Output Dictionary:")
        print("   - Stage A outputs: objectness, classifications, boxes, distance_zones, urgency_scores, uncertainty")
        print("   - Stage B outputs: motion, therapy_state, scene_graph, ocr, scene_description, sound_events")
        print("   - Metadata: stage_a_completed, stage_b_completed, skip_stage_b_reason, stage_a_latency_ms")
        
        return self
    
    def analyze_data_flow(self, model, images: torch.Tensor):
        """Analyze data transformations through the model."""
        print("\n" + "="*80)
        print("DATA FLOW ANALYSIS")
        print("="*80)
        
        with torch.no_grad():
            outputs = model(images)
        
        print("\nInput Shapes:")
        print(f"  images: {images.shape}")
        
        print("\nStage A Output Shapes:")
        if 'objectness' in outputs:
            print(f"  objectness: {outputs['objectness'].shape}")
        if 'classifications' in outputs:
            print(f"  classifications: {outputs['classifications'].shape}")
        if 'boxes' in outputs:
            print(f"  boxes: {outputs['boxes'].shape}")
        if 'distance_zones' in outputs:
            print(f"  distance_zones: {outputs['distance_zones'].shape}")
        if 'urgency_scores' in outputs:
            print(f"  urgency_scores: {outputs['urgency_scores'].shape}")
        if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
            print(f"  uncertainty: {outputs['uncertainty'].shape}")
        
        print("\nStage B Output Shapes:")
        if 'motion' in outputs and outputs['motion'] is not None:
            if isinstance(outputs['motion'], torch.Tensor):
                print(f"  motion: {outputs['motion'].shape}")
            else:
                print(f"  motion: {type(outputs['motion'])}")
        if 'therapy_state' in outputs and outputs['therapy_state'] is not None:
            therapy = outputs['therapy_state']
            if isinstance(therapy, dict):
                for k, v in therapy.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  therapy_state.{k}: {v.shape}")
        if 'scene_graph' in outputs and outputs['scene_graph'] is not None:
            sg = outputs['scene_graph']
            if isinstance(sg, dict):
                if 'edge_index' in sg and sg['edge_index'] is not None:
                    print(f"  scene_graph.edge_index: {sg['edge_index'].shape}")
                if 'edge_attr' in sg and sg['edge_attr'] is not None:
                    print(f"  scene_graph.edge_attr: {sg['edge_attr'].shape}")
        
        print("\nMetadata:")
        print(f"  stage_a_completed: {outputs.get('stage_a_completed', 'N/A')}")
        print(f"  stage_b_completed: {outputs.get('stage_b_completed', 'N/A')}")
        print(f"  skip_stage_b_reason: {outputs.get('skip_stage_b_reason', 'N/A')}")
        if 'stage_a_latency_ms' in outputs:
            print(f"  stage_a_latency_ms: {outputs['stage_a_latency_ms']}")
        
        return outputs
    
    def analyze_decision_points(self, model, images: torch.Tensor):
        """Analyze all decision points in the forward pass."""
        print("\n" + "="*80)
        print("DECISION POINTS ANALYSIS")
        print("="*80)
        
        print("\n1. Tier-Based Component Selection:")
        tier = model.tier_config.tier if hasattr(model, 'tier_config') else None
        print(f"   Tier: {tier}")
        if tier:
            print(f"   - Hybrid backbone: {hasattr(model, 'hybrid_backbone') and model.hybrid_backbone is not None}")
            print(f"   - Temporal encoder: {hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None}")
            print(f"   - Audio encoder: {hasattr(model, 'audio_encoder') and model.audio_encoder is not None}")
            print(f"   - Scene graph: {hasattr(model, 'scene_graph_encoder') and model.scene_graph_encoder is not None}")
            print(f"   - Retrieval: {hasattr(model, 'enable_retrieval') and getattr(model, 'enable_retrieval', False)}")
        
        print("\n2. Stage A → Stage B Skip Conditions:")
        print("   - High latency: stage_a_latency_ms > 200ms")
        print("   - High uncertainty: max(uncertainty) > 0.7")
        print("   - Invalid scene graph: edge_index/edge_attr mismatch")
        
        print("\n3. Conditional Head Execution:")
        print("   - Audio heads: Only if audio_features provided AND use_audio=True")
        print("   - OCR head: Only if tier >= T3")
        print("   - Scene description: Only if training=True OR generate_description=True")
        print("   - Retrieval: Only if tier >= T4 AND use_retrieval=True")
        
        print("\n4. Temporal Processing:")
        print("   - Video input: images.dim() == 5 [B, T, 3, H, W]")
        print("   - Temporal encoder: Only if tier >= T5 AND use_temporal=True")
        print("   - Temporal features: Fused into Stage B backbone")
        
        return self


def main():
    """Run function flow analysis."""
    print("Creating model...")
    model = create_model(
        num_classes=91,
        tier_config=TierConfig.for_tier(CapabilityTier.T2_HYBRID_VIT)
    )
    model.eval()
    
    # Create test input.
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Run analysis.
    tracer = FunctionFlowTracer()
    tracer.trace_forward(model, images)
    outputs = tracer.analyze_data_flow(model, images)
    tracer.analyze_decision_points(model, images)
    
    print("\n" + "="*80)
    print("FUNCTION FLOW ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal outputs: {len(outputs)} keys")
    print(f"Stage A completed: {outputs.get('stage_a_completed', False)}")
    print(f"Stage B completed: {outputs.get('stage_b_completed', False)}")
    
    return 0


if __name__ == "__main__":
    exit(main())


