"""
Model Capability Tiers
Defines capability tiers (T0-T5) with kill switches for controlled complexity.

This module implements:
1. Tiered architecture (baseline → advanced features)
2. Runtime kill switches for each tier
3. Automatic tier selection based on mode and performance
4. Patient mode maximum tier enforcement (T3)
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class CapabilityTier(Enum):
    """Model capability tiers with increasing complexity."""
    T0_BASELINE_CNN = 0  # Baseline ResNet50 + FPN
    T1_ATTENTION = 1  # + SE/CBAM attention
    T2_HYBRID_VIT = 2  # + Hybrid CNN-ViT
    T3_CROSS_TASK = 3  # + Cross-task attention
    T4_CROSS_MODAL = 4  # + Cross-modal attention
    T5_TEMPORAL = 5  # + Temporal modeling (video)


@dataclass
class TierConfig:
    """Configuration for a capability tier."""
    tier: CapabilityTier
    enabled: bool = True
    
    # Component flags
    use_se_attention: bool = False
    use_cbam_attention: bool = False
    use_hybrid_backbone: bool = False
    use_dynamic_conv: bool = False
    use_cross_task_attention: bool = False
    use_cross_modal_attention: bool = False
    use_temporal_modeling: bool = False
    
    # Performance constraints
    max_latency_ms: float = 100.0
    min_confidence: float = 0.3
    
    @classmethod
    def for_tier(cls, tier: CapabilityTier) -> 'TierConfig':
        """Create config for a specific tier."""
        if tier == CapabilityTier.T0_BASELINE_CNN:
            return cls(
                tier=tier,
                max_latency_ms=50.0,
                min_confidence=0.3
            )
        elif tier == CapabilityTier.T1_ATTENTION:
            return cls(
                tier=tier,
                use_se_attention=True,
                use_cbam_attention=True,
                max_latency_ms=70.0,
                min_confidence=0.35
            )
        elif tier == CapabilityTier.T2_HYBRID_VIT:
            return cls(
                tier=tier,
                use_se_attention=True,
                use_cbam_attention=True,
                use_hybrid_backbone=True,
                use_dynamic_conv=True,
                max_latency_ms=100.0,
                min_confidence=0.4
            )
        elif tier == CapabilityTier.T3_CROSS_TASK:
            return cls(
                tier=tier,
                use_se_attention=True,
                use_cbam_attention=True,
                use_hybrid_backbone=True,
                use_dynamic_conv=True,
                use_cross_task_attention=True,
                max_latency_ms=120.0,
                min_confidence=0.4
            )
        elif tier == CapabilityTier.T4_CROSS_MODAL:
            return cls(
                tier=tier,
                use_se_attention=True,
                use_cbam_attention=True,
                use_hybrid_backbone=True,
                use_dynamic_conv=True,
                use_cross_task_attention=True,
                use_cross_modal_attention=True,
                max_latency_ms=150.0,
                min_confidence=0.45
            )
        elif tier == CapabilityTier.T5_TEMPORAL:
            return cls(
                tier=tier,
                use_se_attention=True,
                use_cbam_attention=True,
                use_hybrid_backbone=True,
                use_dynamic_conv=True,
                use_cross_task_attention=True,
                use_cross_modal_attention=True,
                use_temporal_modeling=True,
                max_latency_ms=200.0,
                min_confidence=0.5
            )
        else:
            return cls(tier=tier)


class TierManager:
    """
    Manages capability tier selection and kill switches.
    
    Rules:
    - Patient mode: maximum T3
    - Clinician mode: maximum T4
    - Dev mode: maximum T5
    - Automatic downgrade on performance issues
    """
    
    # Mode-based tier limits
    MODE_TIER_LIMITS = {
        'patient': CapabilityTier.T3_CROSS_TASK,
        'clinician': CapabilityTier.T4_CROSS_MODAL,
        'dev': CapabilityTier.T5_TEMPORAL
    }
    
    def __init__(self, mode: str = 'patient', initial_tier: Optional[CapabilityTier] = None):
        """
        Initialize tier manager.
        
        Args:
            mode: Output mode (patient/clinician/dev)
            initial_tier: Initial tier (defaults to T0)
        """
        self.mode = mode.lower()
        self.max_tier = self.MODE_TIER_LIMITS.get(self.mode, CapabilityTier.T3_CROSS_TASK)
        
        if initial_tier is None:
            # Start at T0 for safety
            self.current_tier = CapabilityTier.T0_BASELINE_CNN
        else:
            # Clamp to max tier for mode
            self.current_tier = self._clamp_tier(initial_tier)
        
        self.tier_config = TierConfig.for_tier(self.current_tier)
        self.degradation_count = 0
    
    def _clamp_tier(self, tier: CapabilityTier) -> CapabilityTier:
        """Clamp tier to maximum allowed for mode."""
        if tier.value > self.max_tier.value:
            return self.max_tier
        return tier
    
    def can_upgrade(self) -> bool:
        """Check if tier can be upgraded."""
        return self.current_tier.value < self.max_tier.value
    
    def upgrade_tier(self) -> bool:
        """
        Upgrade to next tier if allowed.
        
        Returns:
            True if upgraded, False otherwise
        """
        if not self.can_upgrade():
            return False
        
        next_tier_value = self.current_tier.value + 1
        next_tier = CapabilityTier(next_tier_value)
        self.current_tier = next_tier
        self.tier_config = TierConfig.for_tier(self.current_tier)
        self.degradation_count = 0
        return True
    
    def degrade_tier(self) -> bool:
        """
        Degrade to previous tier.
        
        Returns:
            True if degraded, False if already at T0
        """
        if self.current_tier.value == 0:
            return False
        
        prev_tier_value = self.current_tier.value - 1
        prev_tier = CapabilityTier(prev_tier_value)
        self.current_tier = prev_tier
        self.tier_config = TierConfig.for_tier(self.current_tier)
        self.degradation_count += 1
        return True
    
    def get_config(self) -> TierConfig:
        """Get current tier configuration."""
        return self.tier_config
    
    def get_tier_name(self) -> str:
        """Get human-readable tier name."""
        return self.current_tier.name
    
    def reset_to_baseline(self):
        """Reset to T0 baseline."""
        self.current_tier = CapabilityTier.T0_BASELINE_CNN
        self.tier_config = TierConfig.for_tier(self.current_tier)
        self.degradation_count = 0

