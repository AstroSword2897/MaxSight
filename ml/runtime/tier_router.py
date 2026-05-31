"""Bronze/Silver/Gold compute tier routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.runtime.contracts import ComputeTier

TIER_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "tiers"


@dataclass(frozen=True)
class TierProfile:
    """Deployable compute tier configuration."""

    tier: ComputeTier
    model_tier: str
    max_latency_ms: float
    enable_temporal: bool
    enable_rag: bool
    enable_vit: bool

    @classmethod
    def from_yaml(cls, path: Path) -> TierProfile:
        """Load tier profile from YAML."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            tier=ComputeTier(raw["tier"]),
            model_tier=str(raw.get("model_tier", "T0")),
            max_latency_ms=float(raw.get("max_latency_ms", 200)),
            enable_temporal=bool(raw.get("enable_temporal", False)),
            enable_rag=bool(raw.get("enable_rag", False)),
            enable_vit=bool(raw.get("enable_vit", False)),
        )


class TierRouter:
    """Route requests to tier profiles based on device budget and policy."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self._config_dir = config_dir or TIER_CONFIG_DIR
        self._profiles: dict[ComputeTier, TierProfile] = {}
        for tier in ComputeTier:
            path = self._config_dir / f"{tier.value}.yaml"
            if path.exists():
                self._profiles[tier] = TierProfile.from_yaml(path)

    def resolve(
        self,
        requested: ComputeTier,
        *,
        latency_budget_ms: float | None = None,
        battery_low: bool = False,
    ) -> TierProfile:
        """Return tier profile, downgrading when budget is constrained."""
        if requested not in self._profiles:
            raise KeyError(f"tier config missing: {requested.value}")
        profile = self._profiles[requested]
        if battery_low and requested == ComputeTier.GOLD:
            return self._profiles[ComputeTier.SILVER]
        if latency_budget_ms is not None and latency_budget_ms < profile.max_latency_ms:
            if requested == ComputeTier.GOLD:
                return self._profiles.get(ComputeTier.SILVER, profile)
            if requested == ComputeTier.SILVER:
                return self._profiles.get(ComputeTier.BRONZE, profile)
        return profile

    def to_trace(self, profile: TierProfile) -> dict[str, Any]:
        """Serialize routing decision for API traces."""
        return {
            "tier": profile.tier.value,
            "model_tier": profile.model_tier,
            "max_latency_ms": profile.max_latency_ms,
            "enable_temporal": profile.enable_temporal,
            "enable_rag": profile.enable_rag,
            "enable_vit": profile.enable_vit,
        }
