"""Situation Understanding Layer: perception outputs → psychological context for the therapy engine."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class SituationContext:
    """Psychological / behavioral context derived from perception. Feeds the therapy decision engine."""
    crowd_density: float = 0.0
    noise_level: float = 0.0
    navigation_complexity: float = 0.0
    uncertainty: float = 0.0
    environment_stress_level: float = 0.0
    cognitive_load_estimate: float = 0.0
    task_difficulty: float = 0.0
    user_motion_state: str = "unknown"  # e.g. stationary, walking, turning
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crowd_density": self.crowd_density,
            "noise_level": self.noise_level,
            "navigation_complexity": self.navigation_complexity,
            "uncertainty": self.uncertainty,
            "environment_stress_level": self.environment_stress_level,
            "cognitive_load_estimate": self.cognitive_load_estimate,
            "task_difficulty": self.task_difficulty,
            "user_motion_state": self.user_motion_state,
        }


class SituationUnderstandingLayer:
    """
    Converts raw perception stack outputs into a single psychological context vector.
    No neural network: deterministic derived features so therapy stays interpretable and safe.
    """

    def __init__(self, stress_weights: Optional[Dict[str, float]] = None):
        self.stress_weights = stress_weights or {
            "crowd": 0.35,
            "noise": 0.25,
            "navigation": 0.25,
            "uncertainty": 0.15,
        }

    def compute(self, perception: Dict[str, Any]) -> SituationContext:
        """
        Build situation context from perception. Expects keys from model/output pipeline:
        detections or object counts, urgency, distance, motion, audio env, uncertainty, etc.
        """
        raw = dict(perception)

        crowd_density = self._derive_crowd_density(perception)
        noise_level = self._derive_noise_level(perception)
        navigation_complexity = self._derive_navigation_complexity(perception)
        uncertainty = self._extract_float(perception, "uncertainty", "uncertainty_score", default=0.0)
        if hasattr(uncertainty, "item"):
            uncertainty = float(uncertainty.item())

        environment_stress_level = (
            self.stress_weights["crowd"] * crowd_density
            + self.stress_weights["noise"] * noise_level
            + self.stress_weights["navigation"] * navigation_complexity
            + self.stress_weights["uncertainty"] * uncertainty
        )
        environment_stress_level = min(1.0, max(0.0, environment_stress_level))

        cognitive_load_estimate = (navigation_complexity * 0.5 + crowd_density * 0.3 + uncertainty * 0.2)
        cognitive_load_estimate = min(1.0, max(0.0, cognitive_load_estimate))

        task_difficulty = (navigation_complexity + uncertainty) / 2.0
        task_difficulty = min(1.0, max(0.0, task_difficulty))

        user_motion_state = self._derive_motion_state(perception)

        return SituationContext(
            crowd_density=crowd_density,
            noise_level=noise_level,
            navigation_complexity=navigation_complexity,
            uncertainty=uncertainty,
            environment_stress_level=environment_stress_level,
            cognitive_load_estimate=cognitive_load_estimate,
            task_difficulty=task_difficulty,
            user_motion_state=user_motion_state,
            raw=raw,
        )

    def _extract_float(self, d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            if hasattr(v, "item"):
                return float(v.item())
            if isinstance(v, (int, float)):
                return float(v)
        return default

    def _derive_crowd_density(self, perception: Dict[str, Any]) -> float:
        detections = perception.get("detections") or perception.get("objects") or []
        if isinstance(detections, list):
            n = len(detections)
        else:
            n = 0
        person_count = sum(1 for d in (detections if isinstance(detections, list) else [])
                          if isinstance(d, dict) and d.get("class_name", "").lower() in ("person", "people"))
        if not isinstance(detections, list):
            person_count = 0
        density = min(1.0, (person_count * 0.15) + (n * 0.02))
        return min(1.0, max(0.0, density))

    def _derive_noise_level(self, perception: Dict[str, Any]) -> float:
        audio = perception.get("audio_environment") or perception.get("sound_level") or perception.get("noise_level")
        if audio is not None:
            if hasattr(audio, "item"):
                return min(1.0, max(0.0, float(audio.item())))
            if isinstance(audio, (int, float)):
                return min(1.0, max(0.0, float(audio)))
        return 0.0

    def _derive_navigation_complexity(self, perception: Dict[str, Any]) -> float:
        nav = perception.get("navigation_difficulty")
        if nav is not None:
            if hasattr(nav, "item"):
                return min(1.0, max(0.0, float(nav.item())))
            if isinstance(nav, (int, float)):
                return min(1.0, max(0.0, float(nav)))
        urgency = perception.get("urgency")
        if urgency is not None:
            if hasattr(urgency, "shape"):
                u = urgency
                if u.numel() > 0:
                    return min(1.0, float(u.max().item()) / 3.0)
            if isinstance(urgency, (int, float)):
                return min(1.0, float(urgency) / 3.0)
        return 0.0

    def _derive_motion_state(self, perception: Dict[str, Any]) -> str:
        motion = perception.get("motion_magnitude") or perception.get("motion_flow")
        if motion is None:
            return "unknown"
        if hasattr(motion, "mean") and hasattr(motion.mean(), "item"):
            m = float(motion.mean().item())
            if m < 0.1:
                return "stationary"
            if m < 0.4:
                return "walking"
            return "turning"
        return "unknown"
