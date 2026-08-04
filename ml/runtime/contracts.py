"""Runtime API and event contracts shared by orchestrator, simulator, and device bridges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from ml.runtime_constants import MVP_MODEL_OUTPUT_KEYS, filter_mvp_model_outputs

# Perception fields that are not model tensors but must survive MVP filtering.
PERCEPTION_METADATA_KEYS = frozenset(
    {
        "environment_stress_level",
        "cognitive_load_estimate",
        "uncertainty",
        "user_id",
        "frame_id",
        "disability_id",
        "preferred_channel",
    }
)


def _contract_required_fields(
    cls: type,
    optional: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Derive required API fields from dataclass definitions to prevent contract drift."""
    return frozenset(f.name for f in fields(cls) if f.name not in optional)


class ComputeTier(str, Enum):
    """Bronze/Silver/Gold compute routing tiers."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DegradedMode(str, Enum):
    """Runtime health states from docs/productization/04_runtime_boundary_spec.md."""

    D0_NORMAL = "D0"
    D1_HIGH_LOAD = "D1"
    D2_SAFETY_LOCK = "D2"
    D3_FAULT_CONTAINMENT = "D3"


@dataclass
class CriticalEvent:
    """Safety-critical output event contract."""

    event_type: str
    urgency: int
    direction: str
    distance_zone: str
    confidence: float
    uncertainty: float
    timestamp_source: float
    timestamp_emit: float
    distance_meters: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecondaryEvent:
    """Non-critical enrichment event; always preemptible."""

    event_type: str
    content: str
    preemptible: bool = True
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TherapyRecommendation:
    """Therapy output with score trace for SCRUM-19 unified response."""

    intervention_type: str
    channel: str
    content: str
    intensity: float
    score: float
    score_trace: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RagContext:
    """First-class RAG payload embedded in runtime responses."""

    guidance: str
    advisory_score: float
    retrieved_count: int
    grounded: bool
    guard_reason: str = ""
    failure_type: str = ""
    fallback_mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeRequest:
    """Unified inference request contract."""

    frame_id: str
    perception: dict[str, Any]
    user_id: str | None = None
    tier: ComputeTier = ComputeTier.SILVER
    enable_rag: bool = True
    enable_therapy: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tier"] = self.tier.value
        return data


@dataclass
class RuntimeResponse:
    """Unified inference response contract (SCRUM-19)."""

    frame_id: str
    tier: ComputeTier
    degraded_mode: DegradedMode
    classifications: list[dict[str, Any]] = field(default_factory=list)
    critical_events: list[CriticalEvent] = field(default_factory=list)
    secondary_events: list[SecondaryEvent] = field(default_factory=list)
    therapy: list[TherapyRecommendation] = field(default_factory=list)
    rag: RagContext | None = None
    latency_ms: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "tier": self.tier.value,
            "degraded_mode": self.degraded_mode.value,
            "classifications": self.classifications,
            "critical_events": [e.to_dict() for e in self.critical_events],
            "secondary_events": [e.to_dict() for e in self.secondary_events],
            "therapy": [t.to_dict() for t in self.therapy],
            "rag": self.rag.to_dict() if self.rag else None,
            "latency_ms": self.latency_ms,
            "trace": self.trace,
        }


REQUIRED_CRITICAL_EVENT_FIELDS = _contract_required_fields(
    CriticalEvent,
    optional=frozenset({"distance_meters"}),
)

REQUIRED_RUNTIME_RESPONSE_FIELDS = _contract_required_fields(
    RuntimeResponse,
    optional=frozenset({"rag", "trace"}),
)


def validate_critical_event(payload: dict[str, Any]) -> None:
    """Raise ValueError when a critical event violates the runtime contract."""
    missing = REQUIRED_CRITICAL_EVENT_FIELDS - set(payload.keys())
    if missing:
        raise ValueError(f"critical event missing fields: {sorted(missing)}")


def validate_runtime_response(payload: dict[str, Any]) -> None:
    """Raise ValueError when a runtime response violates the contract."""
    missing = REQUIRED_RUNTIME_RESPONSE_FIELDS - set(payload.keys())
    if missing:
        raise ValueError(f"runtime response missing fields: {sorted(missing)}")
    for event in payload.get("critical_events", []):
        validate_critical_event(event)


@dataclass(frozen=True)
class ModelOutputContract:
    """Single import surface for MVP model output keys at runtime."""

    allowed_keys: frozenset[str] = frozenset(MVP_MODEL_OUTPUT_KEYS)

    def filter(self, outputs: dict[str, Any], *, training: bool = False) -> dict[str, Any]:
        """Return outputs restricted to the MVP runtime surface."""
        return filter_mvp_model_outputs(outputs, training=training)

    def validate_keys(self, outputs: dict[str, Any]) -> None:
        """Raise ValueError when outputs contain keys outside the MVP surface."""
        extra = set(outputs.keys()) - self.allowed_keys
        if extra:
            raise ValueError(f"model outputs contain non-MVP keys: {sorted(extra)}")


def validate_model_outputs(outputs: dict[str, Any], *, training: bool = False) -> dict[str, Any]:
    """Filter model output keys while preserving perception metadata."""
    contract = ModelOutputContract()
    has_model_keys = any(key in contract.allowed_keys for key in outputs)
    if not has_model_keys:
        return outputs
    retained: dict[str, Any] = {}
    model_part: dict[str, Any] = {}
    for key, value in outputs.items():
        if key in contract.allowed_keys:
            model_part[key] = value
        elif key in PERCEPTION_METADATA_KEYS:
            retained[key] = value
    return {**retained, **contract.filter(model_part, training=training)}
