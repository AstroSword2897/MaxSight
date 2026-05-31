"""Build scene-based therapy tasks from perception outputs for MaxSight."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.data.ontology.loader import DisabilityOntology


class TherapyTaskType(Enum):
    """Therapy task types that use scene descriptions."""

    ATTENTION_TRAINING = "attention"  # Focus on specific objects in scene.
    CONTRAST_RECOGNITION = "contrast"  # Identify objects with different contrast.
    EDGE_DETECTION = "edge"  # Identify edges and boundaries.
    SPATIAL_AWARENESS = "spatial"  # Understand spatial relationships.
    WARNING_RECOGNITION = "warning"  # Learn to recognize hazard cues over time.


class TherapyTaskIntegrator:
    """Convert scene descriptions and detections into structured therapy tasks."""

    def __init__(self) -> None:
        """Initialize empty task history for integrator sessions."""
        self.task_history: list[dict] = []
        self._ontology: DisabilityOntology | None = (
            None  # Loaded lazily on first disability-routed call.
        )

    def _serialize_task(self, payload: dict) -> dict:
        """Convert enum values to JSON-safe values for API responses."""
        serialized: dict = {}
        for key, value in payload.items():
            if isinstance(value, Enum):
                serialized[key] = value.value
            elif isinstance(value, list):
                serialized[key] = [item.value if isinstance(item, Enum) else item for item in value]
            else:
                serialized[key] = value
        return serialized

    def create_attention_task(
        self, scene_description: str, target_objects: list[str], difficulty: float = 0.5
    ) -> dict:
        """Create attention training task from scene description."""
        return {
            "task_type": TherapyTaskType.ATTENTION_TRAINING,
            "scene_description": scene_description,
            "target_objects": target_objects,
            "difficulty": difficulty,
            "instructions": f"Focus on: {', '.join(target_objects)}",
            "duration": int(30 + (1.0 - difficulty) * 30),  # 30-60 seconds.
        }

    def create_contrast_task(
        self, scene_description: str, contrast_levels: list[float], difficulty: float = 0.5
    ) -> dict:
        """Create contrast recognition task from scene description."""
        return {
            "task_type": TherapyTaskType.CONTRAST_RECOGNITION,
            "scene_description": scene_description,
            "contrast_levels": contrast_levels,
            "difficulty": difficulty,
            "instructions": "Identify objects with different contrast levels",
            "duration": int(30 + (1.0 - difficulty) * 30),
        }

    def create_edge_task(
        self, scene_description: str, edge_types: list[str], difficulty: float = 0.5
    ) -> dict:
        """Create edge detection task from scene description."""
        return {
            "task_type": TherapyTaskType.EDGE_DETECTION,
            "scene_description": scene_description,
            "edge_types": edge_types,
            "difficulty": difficulty,
            "instructions": f"Identify edges: {', '.join(edge_types)}",
            "duration": int(30 + (1.0 - difficulty) * 30),
        }

    def create_spatial_task(
        self, scene_description: str, spatial_relationships: list[str], difficulty: float = 0.5
    ) -> dict:
        """Create spatial awareness task from scene description."""
        return {
            "task_type": TherapyTaskType.SPATIAL_AWARENESS,
            "scene_description": scene_description,
            "spatial_relationships": spatial_relationships,
            "difficulty": difficulty,
            "instructions": f"Identify relationships: {', '.join(spatial_relationships)}",
            "duration": int(30 + (1.0 - difficulty) * 30),
        }

    def create_warning_recognition_task(
        self,
        hazard_type: str,
        urgency_level: int,
        cue_description: str,
        difficulty: float = 0.5,
        scene_description: str | None = None,
    ) -> dict:
        """Create warning recognition task so the user learns to associate cues with hazards."""
        out: dict = {
            "task_type": TherapyTaskType.WARNING_RECOGNITION,
            "hazard_type": hazard_type,
            "urgency_level": urgency_level,
            "cue_description": cue_description,
            "difficulty": difficulty,
            "instructions": (
                f"Learn the cue for {hazard_type}. "
                f"You will hear/feel: {cue_description}. "
                "When you hear this in real use, it means this hazard is present."
            ),
            "duration": int(20 + (1.0 - difficulty) * 25),
        }
        if scene_description is not None:
            out["scene_description"] = scene_description
        return out

    def _task_type_for_disability(self, disability_id: str) -> TherapyTaskType | None:
        """Look up the primary task type for a disability via the ontology.

        Falls back to ``None`` so the caller can use its own default.
        """
        try:
            from ml.data.ontology.loader import DisabilityOntology

            if self._ontology is None:
                self._ontology = DisabilityOntology.load()
            focuses = self._ontology.therapy_focus_for(disability_id)
            if not focuses:
                return None
            focus_map: dict[str, TherapyTaskType] = {
                "contrast_micro": TherapyTaskType.CONTRAST_RECOGNITION,
                "roi_findability": TherapyTaskType.ATTENTION_TRAINING,
                "motion_tracking": TherapyTaskType.ATTENTION_TRAINING,
                "depth_shift": TherapyTaskType.SPATIAL_AWARENESS,
                "gaze_stabilization": TherapyTaskType.EDGE_DETECTION,
                "warning_recognition": TherapyTaskType.WARNING_RECOGNITION,
            }
            return focus_map.get(focuses[0])
        except Exception:
            return None

    def _build_attention_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        difficulty: float,
    ) -> dict:
        target_objects = [d.get("class_name", "object") for d in detections[:3]]
        return self.create_attention_task(scene_description, target_objects, difficulty)

    def _build_contrast_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        difficulty: float,
    ) -> dict:
        contrast_levels = [d.get("contrast", 0.5) for d in detections if "contrast" in d]
        if not contrast_levels:
            contrast_levels = [0.3, 0.5, 0.7]
        return self.create_contrast_task(scene_description, contrast_levels, difficulty)

    def _build_edge_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        difficulty: float,
    ) -> dict:
        edge_types: list[str] = []
        for detection in detections:
            class_name = str(detection.get("class_name", "")).lower()
            if "door" in class_name:
                edge_types.append("door_edge")
            if "stair" in class_name or "step" in class_name:
                edge_types.append("stair_edge")
            if class_name:
                edge_types.append(f"{class_name}_edge")
        if not edge_types:
            edge_types = ["obstacle_edge"]
        return self.create_edge_task(scene_description, edge_types, difficulty)

    def _build_spatial_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        difficulty: float,
    ) -> dict:
        relationships: list[str] = []
        for detection in detections[:8]:
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
            width = max(1.0, float(bbox[2]) - float(bbox[0]))
            if x_center < 0.33:
                relationships.append("left_of")
            elif x_center > 0.66:
                relationships.append("right_of")
            else:
                relationships.append("centered")
            relationships.append("near" if width > 0.25 else "far")
        if not relationships:
            relationships = ["near"]
        return self.create_spatial_task(scene_description, relationships, difficulty)

    def _build_warning_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        difficulty: float,
    ) -> dict:
        hazard = next(
            (d for d in detections if d.get("urgency", 0) >= 1),
            detections[0] if detections else {},
        )
        hazard_type = hazard.get("class_name", "obstacle")
        urgency_level = hazard.get("urgency", 1)
        return self.create_warning_recognition_task(
            hazard_type=hazard_type,
            urgency_level=urgency_level,
            cue_description=f"Alert for {hazard_type} (urgency {urgency_level})",
            difficulty=difficulty,
            scene_description=scene_description,
        )

    def generate_task_from_scene(
        self,
        detections: list[dict],
        scene_description: str,
        task_type: TherapyTaskType | None = None,
        difficulty: float = 0.5,
        disability_id: str | None = None,
    ) -> dict:
        """Build a therapy task dict from detections and a scene description.

        Parameters:
            detections: Perception detection records with class, bbox, urgency fields.
            scene_description: Natural-language scene summary for instructions.
            task_type: ``TherapyTaskType`` selecting the exercise template. When
                ``None`` and ``disability_id`` is provided, the ontology selects
                the appropriate type automatically.
            difficulty: Difficulty in ``[0, 1]`` affecting duration scaling.
            disability_id: Optional ontology disability ID used to auto-select
                the task type when ``task_type`` is not explicitly passed.

        Returns:
            JSON-serializable task dict with enum values converted to strings.
        """
        if task_type is None:
            if disability_id:
                task_type = self._task_type_for_disability(disability_id)
            if task_type is None:
                task_type = TherapyTaskType.ATTENTION_TRAINING

        builders: dict[
            TherapyTaskType,
            Callable[[list[dict], str, float], dict],
        ] = {
            TherapyTaskType.ATTENTION_TRAINING: self._build_attention_from_scene,
            TherapyTaskType.CONTRAST_RECOGNITION: self._build_contrast_from_scene,
            TherapyTaskType.EDGE_DETECTION: self._build_edge_from_scene,
            TherapyTaskType.SPATIAL_AWARENESS: self._build_spatial_from_scene,
            TherapyTaskType.WARNING_RECOGNITION: self._build_warning_from_scene,
        }
        builder = builders.get(task_type, self._build_attention_from_scene)
        return self._serialize_task(builder(detections, scene_description, difficulty))


def create_therapy_integrator() -> TherapyTaskIntegrator:
    """Factory function to create therapy task integrator."""
    return TherapyTaskIntegrator()
