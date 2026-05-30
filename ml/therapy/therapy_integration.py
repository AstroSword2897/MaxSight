"""Build scene-based therapy tasks from perception outputs for MaxSight."""

from typing import Dict, List, Optional, Tuple
from enum import Enum


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
        self.task_history = []

    def _serialize_task(self, payload: Dict) -> Dict:
        """Convert enum values to JSON-safe values for API responses."""
        serialized: Dict = {}
        for key, value in payload.items():
            if isinstance(value, Enum):
                serialized[key] = value.value
            elif isinstance(value, list):
                serialized[key] = [item.value if isinstance(item, Enum) else item for item in value]
            else:
                serialized[key] = value
        return serialized
    
    def create_attention_task(
        self,
        scene_description: str,
        target_objects: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create attention training task from scene description."""
        return {
            'task_type': TherapyTaskType.ATTENTION_TRAINING,
            'scene_description': scene_description,
            'target_objects': target_objects,
            'difficulty': difficulty,
            'instructions': f"Focus on: {', '.join(target_objects)}",
            'duration': int(30 + (1.0 - difficulty) * 30)  # 30-60 seconds.
        }
    
    def create_contrast_task(
        self,
        scene_description: str,
        contrast_levels: List[float],
        difficulty: float = 0.5
    ) -> Dict:
        """Create contrast recognition task from scene description."""
        return {
            'task_type': TherapyTaskType.CONTRAST_RECOGNITION,
            'scene_description': scene_description,
            'contrast_levels': contrast_levels,
            'difficulty': difficulty,
            'instructions': "Identify objects with different contrast levels",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_edge_task(
        self,
        scene_description: str,
        edge_types: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create edge detection task from scene description."""
        return {
            'task_type': TherapyTaskType.EDGE_DETECTION,
            'scene_description': scene_description,
            'edge_types': edge_types,
            'difficulty': difficulty,
            'instructions': f"Identify edges: {', '.join(edge_types)}",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_spatial_task(
        self,
        scene_description: str,
        spatial_relationships: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create spatial awareness task from scene description."""
        return {
            'task_type': TherapyTaskType.SPATIAL_AWARENESS,
            'scene_description': scene_description,
            'spatial_relationships': spatial_relationships,
            'difficulty': difficulty,
            'instructions': f"Identify relationships: {', '.join(spatial_relationships)}",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_warning_recognition_task(
        self,
        hazard_type: str,
        urgency_level: int,
        cue_description: str,
        difficulty: float = 0.5,
        scene_description: Optional[str] = None,
    ) -> Dict:
        """Create warning recognition task so the user learns to associate cues with hazards."""
        out: Dict = {
            'task_type': TherapyTaskType.WARNING_RECOGNITION,
            'hazard_type': hazard_type,
            'urgency_level': urgency_level,
            'cue_description': cue_description,
            'difficulty': difficulty,
            'instructions': (
                f"Learn the cue for {hazard_type}. "
                f"You will hear/feel: {cue_description}. "
                "When you hear this in real use, it means this hazard is present."
            ),
            'duration': int(20 + (1.0 - difficulty) * 25)
        }
        if scene_description is not None:
            out['scene_description'] = scene_description
        return out
    
    def generate_task_from_scene(
        self,
        detections: List[Dict],
        scene_description: str,
        task_type: TherapyTaskType,
        difficulty: float = 0.5
    ) -> Dict:
        """Build a therapy task dict from detections and a scene description.

        Parameters:
            detections: Perception detection records with class, bbox, urgency fields.
            scene_description: Natural-language scene summary for instructions.
            task_type: ``TherapyTaskType`` selecting the exercise template.
            difficulty: Difficulty in ``[0, 1]`` affecting duration scaling.

        Returns:
            JSON-serializable task dict with enum values converted to strings.
        """
        if task_type == TherapyTaskType.ATTENTION_TRAINING:
            # Extract target objects from detections.
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self._serialize_task(self.create_attention_task(scene_description, target_objects, difficulty))
        
        elif task_type == TherapyTaskType.CONTRAST_RECOGNITION:
            # Extract contrast levels from detections.
            contrast_levels = [d.get('contrast', 0.5) for d in detections if 'contrast' in d]
            if not contrast_levels:
                contrast_levels = [0.3, 0.5, 0.7]  # Default levels.
            return self._serialize_task(self.create_contrast_task(scene_description, contrast_levels, difficulty))
        
        elif task_type == TherapyTaskType.EDGE_DETECTION:
            edge_types = []
            for detection in detections:
                class_name = str(detection.get('class_name', '')).lower()
                if "door" in class_name:
                    edge_types.append("door_edge")
                if "stair" in class_name or "step" in class_name:
                    edge_types.append("stair_edge")
                if class_name:
                    edge_types.append(f"{class_name}_edge")
            if not edge_types:
                edge_types = ['obstacle_edge']
            return self._serialize_task(self.create_edge_task(scene_description, edge_types, difficulty))
        
        elif task_type == TherapyTaskType.SPATIAL_AWARENESS:
            relationships = []
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
                relationships = ['near']
            return self._serialize_task(self.create_spatial_task(scene_description, relationships, difficulty))
        
        elif task_type == TherapyTaskType.WARNING_RECOGNITION:
            # Use first high-urgency detection for warning recognition drill.
            hazard = next((d for d in detections if d.get('urgency', 0) >= 1), detections[0] if detections else {})
            hazard_type = hazard.get('class_name', 'obstacle')
            urgency_level = hazard.get('urgency', 1)
            return self._serialize_task(self.create_warning_recognition_task(
                hazard_type=hazard_type,
                urgency_level=urgency_level,
                cue_description=f"Alert for {hazard_type} (urgency {urgency_level})",
                difficulty=difficulty,
                scene_description=scene_description,
            ))
        
        else:
            # Default: attention task.
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self._serialize_task(self.create_attention_task(scene_description, target_objects, difficulty))


def create_therapy_integrator() -> TherapyTaskIntegrator:
    """Factory function to create therapy task integrator."""
    return TherapyTaskIntegrator()







