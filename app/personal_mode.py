"""User-specific personalization for fusion weights, exploration, and output scaling."""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ml.retrieval.fusion.meta_fusion import (
    ActiveSceneExploration,
    MetaFusionWeights,
    PredictiveNavigationGuidance,
    UserProfile,
)


@dataclass
class PersonalizationState:
    """Per-user personalization state tracked across sessions.

    Attributes:
        user_id: Stable user identifier.
        preferences: Task-type preference scores updated via EMA.
        task_history: Recent task type names (bounded list).
        performance_metrics: Per-task score history used for adaptation.
        adaptation_count: Number of preference updates performed.
        adapted_weights: Latest fusion weights from meta-learning adaptation.
    """

    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)
    task_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)
    adaptation_count: int = 0
    adapted_weights: Optional[torch.Tensor] = None


class PersonalMode:
    """Manage personalization for fusion, exploration, navigation, and outputs.

    Integrate at session start by constructing one ``PersonalMode`` per runtime
    instance, then call ``update_preferences`` after each task and
    ``fuse_with_personalization`` / ``get_personalized_outputs`` during inference.
    """

    def __init__(self, embed_dim: int = 256, num_modalities: int = 3) -> None:
        """Initialize personalization subsystems.

        Parameters:
            embed_dim: Embedding dimension shared by fusion/exploration modules.
            num_modalities: Number of modalities (vision, audio, haptic).

        Side effects:
            Constructs neural modules for fusion, exploration, and navigation.
        """
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.meta_fusion = MetaFusionWeights(num_modalities=num_modalities, embed_dim=embed_dim)
        self.active_exploration = ActiveSceneExploration(embed_dim=embed_dim)
        self.navigation_guidance = PredictiveNavigationGuidance(embed_dim=embed_dim)
        self.user_states: Dict[str, PersonalizationState] = {}

    def get_user_state(self, user_id: str) -> PersonalizationState:
        """Return existing user state or create a new one.

        Parameters:
            user_id: Stable user identifier.

        Returns:
            ``PersonalizationState`` for the given user.
        """
        if user_id not in self.user_states:
            self.user_states[user_id] = PersonalizationState(user_id=user_id)
        return self.user_states[user_id]

    def update_preferences(
        self,
        user_id: str,
        task_type: str,
        performance_score: float,
        preferred_modalities: Optional[List[str]] = None,
    ) -> None:
        """Record task performance and periodically adapt fusion weights.

        Parameters:
            user_id: User receiving the update.
            task_type: Task identifier (e.g. navigation, therapy).
            performance_score: Normalized score in ``[0, 1]``.
            preferred_modalities: Optional modality preference list.

        Side effects:
            Updates in-memory user state and may refresh ``adapted_weights``.
        """
        state = self.get_user_state(user_id)
        state.task_history.append(task_type)
        if len(state.task_history) > 100:
            state.task_history.pop(0)

        if task_type not in state.performance_metrics:
            state.performance_metrics[task_type] = []
        state.performance_metrics[task_type].append(performance_score)

        if task_type not in state.preferences:
            state.preferences[task_type] = performance_score
        else:
            alpha = 0.1
            state.preferences[task_type] = (
                alpha * performance_score + (1 - alpha) * state.preferences[task_type]
            )

        if state.adaptation_count % 10 == 0:
            user_profile = UserProfile(
                user_id=user_id,
                preferred_modalities=preferred_modalities or ["vision", "audio"],
                task_preferences=state.preferences,
            )
            adapted_weights = self.meta_fusion.adapt_to_user(user_profile, state.performance_metrics)
            state.adapted_weights = adapted_weights.detach().clone()

        state.adaptation_count += 1

    def fuse_with_personalization(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
        urgency: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality embeddings with user-specific weight adaptation.

        Parameters:
            modality_embeddings: Dict mapping modality name to ``[B, D]`` tensor.
            user_id: Optional user identifier for personalized fusion.
            task_type: Optional task hint for fusion policy.
            urgency: Optional urgency score in ``[0, 1]``.
            confidence: Optional confidence score in ``[0, 1]``.

        Returns:
            Tuple of ``(fused_embedding, fusion_weights)``.

        Failure modes:
            Raises if ``modality_embeddings`` is empty.
        """
        user_id_tensor = None
        if user_id is not None:
            digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
            user_hash = int(digest, 16) % 1000
            user_id_tensor = torch.tensor(
                [user_hash], device=next(iter(modality_embeddings.values())).device
            )

        fused_embedding, fusion_weights = self.meta_fusion(
            modality_embeddings=modality_embeddings,
            user_id=user_id_tensor,
            task_type=task_type,
            urgency=urgency,
            confidence=confidence,
        )
        if user_id is not None:
            state = self.get_user_state(user_id)
            if state.adapted_weights is not None:
                device = fused_embedding.device
                adapted = state.adapted_weights.to(device=device)
                modality_names = ["vision", "audio", "haptic"]
                modality_boost = 0.0
                present_modalities = 0
                for idx, name in enumerate(modality_names):
                    if name in modality_embeddings:
                        modality_boost += float(adapted[idx].item())
                        present_modalities += 1
                if present_modalities > 0:
                    scale = (modality_boost / present_modalities) * self.num_modalities
                    fused_embedding = fused_embedding * float(scale)
        return fused_embedding, fusion_weights

    def explore_scene(
        self,
        region_embeddings: torch.Tensor,
        uncertainties: torch.Tensor,
        user_id: Optional[str] = None,
        urgency: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score regions for active exploration.

        Parameters:
            region_embeddings: ``[B, N, D]`` region feature tensor.
            uncertainties: ``[B, N]`` uncertainty scores.
            user_id: Optional user identifier for preference bias.
            urgency: Optional urgency hint.

        Returns:
            Tuple of ``(exploration_scores, selected_region_indices)``.
        """
        user_preference = None
        if user_id:
            state = self.get_user_state(user_id)
            if state.preferences:
                user_preference = sum(state.preferences.values()) / len(state.preferences)

        urgency_tensor = None
        if urgency is not None:
            urgency_tensor = torch.tensor([urgency], device=region_embeddings.device)

        return self.active_exploration(
            region_embeddings=region_embeddings,
            uncertainties=uncertainties,
            urgency=urgency_tensor,
            user_preference=user_preference,
        )

    def predict_navigation(
        self,
        current_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict navigation guidance from current and goal embeddings.

        Returns:
            Dict with direction, distance, confidence, and guidance priority tensors.
        """
        return self.navigation_guidance(
            current_embedding=current_embedding,
            goal_embedding=goal_embedding,
            scene_context=scene_context,
        )

    def get_personalized_outputs(
        self,
        model_outputs: Dict[str, torch.Tensor],
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Scale tensor outputs using user task preferences.

        Parameters:
            model_outputs: Model output dict (shallow-copied before mutation).
            user_id: Optional user identifier.
            task_type: Task type used to lookup preference score.

        Returns:
            Copy of outputs with tensor values scaled when preference exists.
        """
        personalized = model_outputs.copy()
        if user_id and task_type in self.get_user_state(user_id).preferences:
            preference = self.get_user_state(user_id).preferences[task_type]
            scale_factor = 0.5 + preference
            for key, value in list(personalized.items()):
                if torch.is_tensor(value):
                    personalized[key] = value * float(scale_factor)
        return personalized
