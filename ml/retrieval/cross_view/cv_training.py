"""Cross-View Training and Augmentation for Robust Retrieval."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class CrossViewTrainer(nn.Module):
    """Multi-view contrastive learning for robust embeddings."""

    def __init__(self, embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def contrastive_loss(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss for cross-view learning."""
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.matmul(anchor, negatives.t()) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class CrossViewAugmentation:
    """Augmentations for cross-view robustness."""

    def __init__(self):
        self.augmentations = T.Compose(
            [
                T.RandomRotation(degrees=45),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentations."""
        return self.augmentations(image)
