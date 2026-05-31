"""Fusion MLP Training Script for Multi-Vector Retrieval Trains a fusion MLP that combines multiple embedding types. Based on provided script with enhancements."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class FusionDataset(Dataset):
    """Dataset for fusion MLP training."""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir)

        # Load embeddings.
        self.vision = np.load(self.root / "vision.npy")
        self.text = np.load(self.root / "text.npy")
        self.ocr = np.load(self.root / "ocr.npy")
        self.labels = np.load(self.root / "labels.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.vision[idx], dtype=torch.float32),
            torch.tensor(self.text[idx], dtype=torch.float32),
            torch.tensor(self.ocr[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class FusionMLP(nn.Module):
    """Fusion MLP for combining multiple embeddings."""

    def __init__(self, v_dim: int, t_dim: int, o_dim: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(v_dim + t_dim + o_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

        self.classifier = nn.Linear(out_dim, 1000)  # Adjust num_classes as needed.

    def forward(self, v, t, o):
        """Forward pass."""
        f = torch.cat([v, t, o], dim=-1)
        x = self.fc(f)
        logits = self.classifier(x)
        return x, logits


def train(
    data_dir: str = "data/",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "fusion_mlp.pt",
):
    """Training loop for fusion MLP."""

    dataset = FusionDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    v_dim = dataset.vision.shape[1]
    t_dim = dataset.text.shape[1]
    o_dim = dataset.ocr.shape[1]

    model = FusionMLP(v_dim, t_dim, o_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    cos_loss = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        total_loss = 0

        for v, t, o, label in loader:
            v, t, o, label = v.to(device), t.to(device), o.to(device), label.to(device)

            optimizer.zero_grad()

            fused_vec, logits = model(v, t, o)

            # Multi-task loss.
            loss1 = ce_loss(logits, label)
            target = torch.ones(len(v)).to(device)
            loss2 = cos_loss(fused_vec, v, target)  # Keep fused close to vision.

            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
