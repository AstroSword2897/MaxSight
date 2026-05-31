"""PCA and cross-modality feature scaling with persisted artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureTransformArtifact:
    """Persisted PCA + scaler state for train/infer parity."""

    mean: np.ndarray
    scale: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int

    def save(self, path: Path) -> None:
        """Write artifact JSON with numpy arrays as lists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "pca_components": self.pca_components.tolist(),
            "pca_mean": self.pca_mean.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "n_components": self.n_components,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureTransformArtifact:
        """Load artifact from JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            pca_components=np.asarray(payload["pca_components"], dtype=np.float64),
            pca_mean=np.asarray(payload["pca_mean"], dtype=np.float64),
            explained_variance_ratio=np.asarray(
                payload["explained_variance_ratio"], dtype=np.float64
            ),
            n_components=int(payload["n_components"]),
        )


def fit_feature_transform(
    features: np.ndarray,
    *,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> FeatureTransformArtifact:
    """Fit standard scaler + PCA on feature matrix (n_samples, n_features)."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must be 2D with at least 2 samples")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (features - mean) / scale
    cov = np.cov(scaled, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum()) or 1.0
    ratios = eigvals / total
    if n_components is None:
        cumulative = np.cumsum(ratios)
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_components = max(1, min(n_components, scaled.shape[1]))
    components = eigvecs[:, :n_components].T
    return FeatureTransformArtifact(
        mean=mean,
        scale=scale,
        pca_components=components,
        pca_mean=mean.copy(),
        explained_variance_ratio=ratios[:n_components],
        n_components=n_components,
    )


def transform_features(artifact: FeatureTransformArtifact, features: np.ndarray) -> np.ndarray:
    """Apply persisted scaler + PCA transform."""
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scaled = (features - artifact.mean) / artifact.scale
    centered = scaled - artifact.pca_mean
    return centered @ artifact.pca_components.T


def transform_summary(artifact: FeatureTransformArtifact) -> dict[str, Any]:
    """Return metadata for logging and CI checks."""
    return {
        "n_components": artifact.n_components,
        "explained_variance_sum": float(artifact.explained_variance_ratio.sum()),
    }
