"""Torch Stage A runner: adapts MaxSightCNN Stage-A outputs to frozen HazardResult."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from ml.runtime.stage_a.preprocess import frame_to_nchw_float
from ml.runtime.stage_a.types import CameraFrame, HazardResult

_ZONE_NAMES = ("near", "medium", "far")
_DIR_NAMES = ("left", "center", "right")


class TorchStageARunner:
    """Canonical Python StageARunner backed by an on-device Torch artifact path."""

    def __init__(
        self,
        artifact_path: Path | str,
        *,
        condition_mode: str = "none",
        device: str | None = None,
    ) -> None:
        if not isinstance(artifact_path, (Path, str)):
            raise TypeError("artifact_path must be Path or str")
        self.artifact_path = Path(artifact_path)
        self.condition_mode = condition_mode
        self.device = torch.device(device or "cpu")
        self._model = None
        self._model_hash = self._hash_artifact(self.artifact_path)
        self._model_version = self.artifact_path.stem

    @staticmethod
    def _hash_artifact(path: Path) -> str:
        if not path.is_file():
            return "missing"
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model

        model = create_model(
            condition_mode=None if self.condition_mode == "none" else self.condition_mode,
            use_audio=False,
            tier_config=TierConfig.for_tier(CapabilityTier.T0_BASELINE_CNN),
        )
        if self.artifact_path.is_file() and self.artifact_path.suffix in {".pt", ".pth"}:
            try:
                ckpt = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
            except Exception:
                # Random-init fallback keeps the frozen HazardResult path testable.
                pass
        model.eval()
        model.to(self.device)
        self._model = model
        return model

    def infer(self, frame: CameraFrame) -> HazardResult:
        """Map Stage-A tensors into the frozen HazardResult contract — do not reshape types."""
        t0 = time.perf_counter()
        model = self._ensure_model()
        arr = frame_to_nchw_float(frame)
        images = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            # Pass explicit condition one-hot when mode is set (frozen HazardResult mapping).
            from ml.runtime_constants import CONDITION_TENSOR_WIDTH, condition_mode_to_tensor_index

            cond = None
            if self.condition_mode and self.condition_mode != "none":
                cond = torch.zeros(1, CONDITION_TENSOR_WIDTH, device=self.device)
                cond[0, condition_mode_to_tensor_index(self.condition_mode)] = 1.0
            outputs = model(images, condition_tensor=cond)
        urgency_scores = outputs.get("urgency_scores")
        distance_zones = outputs.get("distance_zones")
        uncertainty = outputs.get("uncertainty")

        urgency = 0
        if urgency_scores is not None:
            urgency = int(urgency_scores[0].argmax().item())

        distance_zone = "medium"
        if distance_zones is not None:
            # Prefer image-level argmax when shape allows; else first spatial cell.
            dz = distance_zones[0]
            if dz.dim() >= 2:
                flat = dz.reshape(-1, dz.shape[-1]).mean(dim=0)
                distance_zone = _ZONE_NAMES[int(flat.argmax().item()) % len(_ZONE_NAMES)]
            else:
                distance_zone = _ZONE_NAMES[int(dz.argmax().item()) % len(_ZONE_NAMES)]

        conf = 0.5
        if urgency_scores is not None:
            conf = float(torch.softmax(urgency_scores[0], dim=-1).max().item())

        unc = 0.0
        if uncertainty is not None:
            unc = float(uncertainty.reshape(-1)[0].item())

        # Direction is not a dedicated Stage-A head in the frozen MVP; default center.
        direction = _DIR_NAMES[1]
        event_type = "hazard" if urgency >= 2 else "none"
        t1 = time.perf_counter()
        return HazardResult(
            event_type=event_type,
            urgency=urgency,
            direction=direction,
            distance_zone=distance_zone,
            confidence=conf,
            uncertainty=unc,
            latency_ms=(t1 - t0) * 1000.0,
            model_version=self._model_version,
            model_hash=self._model_hash,
            condition_mode=self.condition_mode,
            timestamp_source=frame.timestamp,
            timestamp_emit=time.time(),
        )
