"""SageMaker inference handler for MaxSight endpoints.

SageMaker calls model_fn / input_fn / predict_fn / output_fn.
The model artefact (model.tar.gz) must contain best.pt + model_meta.json.

Container env vars
------------------
SM_MODEL_DIR   Where SageMaker unpacks model.tar.gz.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


# ── SageMaker hooks ───────────────────────────────────────────────────────────

def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load model from the SageMaker model directory."""
    import sys

    # Ensure repo is importable (copied into container by SageMaker).
    repo = Path(model_dir).parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier  # type: ignore

    meta_path = Path(model_dir) / "model_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    tier_name = meta.get("tier", "T2_DETECTOR")
    tier = CapabilityTier[tier_name]
    cfg = TierConfig.for_tier(tier)
    model = create_model(tier=tier, config=cfg)

    ckpt_path = Path(model_dir) / "best.pt"
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning("No checkpoint at %s — using random weights", ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return {"model": model, "device": device, "meta": meta}


def input_fn(request_body: bytes, content_type: str = "application/json") -> Dict[str, Any]:
    """Deserialise an inference request."""
    if content_type == "application/json":
        payload = json.loads(request_body)
    elif content_type == "application/octet-stream":
        payload = {"image_bytes": request_body}
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    return payload


def predict_fn(data: Dict[str, Any], model_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single frame."""
    from PIL import Image  # type: ignore
    import torchvision.transforms.functional as TF  # type: ignore

    model = model_pack["model"]
    device = model_pack["device"]

    # Decode image.
    if "image_b64" in data:
        img_bytes = base64.b64decode(data["image_b64"])
    elif "image_bytes" in data:
        img_bytes = data["image_bytes"]
    elif "ping" in data:
        return {"status": "ok", "meta": model_pack.get("meta", {})}
    else:
        raise ValueError("Payload must contain image_b64, image_bytes, or ping")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        outputs = model(tensor)

    # Serialise outputs (move tensors to CPU / Python scalars).
    result: Dict[str, Any] = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().tolist()
        else:
            result[k] = v
    return result


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> bytes:
    """Serialise the prediction to the response format."""
    return json.dumps(prediction).encode()
