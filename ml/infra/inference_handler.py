"""SageMaker inference handler for MaxSight endpoints.

SageMaker calls model_fn / input_fn / predict_fn / output_fn.
The model artefact (model.tar.gz) must contain best.pt + model_meta.json.

Every response from predict_fn is an InferenceOutput TypedDict. Downstream
callers must read output_policy_applied and suppressed before acting on
therapy_feedback — bypassing the output policy layer is a correctness error.

Container env vars
------------------
SM_MODEL_DIR   Where SageMaker unpacks model.tar.gz.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml.middleware.error_sanitizer import (
    ConfigError,
    ModelInferenceError,
    ModelLoadError,
    UnsupportedContentTypeError,
    ValidationError,
    error_context,
    log_error,
)

if TYPE_CHECKING:
    from ml.therapy.therapy_integration import TherapyTaskIntegrator as _TTI

import torch

logger = logging.getLogger(__name__)


# ── Output contract ────────────────────────────────────────────────────────────

try:
    from typing import TypedDict

    class InferenceOutput(TypedDict, total=False):
        """Required output shape for all predict_fn responses.

        Downstream services must check output_policy_applied and suppressed
        before consuming therapy_feedback — the output policy layer sets these.
        """

        therapy_feedback: dict[str, Any]
        output_policy_applied: bool
        suppressed: bool

except ImportError:
    # Python <3.8 fallback — not expected in containers but safe to guard.
    InferenceOutput = dict  # type: ignore


# ── Lazy therapy initialiser ───────────────────────────────────────────────────

_THERAPY: _TTI | None = None


def _get_therapy() -> _TTI:
    """Return the shared TherapyTaskIntegrator, constructing it on first call.

    Deferred construction avoids import-time GPU/env-var side effects in
    SageMaker containers where the model isn't loaded until model_fn runs.
    """
    global _THERAPY
    if _THERAPY is None:
        from ml.therapy.therapy_integration import TherapyTaskIntegrator

        _THERAPY = TherapyTaskIntegrator()
    return _THERAPY


# ── SageMaker hooks ───────────────────────────────────────────────────────────


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load model from the SageMaker model directory."""
    import sys

    repo = Path(model_dir).parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model  # type: ignore
    except ImportError as exc:
        raise ConfigError(f"Cannot import model module from {repo}: {exc}") from exc

    meta_path = Path(model_dir) / "model_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to read model_meta.json: {exc}") from exc

    try:
        tier_name = meta.get("tier", "T2_DETECTOR")
        tier = CapabilityTier[tier_name]
        cfg = TierConfig.for_tier(tier)
        # create_model takes tier_config=; tier=/config= raise TypeError at load time.
        model = create_model(tier_config=cfg)
    except (KeyError, Exception) as exc:
        raise ModelLoadError(f"Failed to initialise model architecture: {exc}") from exc

    ckpt_path = Path(model_dir) / "best.pt"
    if ckpt_path.exists():
        try:
            state = torch.load(str(ckpt_path), map_location="cpu")
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint: %s", ckpt_path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint {ckpt_path}: {exc}") from exc
    else:
        logger.warning("No checkpoint at %s — using random weights", ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return {"model": model, "device": device, "meta": meta}


def input_fn(request_body: bytes, content_type: str = "application/json") -> dict[str, Any]:
    """Deserialise an inference request; raises typed errors on bad input."""
    if content_type == "application/json":
        try:
            payload = json.loads(request_body)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Request body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("JSON body must be an object, not a list or scalar.")
        return payload
    elif content_type == "application/octet-stream":
        return {"image_bytes": request_body}
    else:
        raise UnsupportedContentTypeError(
            f"Received content-type {content_type!r}. "
            "Accepted: application/json, application/octet-stream."
        )


def _predict_impl(data: dict[str, Any], model_pack: dict[str, Any]) -> InferenceOutput:
    """Inner predict logic — all exceptions here are wrapped by predict_fn."""
    import torchvision.transforms.functional as TF  # type: ignore
    from PIL import Image  # type: ignore

    from ml.therapy.therapy_integration import TherapyTaskType

    model = model_pack["model"]
    device = model_pack["device"]

    if "ping" in data:
        return {  # type: ignore[return-value]
            "status": "ok",
            "meta": model_pack.get("meta", {}),
            "therapy_feedback": {},
            "output_policy_applied": False,
            "suppressed": False,
        }

    if "image_b64" in data:
        try:
            img_bytes = base64.b64decode(data["image_b64"])
        except Exception as exc:
            raise ValidationError(f"image_b64 is not valid base64: {exc}") from exc
    elif "image_bytes" in data:
        img_bytes = data["image_bytes"]
    else:
        raise ValidationError("Payload must contain image_b64, image_bytes, or ping.")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise ValidationError(f"Could not decode image bytes: {exc}") from exc

    try:
        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            raw_outputs = model(tensor)
    except Exception as exc:
        raise ModelInferenceError(f"Forward pass failed: {exc}") from exc

    result: dict[str, Any] = {}
    for k, v in raw_outputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().tolist()
        else:
            result[k] = v

    # Required therapy pass — not optional.  Skipping this call bypasses the
    # safety and adaptation systems; it must remain in the call chain.
    try:
        detections: list[dict[str, Any]] = result.get("detections", [])
        scene_desc = (
            ", ".join(d.get("class_name", "object") for d in detections[:5]) or "inference scene"
        )
        therapy_feedback = _get_therapy().generate_task_from_scene(
            detections=detections,
            scene_description=scene_desc,
            task_type=TherapyTaskType.ATTENTION_TRAINING,
        )
    except Exception as exc:
        # Therapy failure is non-fatal: log and degrade gracefully.
        log_error(exc, context={"stage": "therapy_pass"})
        therapy_feedback = {}

    result["therapy_feedback"] = therapy_feedback
    result["output_policy_applied"] = False
    result["suppressed"] = False
    return result  # type: ignore[return-value]


def predict_fn(data: dict[str, Any], model_pack: dict[str, Any]) -> InferenceOutput:
    """Run inference on a single frame with structured error handling.

    All exceptions are caught, logged with a correlation ID, and re-raised as
    typed AppErrors so the SageMaker output layer (or test harness) can produce
    a sanitized response without exposing internal details.
    """
    with error_context() as eid:
        try:
            return _predict_impl(data, model_pack)
        except Exception as exc:
            log_error(
                exc,
                error_id=eid,
                context={
                    "stage": "predict_fn",
                    "model_tier": model_pack.get("meta", {}).get("tier", "unknown"),
                    "payload_keys": list(data.keys()),
                },
            )
            raise


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> bytes:
    """Serialise the prediction to the response format."""
    return json.dumps(prediction).encode()
