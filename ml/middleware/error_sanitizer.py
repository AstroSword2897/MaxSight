"""Production-grade error contract system for MaxSight.

Three concerns are kept strictly separate:

1. **Error taxonomy** — typed hierarchy so callers can distinguish client
   errors (400-range) from system errors (500-range) and retryable transients.
2. **Sanitization** — user-facing responses never expose internal paths,
   tracebacks, or model architecture details.  Debug depth is controlled by
   ``DEBUG_LEVEL`` (off / basic / deep), not a raw boolean switch.
3. **Structured logging** — every error emits a machine-parseable JSON record
   with a correlation ``error_id`` so a single event can be traced across
   logs, inference, tracker, and S3 without string-searching tracebacks.

Usage
-----
    from ml.middleware.error_sanitizer import (
        ValidationError,
        ModelInferenceError,
        sanitize_error,
        log_error,
        current_error_id,
    )

    # In a request handler:
    try:
        result = predict_fn(data, model)
    except Exception as exc:
        log_error(exc, context={"stage": "predict_fn", "model_tier": "T2"})
        return sanitize_error(exc), 500
"""

from __future__ import annotations

import logging
import os
import traceback
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# ── Debug level ────────────────────────────────────────────────────────────────
# Controlled by env var DEBUG_LEVEL: "off" | "basic" | "deep"
# "off"   → generic message only (production default)
# "basic" → error type + message, no traceback
# "deep"  → type + message + full traceback (internal tooling only)

_ENV_DEBUG = os.environ.get("DEBUG_LEVEL", "off").lower().strip()
if _ENV_DEBUG not in ("off", "basic", "deep"):
    # Tolerate legacy DEBUG=1 from old infra.
    _legacy = os.environ.get("DEBUG", "0").lower().strip()
    _ENV_DEBUG = "basic" if _legacy in ("1", "true", "yes") else "off"

_DEBUG_LEVEL: str = _ENV_DEBUG


def _debug_level() -> str:
    """Return effective debug level (reads env var at call time for test overrides)."""
    lvl = os.environ.get("DEBUG_LEVEL", "").lower().strip()
    if lvl in ("off", "basic", "deep"):
        return lvl
    legacy = os.environ.get("DEBUG", "0").lower().strip()
    return "basic" if legacy in ("1", "true", "yes") else "off"


# ── Correlation ID ─────────────────────────────────────────────────────────────
# Stored in a ContextVar so each async task / thread has its own ID.

_error_id_var: ContextVar[str | None] = ContextVar("error_id", default=None)


def generate_error_id() -> str:
    """Return a 12-character hex correlation ID (96 bits of randomness)."""
    return uuid.uuid4().hex[:12]


def current_error_id() -> str | None:
    """Return the correlation ID bound to the current execution context, if any."""
    return _error_id_var.get()


@contextmanager
def error_context(error_id: str | None = None) -> Generator[str, None, None]:
    """Bind a correlation ID to the current context for the duration of a block.

    Usage::

        with error_context() as eid:
            log_error(exc, context={"stage": "predict_fn"})
            return sanitize_error(exc)   # includes the same eid
    """
    eid = error_id or generate_error_id()
    token = _error_id_var.set(eid)
    try:
        yield eid
    finally:
        _error_id_var.reset(token)


# ── PII / secret redaction ─────────────────────────────────────────────────────

_REDACT_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "image_bytes",
        "image_b64",  # raw image data — large + potentially PII
        "patient_id",
        "user_id",  # MaxSight-specific PII
    }
)


def _sanitize_context(ctx: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``ctx`` with sensitive values replaced by ``[REDACTED]``.

    Keys are matched case-insensitively.  Values that are themselves dicts
    are recursed into.
    """
    if not ctx:
        return {}
    out: dict[str, Any] = {}
    for k, v in ctx.items():
        if k.lower() in _REDACT_KEYS:
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = _sanitize_context(v)
        else:
            out[k] = v
    return out


# ── Error taxonomy ─────────────────────────────────────────────────────────────


class AppError(Exception):
    """Base class for all MaxSight application errors.

    Subclasses declare ``http_status``, ``code``, and ``safe_message`` so
    callers get consistent structured responses without inspecting raw
    exception strings.
    """

    http_status: int = 500
    code: str = "internal_error"
    safe_message: str = "An internal error occurred."
    retryable: bool = False


class ValidationError(AppError):
    """Malformed or out-of-contract input from the caller."""

    http_status = 400
    code = "validation_error"
    safe_message = "Invalid input. Check the request format and try again."
    retryable = False


class UnsupportedContentTypeError(ValidationError):
    """Content-type not accepted by the endpoint."""

    code = "unsupported_content_type"
    safe_message = "Unsupported content type. Use application/json or application/octet-stream."


class ModelInferenceError(AppError):
    """The model produced an error during forward pass or post-processing."""

    http_status = 500
    code = "inference_error"
    safe_message = "Model inference failed. The request has been logged."
    retryable = False


class ModelLoadError(AppError):
    """The model artefact could not be loaded from the model directory."""

    http_status = 503
    code = "model_load_error"
    safe_message = "Model is not available. Please try again later."
    retryable = True


class ResourceError(AppError):
    """External resource (S3, registry, downstream service) is unavailable."""

    http_status = 503
    code = "resource_unavailable"
    safe_message = "A required resource is temporarily unavailable. Please retry."
    retryable = True


class ThrottleError(ResourceError):
    """Request rate limit exceeded on a downstream resource."""

    code = "throttle_error"
    safe_message = "Request rate limit exceeded. Please back off and retry."
    retryable = True


class ConfigError(AppError):
    """The system is misconfigured (bad env vars, missing artefacts, etc.)."""

    http_status = 500
    code = "config_error"
    safe_message = "The service is misconfigured. Contact support."
    retryable = False


# ── Sanitize (user-facing response) ───────────────────────────────────────────


def sanitize_error(
    error: Exception,
    *,
    error_id: str | None = None,
    debug_level: str | None = None,
) -> dict[str, Any]:
    """Return a safe, structured error dict for inclusion in HTTP responses.

    Parameters
    ----------
    error :
        The exception to sanitize.
    error_id :
        Correlation ID to include in the response so callers can reference it
        in support requests.  Falls back to ``current_error_id()`` then
        generates a fresh one.
    debug_level :
        Override the process-level ``DEBUG_LEVEL`` for this call.
        One of ``"off"``, ``"basic"``, ``"deep"``.

    Returns
    -------
    dict
        Always contains ``error``, ``code``, ``error_id``, ``retryable``.
        Optionally contains ``debug`` when ``debug_level`` is not ``"off"``.
    """
    eid = error_id or current_error_id() or generate_error_id()
    level = debug_level or _debug_level()

    if isinstance(error, AppError):
        response: dict[str, Any] = {
            "error": error.safe_message,
            "code": error.code,
            "http_status": error.http_status,
            "retryable": error.retryable,
            "error_id": eid,
        }
    else:
        response = {
            "error": "An internal error occurred.",
            "code": "internal_error",
            "http_status": 500,
            "retryable": False,
            "error_id": eid,
        }

    if level == "basic":
        response["debug"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    elif level == "deep":
        response["debug"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    # "off" → no debug key at all

    return response


# ── Structured logging ─────────────────────────────────────────────────────────


def log_error(
    error: Exception,
    *,
    context: dict[str, Any] | None = None,
    error_id: str | None = None,
    logger_: logging.Logger | None = None,
) -> str:
    """Log a structured error record server-side and return the correlation ID.

    The log record is a single JSON-serialisable ``extra`` dict so
    log aggregators (CloudWatch, Datadog, etc.) can parse it without regex.
    PII and secrets in ``context`` are redacted before writing.

    Parameters
    ----------
    error :
        The exception to log.
    context :
        Arbitrary key/value pairs describing where the error occurred.
        Sensitive keys are automatically redacted.
    error_id :
        Correlation ID.  Falls back to ``current_error_id()`` then generates
        a fresh one.  Always included in the log record.
    logger_ :
        Logger to use; defaults to this module's logger.

    Returns
    -------
    str
        The correlation ``error_id`` used in the log record.
    """
    eid = error_id or current_error_id() or generate_error_id()
    target = logger_ or logger

    safe_ctx = _sanitize_context(context)

    extra: dict[str, Any] = {
        "error_id": eid,
        "error_type": type(error).__name__,
        "error_code": getattr(error, "code", "internal_error"),
        "error_message": str(error),
        "retryable": getattr(error, "retryable", False),
        "http_status": getattr(error, "http_status", 500),
        "traceback": traceback.format_exc(),
        "context": safe_ctx,
    }

    target.error(
        "maxsight_error_event error_id=%s type=%s code=%s",
        eid,
        extra["error_type"],
        extra["error_code"],
        extra={"maxsight_error": extra},
    )
    return eid
