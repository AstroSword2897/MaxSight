"""Application-level components for MaxSight 3.0.

``PersonalMode`` pulls in torch, so it is loaded lazily to keep the light
``app.*`` submodules (connectivity, stage_b, model_update) importable in the
slim CI profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .personal_mode import PersonalMode

__all__ = ["PersonalMode"]


def __getattr__(name: str) -> Any:
    if name == "PersonalMode":
        from .personal_mode import PersonalMode

        return PersonalMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
