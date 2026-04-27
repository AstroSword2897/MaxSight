"""Gold pipeline errors (kept local so ml.data does not import run_config)."""


class GoldConfigError(ValueError):
    """Raised for unsupported gold build modes (reserved for video, etc.)."""


__all__ = ["GoldConfigError"]
