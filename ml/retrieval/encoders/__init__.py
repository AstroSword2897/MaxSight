"""Retrieval encoders for multi-vector retrieval."""

# Make imports optional to handle missing dependencies gracefully.
try:
    from .global_encoder import GlobalEncoder
except (ImportError, ModuleNotFoundError):
    GlobalEncoder = None

try:
    from .region_extractor import RegionExtractor
except (ImportError, ModuleNotFoundError):
    RegionExtractor = None

try:
    from .patch_extractor import PatchExtractor
except (ImportError, ModuleNotFoundError):
    PatchExtractor = None

try:
    from .depth_extractor import DepthExtractor
except (ImportError, ModuleNotFoundError):
    DepthExtractor = None

try:
    from .ocr_encoder import OCREncoder
except (ImportError, ModuleNotFoundError):
    OCREncoder = None

try:
    from .audio_encoder import AudioEncoder
except (ImportError, ModuleNotFoundError):
    AudioEncoder = None

try:
    from .scene_graph_encoder import SceneGraphRetrievalEncoder
except (ImportError, ModuleNotFoundError):
    SceneGraphRetrievalEncoder = None

__all__ = [
    'GlobalEncoder',
    'RegionExtractor',
    'PatchExtractor',
    'DepthExtractor',
    'OCREncoder',
    'AudioEncoder',
    'SceneGraphRetrievalEncoder',
]







