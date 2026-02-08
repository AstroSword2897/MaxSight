"""File Magic Number Detection for Input Validation Detects file types by checking magic numbers (file signatures) to prevent malicious file uploads."""

from typing import Optional


# Common image magic numbers (first few bytes)
MAGIC_NUMBERS = {
    b"\xFF\xD8\xFF": "jpg",  # JPEG.
    b"\x89PNG\r\n\x1A\n": "png",  # PNG.
    b"GIF87a": "gif",  # GIF87a.
    b"GIF89a": "gif",  # GIF89a.
    b"BM": "bmp",  # BMP.
    b"RIFF": "webp",  # WEBP (needs more checking)
    b"II*\x00": "tiff",  # TIFF (little-endian)
    b"MM\x00*": "tiff",  # TIFF (big-endian)
}


def detect_magic(file_bytes: bytes) -> Optional[str]:
    """Detect file type from magic number (first few bytes). Args: file_bytes: Raw file bytes to check Returns: File type string (e.g., 'jpg', 'png') or None if unknown."""
    if len(file_bytes) < 4:
        return None
    
    for magic_bytes, file_type in MAGIC_NUMBERS.items():
        if file_bytes.startswith(magic_bytes):
            # WEBP requires both RIFF header and WEBP identifier at offset 8.
            if file_type == "webp" and len(file_bytes) >= 12:
                if b"WEBP" in file_bytes[8:12]:
                    return "webp"
                return None
            return file_type
    
    return None


def validate_image_magic(file_bytes: bytes, allowed_types: tuple = ('jpg', 'png', 'gif', 'bmp', 'webp', 'tiff')) -> bool:
    """Validate that file bytes match an allowed image type."""
    detected_type = detect_magic(file_bytes)
    return detected_type is not None and detected_type in allowed_types






