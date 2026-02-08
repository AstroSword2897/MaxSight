"""Input Validation Utilities for MaxSight Provides validation for Base64, file uploads, and other user inputs."""

import base64
import binascii
from typing import Optional
from ml.security.magic import validate_image_magic


def is_valid_b64(s: str) -> bool:
    """Validate Base64 string format. Args: s: String to validate Returns: True if valid Base64, False otherwise."""
    try:
        base64.b64decode(s, validate=True)
        return True
    except (binascii.Error, ValueError):
        return False


def decode_and_validate_image(base64_str: str, max_size_mb: int = 10, allowed_types: tuple = ('jpg', 'png', 'gif', 'bmp', 'webp', 'tiff')) -> tuple[bool, Optional[bytes], Optional[str]]:
    """Decode Base64 image and validate format and size."""
    if not is_valid_b64(base64_str):
        return False, None, "Invalid Base64 format"
    
    try:
        decoded = base64.b64decode(base64_str)
        
        size_mb = len(decoded) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, None, f"File too large: {size_mb:.2f}MB > {max_size_mb}MB"
        
        # Validate file signature to prevent malicious uploads.
        if not validate_image_magic(decoded, allowed_types):
            return False, None, f"Invalid image format. Allowed: {allowed_types}"
        
        return True, decoded, None
        
    except Exception as e:
        return False, None, f"Decoding error: {str(e)}"






