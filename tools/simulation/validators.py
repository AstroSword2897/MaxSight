"""
Input validation for MaxSight Web Simulator.
Validates all user inputs before processing.
"""
from typing import Any, Dict, Optional
from PIL import Image
from io import BytesIO
import base64
from .exceptions import ValidationError, InvalidImageError, ImageTooLargeError
from .config import config


def validate_session_id(session_id: Optional[str]) -> str:
    """
    Validate session ID format.
    
    Args:
        session_id: Session ID to validate
    
    Returns:
        Validated session ID
    
    Raises:
        ValidationError: If session ID is invalid
    """
    if not session_id:
        raise ValidationError("Session ID is required")
    
    if not isinstance(session_id, str):
        raise ValidationError("Session ID must be a string")
    
    if len(session_id) < 10 or len(session_id) > 100:
        raise ValidationError("Session ID has invalid length")
    
    # Basic format check (UUID-like)
    if not all(c.isalnum() or c in '-_' for c in session_id):
        raise ValidationError("Session ID contains invalid characters")
    
    return session_id


def validate_condition(condition: str) -> str:
    """
    Validate visual condition.
    
    Args:
        condition: Condition name to validate
    
    Returns:
        Validated condition name
    
    Raises:
        ValidationError: If condition is invalid
    """
    valid_conditions = [
        'normal', 'myopia', 'hyperopia', 'astigmatism', 'cataracts',
        'glaucoma', 'amd', 'diabetic_retinopathy', 'retinitis_pigmentosa',
        'color_blindness', 'cvi', 'amblyopia', 'strabismus'
    ]
    
    if not condition:
        raise ValidationError("Condition is required")
    
    if condition not in valid_conditions:
        raise ValidationError(f"Invalid condition: {condition}. Must be one of {valid_conditions}")
    
    return condition


def validate_scenario(scenario: str) -> str:
    """
    Validate scenario name.
    
    Args:
        scenario: Scenario name to validate
    
    Returns:
        Validated scenario name
    
    Raises:
        ValidationError: If scenario is invalid
    """
    valid_scenarios = [
        'general', 'navigation', 'text_reading', 'therapy', 'safety', 'accessibility'
    ]
    
    if not scenario:
        raise ValidationError("Scenario is required")
    
    if scenario not in valid_scenarios:
        raise ValidationError(f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}")
    
    return scenario


def validate_output_mode(mode: str) -> str:
    """
    Validate output mode.
    
    Args:
        mode: Output mode to validate
    
    Returns:
        Validated output mode
    
    Raises:
        ValidationError: If mode is invalid
    """
    valid_modes = ['patient', 'clinician', 'dev']
    
    if not mode:
        raise ValidationError("Output mode is required")
    
    if mode not in valid_modes:
        raise ValidationError(f"Invalid output mode: {mode}. Must be one of {valid_modes}")
    
    return mode


def validate_image_file(image_bytes: bytes, max_size_mb: int = None) -> Image.Image:
    """
    Validate and load image file.
    
    Args:
        image_bytes: Image file bytes
        max_size_mb: Maximum size in MB (uses config if None)
    
    Returns:
        PIL Image object
    
    Raises:
        InvalidImageError: If image format is invalid
        ImageTooLargeError: If image is too large
    """
    if max_size_mb is None:
        max_size_mb = config.max_image_size_mb
    
    # Check size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ImageTooLargeError(
            f"Image too large: {size_mb:.2f}MB exceeds maximum {max_size_mb}MB"
        )
    
    if len(image_bytes) == 0:
        raise InvalidImageError("Empty image file")
    
    # Try to open image
    try:
        # Ensure we have bytes, not a BytesIO object
        if isinstance(image_bytes, BytesIO):
            image_bytes = image_bytes.read()
        
        # Create BytesIO buffer and ensure it's at position 0
        image_buffer = BytesIO(image_bytes)
        image_buffer.seek(0)
        
        # Open image
        image = Image.open(image_buffer)
        
        # Load image to verify it's valid (forces decoding)
        image.load()
        
        # Check format
        if image.format not in config.allowed_image_formats:
            raise InvalidImageError(
                f"Unsupported image format: {image.format}. "
                f"Supported formats: {', '.join(config.allowed_image_formats)}"
            )
        
        return image
        
    except Exception as e:
        if isinstance(e, (InvalidImageError, ImageTooLargeError)):
            raise
        raise InvalidImageError(f"Failed to open image: {str(e)}") from e


def validate_image_data(image_data: str) -> Image.Image:
    """
    Validate base64 encoded image data.
    
    Args:
        image_data: Base64 encoded image string (may include data URI prefix)
    
    Returns:
        PIL Image object
    
    Raises:
        InvalidImageError: If image data is invalid
    """
    try:
        # Remove data URI prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        return validate_image_file(image_bytes)
        
    except Exception as e:
        if isinstance(e, (InvalidImageError, ImageTooLargeError)):
            raise
        raise InvalidImageError(f"Failed to decode image data: {str(e)}") from e


def validate_init_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate /api/init request data.
    
    Args:
        data: Request JSON data
    
    Returns:
        Validated and normalized data
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    
    validated = {}
    
    # Condition (optional, defaults to 'normal')
    if 'condition' in data:
        validated['condition'] = validate_condition(data['condition'])
    else:
        validated['condition'] = 'normal'
    
    # Scenario (optional, defaults to 'general')
    if 'scenario' in data:
        validated['scenario'] = validate_scenario(data['scenario'])
    else:
        validated['scenario'] = 'general'
    
    # Output mode (optional, defaults to 'patient')
    if 'output_mode' in data:
        validated['output_mode'] = validate_output_mode(data['output_mode'])
    else:
        validated['output_mode'] = 'patient'
    
    # Start session (optional boolean)
    validated['start_session'] = bool(data.get('start_session', False))
    
    return validated

