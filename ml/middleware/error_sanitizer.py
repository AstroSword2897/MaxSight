"""Error Sanitization Middleware Sanitizes error messages to prevent information leakage in production."""

import os
import traceback
from typing import Dict, Any


def sanitize_error(error: Exception, debug: bool = None) -> Dict[str, Any]:
    """Sanitize error for user-facing response."""
    if debug is None:
        debug = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes')
    
    if debug:
        # Show full error in debug mode.
        return {
            'error': str(error),
            'type': type(error).__name__,
            'traceback': traceback.format_exc()
        }
    else:
        # Generic error in production.
        return {
            'error': 'An internal error occurred. Please try again or contact support.',
            'type': 'InternalError'
        }


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log detailed error information server-side. Args: error: Exception that occurred context: Additional context information."""
    import logging
    logger = logging.getLogger(__name__)
    
    error_info = {
        'error': str(error),
        'type': type(error).__name__,
        'traceback': traceback.format_exc()
    }
    
    if context:
        error_info['context'] = context
    
    logger.error(f"Error occurred: {error_info}", exc_info=True)






