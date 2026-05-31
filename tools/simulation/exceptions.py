"""Custom exceptions for MaxSight Web Simulator. Provides consistent error handling hierarchy."""


class MaxSightSimulatorError(Exception):
    """Base exception for all simulator errors."""

    pass


class SessionError(MaxSightSimulatorError):
    """Session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Session ID not found."""

    pass


class SessionExpiredError(SessionError):
    """Session has expired."""

    pass


class SessionNotInitializedError(SessionError):
    """Session not properly initialized."""

    pass


class ImageProcessingError(MaxSightSimulatorError):
    """Image processing errors."""

    pass


class InvalidImageError(ImageProcessingError):
    """Invalid or unsupported image format."""

    pass


class ImageTooLargeError(ImageProcessingError):
    """Image file too large."""

    pass


class RateLimitError(MaxSightSimulatorError):
    """Rate limit exceeded."""

    pass


class ValidationError(MaxSightSimulatorError):
    """Input validation error."""

    pass


class ModelError(MaxSightSimulatorError):
    """Model inference errors."""

    pass
