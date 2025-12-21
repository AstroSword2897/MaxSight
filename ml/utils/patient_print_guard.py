"""
Patient Print Guard
Prevents direct print() usage in patient mode and enforces logging discipline.

This module provides:
1. A context manager that forbids print() in patient mode
2. A safe_print() function that routes to logger
3. Runtime enforcement to prevent regressions
"""

import sys
import logging
from contextlib import contextmanager
from typing import Optional
from ml.utils.runtime_output_contract import OutputValidator

logger = logging.getLogger(__name__)


class PrintGuardViolation(Exception):
    """Raised when print() is used in patient mode."""
    pass


class PatientPrintGuard:
    """
    Guards against direct print() usage in patient mode.
    
    Usage:
        guard = PatientPrintGuard(patient_mode=True)
        guard.enable()
        # Now print() will raise PrintGuardViolation
        guard.disable()
    """
    
    def __init__(self, patient_mode: bool = False):
        self.patient_mode = patient_mode
        self.original_stdout = None
        self.original_stderr = None
        self._enabled = False
    
    def enable(self):
        """Enable print guard."""
        if not self.patient_mode or self._enabled:
            return
        
        self._enabled = True
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Replace stdout/stderr with guarded versions
        sys.stdout = GuardedOutput(self.original_stdout, "stdout")
        sys.stderr = GuardedOutput(self.original_stderr, "stderr")
    
    def disable(self):
        """Disable print guard."""
        if not self._enabled:
            return
        
        self._enabled = False
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
    
    def __enter__(self):
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
        return False


class GuardedOutput:
    """
    Wrapper for stdout/stderr that blocks or redirects print() calls.
    """
    
    def __init__(self, original_stream, stream_name: str):
        self.original_stream = original_stream
        self.stream_name = stream_name
    
    def write(self, text: str):
        """Intercept write calls."""
        if text and text.strip():
            # Instead of printing, log it
            logger.warning(
                f"Attempted print() in patient mode blocked: {text[:100]}"
            )
            # Raise to enforce discipline
            raise PrintGuardViolation(
                f"Direct print() usage is forbidden in patient mode. "
                f"Use safe_print() or logging instead."
            )
        # Allow empty writes (some libraries write empty strings)
        return len(text)
    
    def flush(self):
        """Flush the original stream."""
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()
    
    def isatty(self):
        """Check if stream is a TTY."""
        if hasattr(self.original_stream, 'isatty'):
            return self.original_stream.isatty()
        return False


def safe_print(
    message: str,
    level: str = "INFO",
    validate_symbols: bool = True,
    patient_mode: bool = False
):
    """
    Safe print function that routes to logger and validates output.
    
    Arguments:
        message: Message to print
        level: Log level (INFO, WARNING, ERROR)
        validate_symbols: Whether to validate and remove symbols
        patient_mode: Whether in patient mode (enforces stricter rules)
    """
    # Validate and sanitize
    if validate_symbols:
        is_valid, error = OutputValidator.validate_message(
            message,
            mode="patient" if patient_mode else "dev"
        )
        if not is_valid:
            message = OutputValidator.sanitize_message(message)
            logger.debug(f"Message sanitized due to: {error}")
    
    # Route to logger
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


@contextmanager
def patient_mode_context(enabled: bool = True):
    """
    Context manager for patient mode execution.
    
    Usage:
        with patient_mode_context(enabled=True):
            # Any print() calls here will raise
            safe_print("This works fine")
    """
    guard = PatientPrintGuard(patient_mode=enabled)
    guard.enable()
    try:
        yield guard
    finally:
        guard.disable()


# Global guard instance
_global_guard: Optional[PatientPrintGuard] = None


def enable_patient_mode():
    """Enable patient mode globally."""
    global _global_guard
    if _global_guard is None:
        _global_guard = PatientPrintGuard(patient_mode=True)
    _global_guard.enable()


def disable_patient_mode():
    """Disable patient mode globally."""
    global _global_guard
    if _global_guard:
        _global_guard.disable()


def is_patient_mode_enabled() -> bool:
    """Check if patient mode is currently enabled."""
    global _global_guard
    return _global_guard is not None and _global_guard._enabled

