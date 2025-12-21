"""
Production-grade logging configuration for MaxSight.

Provides centralized logging setup with:
- File and console handlers
- Proper log levels
- Structured formatting
- Rotation for log files
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Path = Path("logs")
) -> logging.Logger:
    """
    Setup production-grade logging configuration.
    
        Arguments:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file path
        log_dir: Directory for log files
    
    Returns:
        Configured root logger
    """
    # Create log directory if needed
    if log_file is None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "maxsight.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
    except ImportError:
        # Fallback to basic FileHandler
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
        Arguments:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# PATIENT PRINT GUARD (Enforces Print Discipline)
# ============================================================================
"""
Patient Print Guard
Prevents direct print() usage in patient mode and enforces logging discipline.

This section provides:
1. A context manager that forbids print() in patient mode
2. A safe_print() function that routes to logger
3. Runtime enforcement to prevent regressions
"""

from contextlib import contextmanager


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
            logger = logging.getLogger(__name__)
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
    patient_mode: bool = False
):
    """
    Safe print function that routes to logger.
    
    Arguments:
        message: Message to print
        level: Log level (INFO, WARNING, ERROR)
        patient_mode: Whether in patient mode (enforces stricter rules)
    """
    # Route to logger
    logger = logging.getLogger(__name__)
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

