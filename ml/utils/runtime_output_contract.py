"""
Runtime Output Contract
Enforces strict schema for all patient-facing outputs to ensure safety and predictability.

This module implements the runtime output contract that ensures:
1. Patient mode only receives safe, minimal information
2. No debug fields leak to patients
3. All outputs conform to a strict schema
4. Symbol characters are rejected at runtime
"""

from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import re


class OutputMode(Enum):
    """Output mode for different user types."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    DEV = "dev"


class Severity(Enum):
    """Message severity levels."""
    INFO = "info"
    WARNING = "warning"
    HAZARD = "hazard"
    CRITICAL = "critical"


@dataclass
class RuntimeOutput:
    """
    Strict runtime output contract.
    
    All user-facing messages must conform to this schema.
    Patient mode only receives: mode, severity, message, confidence, cooldown_applied.
    """
    mode: OutputMode
    severity: Severity
    message: str
    confidence: float  # 0.0-1.0
    cooldown_applied: bool
    
    # Optional fields (only for clinician/dev modes)
    latency_ms: Optional[float] = None
    uncertainty: Optional[float] = None
    fallback_used: Optional[bool] = None
    component_breakdown: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, respecting mode constraints."""
        result = {
            'mode': self.mode.value,
            'severity': self.severity.value,
            'message': self.message,
            'confidence': self.confidence,
            'cooldown_applied': self.cooldown_applied
        }
        
        # Only include extended fields for clinician/dev modes
        if self.mode in (OutputMode.CLINICIAN, OutputMode.DEV):
            if self.latency_ms is not None:
                result['latency_ms'] = self.latency_ms
            if self.uncertainty is not None:
                result['uncertainty'] = self.uncertainty
            if self.fallback_used is not None:
                result['fallback_used'] = self.fallback_used
            if self.component_breakdown is not None:
                result['component_breakdown'] = self.component_breakdown
            
        # Debug info only for dev mode
        if self.mode == OutputMode.DEV and self.debug_info is not None:
            result['debug_info'] = self.debug_info
        
        return result


class OutputValidator:
    """
    Validates runtime outputs against the contract.
    Rejects symbol characters and enforces schema constraints.
    """
    
    # Forbidden symbol characters (anything that's not alphanumeric, space, or basic punctuation)
    FORBIDDEN_SYMBOLS_PATTERN = re.compile(r'[✅❌✓✗✔✖=×→←↑↓▶◀▲▼■□●○★☆♦♥♠♣]')
    
    @classmethod
    def validate_message(cls, message: str, mode: OutputMode) -> tuple[bool, Optional[str]]:
        """
        Validate a message string.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for forbidden symbols
        if cls.FORBIDDEN_SYMBOLS_PATTERN.search(message):
            return False, "Message contains forbidden symbol characters"
        
        # Patient mode: enforce brevity
        if mode == OutputMode.PATIENT:
            if len(message) > 200:
                return False, "Patient mode messages must be under 200 characters"
        
        # Basic sanity checks
        if not message or not message.strip():
            return False, "Message cannot be empty"
        
        return True, None
    
    @classmethod
    def validate_confidence(cls, confidence: float) -> tuple[bool, Optional[str]]:
        """Validate confidence value."""
        if not (0.0 <= confidence <= 1.0):
            return False, f"Confidence must be in [0.0, 1.0], got {confidence}"
        return True, None
    
    @classmethod
    def validate_output(cls, output: RuntimeOutput) -> tuple[bool, Optional[str]]:
        """
        Validate a complete RuntimeOutput.
        
        Returns:
            (is_valid, error_message)
        """
        # Validate message
        is_valid, error = cls.validate_message(output.message, output.mode)
        if not is_valid:
            return False, error
        
        # Validate confidence
        is_valid, error = cls.validate_confidence(output.confidence)
        if not is_valid:
            return False, error
        
        # Patient mode must not have debug fields
        if output.mode == OutputMode.PATIENT:
            if output.debug_info is not None:
                return False, "Patient mode cannot include debug_info"
            # Allow component_breakdown in clinician mode only
            if output.component_breakdown is not None:
                return False, "Patient mode cannot include component_breakdown"
        
        return True, None
    
    @classmethod
    def sanitize_message(cls, message: str) -> str:
        """Remove forbidden symbols from message."""
        return cls.FORBIDDEN_SYMBOLS_PATTERN.sub('', message)


def create_patient_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False
) -> RuntimeOutput:
    """
    Create a patient-safe output.
    Automatically sanitizes message and validates.
    """
    # Sanitize message
    message = OutputValidator.sanitize_message(message)
    
    # Truncate if too long
    if len(message) > 200:
        message = message[:197] + "..."
    
    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.PATIENT,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied
    )


def create_clinician_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False,
    latency_ms: Optional[float] = None,
    uncertainty: Optional[float] = None,
    fallback_used: Optional[bool] = None,
    component_breakdown: Optional[Dict[str, Any]] = None
) -> RuntimeOutput:
    """Create a clinician-mode output with extended metrics."""
    message = OutputValidator.sanitize_message(message)
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.CLINICIAN,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied,
        latency_ms=latency_ms,
        uncertainty=uncertainty,
        fallback_used=fallback_used,
        component_breakdown=component_breakdown
    )


def create_dev_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False,
    latency_ms: Optional[float] = None,
    uncertainty: Optional[float] = None,
    fallback_used: Optional[bool] = None,
    component_breakdown: Optional[Dict[str, Any]] = None,
    debug_info: Optional[Dict[str, Any]] = None
) -> RuntimeOutput:
    """Create a dev-mode output with full debug information."""
    message = OutputValidator.sanitize_message(message)
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.DEV,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied,
        latency_ms=latency_ms,
        uncertainty=uncertainty,
        fallback_used=fallback_used,
        component_breakdown=component_breakdown,
        debug_info=debug_info
    )

