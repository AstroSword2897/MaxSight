"""UI Components Module Contains user interface components: - Voice feedback - Haptic feedback - Visual guidance See docs/therapy_system_implementation_plan.md Phase 4 for implementation details."""

from .haptic_feedback import HapticFeedback, HapticPattern
from .voice_feedback import VoiceFeedback, VoicePrompt

__all__ = ["VoiceFeedback", "VoicePrompt", "HapticFeedback", "HapticPattern"]
