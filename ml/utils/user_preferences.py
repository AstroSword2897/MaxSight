"""User Preferences Management for MaxSight Handles user preference persistence, custom labels, and verbosity customization. Sprint 3 Day 28: User Customization."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class UserPreferences:
    """User preferences for MaxSight customization."""

    # Verbosity settings.
    verbosity_level: int = 1  # 0-3: brief, normal, detailed, very_detailed.
    detection_verbosity: int = 1  # Per-feature verbosity.
    ocr_verbosity: int = 1
    navigation_verbosity: int = 1

    # Alert frequency.
    alert_frequency: str = "medium"  # "low", "medium", "high"

    # Output channel preferences.
    preferred_channel: str = "audio"  # "audio", "visual", "haptic", "hybrid"
    audio_volume: float = 0.7  # 0.0-1.0.
    haptic_intensity: float = 0.7  # 0.0-1.0.

    # Condition-specific settings.
    condition_mode: str | None = None  # Vision condition (glaucoma, AMD, etc.)

    # Custom labels.
    custom_labels: dict[str, str] = field(default_factory=dict)

    # Adaptive assistance.
    enable_adaptive_assistance: bool = True
    adaptive_thresholds: dict[str, float] = field(default_factory=dict)

    # Timestamp.
    last_updated: float = 0.0

    def __post_init__(self):
        """Stamp first-use time when caller leaves last_updated unset."""
        if self.last_updated == 0.0:
            self.last_updated = time.time()


class UserPreferencesManager:
    """Manages user preferences persistence and customization. Sprint 3 Day 28: User Customization."""

    def __init__(self, preferences_file: Path | None = None):
        """Initialize preferences manager. Arguments: preferences_file: Path to preferences JSON file (default: ~/.maxsight/preferences.json)"""
        if preferences_file is None:
            # Default location.
            home_dir = Path.home()
            prefs_dir = home_dir / ".maxsight"
            prefs_dir.mkdir(exist_ok=True)
            preferences_file = prefs_dir / "preferences.json"

        self.preferences_file = Path(preferences_file)
        self.preferences: UserPreferences | None = None

    def load_preferences(self) -> UserPreferences:
        """Load user preferences from file. Returns: UserPreferences object."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file) as f:
                    data = json.load(f)

                # Convert to UserPreferences.
                prefs = UserPreferences(**data)
                self.preferences = prefs
                return prefs
            except Exception as e:
                print(f"Failed to load preferences: {e}, using defaults")

        # Return defaults.
        self.preferences = UserPreferences()
        return self.preferences

    def save_preferences(self, preferences: UserPreferences | None = None) -> bool:
        """Save user preferences to file."""
        if preferences is None:
            preferences = self.preferences

        if preferences is None:
            return False

        try:
            # Update timestamp.
            preferences.last_updated = time.time()

            # Ensure directory exists.
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to file.
            with open(self.preferences_file, "w") as f:
                json.dump(asdict(preferences), f, indent=2)

            self.preferences = preferences
            return True
        except Exception as e:
            print(f"Failed to save preferences: {e}")
            return False

    def update_verbosity(
        self,
        level: int | None = None,
        detection: int | None = None,
        ocr: int | None = None,
        navigation: int | None = None,
    ) -> bool:
        """Update verbosity levels."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None

        if level is not None:
            self.preferences.verbosity_level = max(0, min(3, level))
        if detection is not None:
            self.preferences.detection_verbosity = max(0, min(3, detection))
        if ocr is not None:
            self.preferences.ocr_verbosity = max(0, min(3, ocr))
        if navigation is not None:
            self.preferences.navigation_verbosity = max(0, min(3, navigation))

        return self.save_preferences()

    def add_custom_label(self, object_id: str, custom_name: str) -> bool:
        """Add or update a custom label for an object."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None

        self.preferences.custom_labels[object_id] = custom_name
        return self.save_preferences()

    def remove_custom_label(self, object_id: str) -> bool:
        """Remove a custom label. Arguments: object_id: Object identifier to remove Returns: True if removed successfully."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None

        if object_id in self.preferences.custom_labels:
            del self.preferences.custom_labels[object_id]
            return self.save_preferences()
        return False

    def get_custom_label(self, object_id: str) -> str | None:
        """Get custom label for an object. Arguments: object_id: Object identifier Returns: Custom label if exists, None otherwise."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None

        return self.preferences.custom_labels.get(object_id)

    def update_adaptive_thresholds(self, thresholds: dict[str, float]) -> bool:
        """Update adaptive assistance thresholds. Arguments: thresholds: Dictionary of threshold name -> value Returns: True if updated successfully."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None

        self.preferences.adaptive_thresholds.update(thresholds)
        return self.save_preferences()

    def get_preferences(self) -> UserPreferences:
        """Get current preferences (loads if not already loaded). Returns: UserPreferences object."""
        if self.preferences is None:
            self.load_preferences()
        assert self.preferences is not None
        return self.preferences
