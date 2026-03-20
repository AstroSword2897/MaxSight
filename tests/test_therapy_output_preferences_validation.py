import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_validate_init_request_default_therapy_prefs() -> None:
    from tools.simulation.validators import validate_init_request

    validated = validate_init_request({"scenario": "therapy", "output_mode": "patient"})
    assert validated["therapy_voice_enabled"] is True
    assert validated["therapy_haptic_enabled"] is True
    assert validated["therapy_preferred_channel"] == "audio"


def test_validate_init_request_allows_disabling_voice() -> None:
    from tools.simulation.validators import validate_init_request

    validated = validate_init_request(
        {
            "scenario": "therapy",
            "output_mode": "patient",
            "therapy_voice_enabled": False,
            "therapy_haptic_enabled": True,
            "therapy_preferred_channel": "audio",
        }
    )
    assert validated["therapy_voice_enabled"] is False
    assert validated["therapy_haptic_enabled"] is True
    assert validated["therapy_preferred_channel"] == "audio"


def test_validate_init_request_rejects_invalid_preferred_channel() -> None:
    import pytest
    from tools.simulation.validators import validate_init_request

    with pytest.raises(Exception):
        validate_init_request(
            {
                "scenario": "therapy",
                "output_mode": "patient",
                "therapy_preferred_channel": "invalid_channel",
            }
        )

