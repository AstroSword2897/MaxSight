import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.simulation.exceptions import ImageTooLargeError, ValidationError  # noqa: E402
from tools.simulation.validators import validate_frames_data  # noqa: E402


def _to_b64_png(size: int = 8, noisy: bool = False) -> str:
    if noisy:
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.new("RGB", (size, size), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_validate_frames_data_accepts_small_batch() -> None:
    frames_data = [_to_b64_png(), _to_b64_png()]
    out = validate_frames_data(frames_data, max_frames=4, max_payload_mb=2.0)
    assert out["count"] == 2
    assert len(out["frames"]) == 2
    assert out["total_decoded_bytes"] > 0
    assert all(hasattr(im, "size") for im in out["frames"])


def test_validate_frames_data_rejects_count_over_limit() -> None:
    frames_data = [_to_b64_png() for _ in range(3)]
    with pytest.raises(ValidationError) as exc:
        validate_frames_data(frames_data, max_frames=2, max_payload_mb=2.0)
    assert "maximum length" in str(exc.value).lower() or "exceeds" in str(exc.value).lower()


def test_validate_frames_data_rejects_large_payload() -> None:
    frames_data = [_to_b64_png(size=512, noisy=True) for _ in range(4)]
    with pytest.raises(ImageTooLargeError) as exc:
        validate_frames_data(frames_data, max_frames=16, max_payload_mb=0.01)
    assert "payload" in str(exc.value).lower() or "large" in str(exc.value).lower()


def test_validate_frames_data_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        validate_frames_data([], max_frames=4, max_payload_mb=2.0)


def test_validate_frames_data_rejects_non_string_element() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_frames_data([123], max_frames=4, max_payload_mb=2.0)
    assert "base64" in str(exc.value).lower() or "string" in str(exc.value).lower()
