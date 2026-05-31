import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_run_sprint_self_tests_all_pass() -> None:
    from tools.simulation.sprint_self_tests import run_sprint_self_tests

    report = run_sprint_self_tests()
    assert report["ok"] is True, report
    assert report["passed"] == report["total"] == 5
    for r in report["results"]:
        assert set(r.keys()) == {"name", "ok", "detail", "ms"}
        assert r["ok"] is True
        assert isinstance(r["name"], str) and len(r["name"]) > 0
        assert r["ms"] >= 0.0


def test_run_manifest_json_check_invalid() -> None:
    from tools.simulation.sprint_self_tests import run_manifest_json_check

    r = run_manifest_json_check("not json")
    assert r["ok"] is False
    assert "errors" in r
    assert len(r["errors"]) >= 1


def test_run_manifest_json_check_valid_fixed_t8() -> None:
    from tools.simulation.sprint_self_tests import run_manifest_json_check

    manifest = """{
      "schema_version": "1.0",
      "clips": [{
        "clip_id": "x", "video_id": "v",
        "start_frame": 0, "end_frame": 8,
        "temporal_window": 8, "temporal_stride": 1, "temporal_overlap": 0,
        "frame_paths": ["a.jpg","b.jpg","c.jpg","d.jpg","e.jpg","f.jpg","g.jpg","h.jpg"],
        "frames_segments": [[],[],[],[],[],[],[],[]]
      }]
    }"""
    r = run_manifest_json_check(manifest)
    assert r["ok"] is True, r.get("errors")
    assert r.get("errors") in (None, [])
