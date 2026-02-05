"""
Tests for therapy modules: SessionManager, TaskGenerator, TherapyTaskIntegrator, TherapyStateHead.

Ensures therapy methods work end-to-end for session tracking, task generation,
scene-based therapy tasks, and the therapy state model head.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

# Therapy modules
from ml.therapy.session_manager import SessionManager
from ml.therapy.task_generator import TaskGenerator, TaskType
from ml.therapy.therapy_integration import (
    TherapyTaskIntegrator,
    TherapyTaskType,
    create_therapy_integrator,
)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

def test_session_manager_start_and_end():
    """SessionManager starts a session and returns a valid session ID."""
    mgr = SessionManager(user_id="test_user")
    sid = mgr.start_session()
    assert sid.startswith("session_")
    assert mgr.current_session is not None
    assert mgr.current_session["session_id"] == sid
    assert "start_time" in mgr.current_session
    assert mgr.current_session["metrics"]["total_tasks"] == 0

    report = mgr.end_session()
    assert "session_id" in report
    assert "skill_curve" in report
    assert "summary" in report
    assert mgr.current_session is None


def test_session_manager_log_task_attempt():
    """SessionManager logs task attempts and updates metrics."""
    mgr = SessionManager()
    mgr.start_session()

    mgr.log_task_attempt(
        task_type="contrast_micro",
        task_config={"difficulty": 0.5},
        result={"success": True, "reaction_time": 1.2, "misses": 0, "fails": 0},
    )
    mgr.log_task_attempt(
        task_type="motion_tracking",
        task_config={"difficulty": 0.7},
        result={"success": False, "reaction_time": 2.0, "misses": 1, "fails": 1},
    )

    assert mgr.current_session is not None
    assert mgr.current_session["metrics"]["total_tasks"] == 2
    assert mgr.current_session["metrics"]["completed_tasks"] == 1
    assert mgr.current_session["metrics"]["failed_tasks"] == 1
    assert mgr.current_session["metrics"]["total_time"] == pytest.approx(3.2)

    report = mgr.end_session()
    assert report and "summary" in report
    assert report["summary"]["success_rate"] == 0.5
    assert report["summary"]["total_tasks"] == 2
    assert len(report["skill_curve"]) == 2


def test_session_manager_skill_curve():
    """SessionManager produces a skill curve with cumulative success rate."""
    mgr = SessionManager()
    mgr.start_session()
    for i in range(3):
        mgr.log_task_attempt(
            task_type="roi",
            task_config={},
            result={"success": i < 2, "reaction_time": 1.0},
        )
    report = mgr.end_session()
    curve = report["skill_curve"]
    assert len(curve) == 3
    assert curve[0]["cumulative_success_rate"] == pytest.approx(1.0)
    assert curve[1]["cumulative_success_rate"] == pytest.approx(1.0)
    assert curve[2]["cumulative_success_rate"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# TaskGenerator
# ---------------------------------------------------------------------------

def test_task_generator_fatigue_rest():
    """TaskGenerator returns FATIGUE_REST when fatigue is high."""
    gen = TaskGenerator()
    task = gen.generate_task(uncertainty=0.2, fatigue_score=0.8, recent_performance=[])
    assert task["task_type"] == TaskType.FATIGUE_REST
    assert task["difficulty"] == 0.0
    assert task["duration"] == 60
    assert task["target_speed"] == 0.0


def test_task_generator_adaptive_difficulty():
    """TaskGenerator reduces difficulty when uncertainty is high."""
    gen = TaskGenerator()
    task_lo = gen.generate_task(uncertainty=0.9, fatigue_score=0.0, recent_performance=[])
    task_hi = gen.generate_task(uncertainty=0.1, fatigue_score=0.0, recent_performance=[])
    assert task_lo["difficulty"] < task_hi["difficulty"]
    assert task_lo["duration"] >= 30
    assert task_hi["duration"] <= 60


def test_task_generator_choose_task_type():
    """TaskGenerator cycles through task types when history is present."""
    gen = TaskGenerator()
    gen.task_history = [{"task_type": TaskType.CONTRAST_MICRO}]
    task = gen.generate_task(uncertainty=0.3, fatigue_score=0.0, recent_performance=[])
    assert task["task_type"] in (
        TaskType.CONTRAST_MICRO,
        TaskType.MOTION_TRACKING,
        TaskType.DEPTH_SHIFT,
        TaskType.GAZE_STABILIZATION,
        TaskType.ROI_FINDABILITY,
    )


def test_task_generator_update_performance():
    """TaskGenerator records performance and recent failures."""
    gen = TaskGenerator()
    gen.update_performance({"task_type": TaskType.CONTRAST_MICRO, "failed": False})
    gen.update_performance({"task_type": TaskType.MOTION_TRACKING, "failed": True})
    assert len(gen.task_history) == 2
    assert len(gen.recent_failures) == 1


# ---------------------------------------------------------------------------
# TherapyTaskIntegrator
# ---------------------------------------------------------------------------

def test_therapy_integrator_attention_task():
    """TherapyTaskIntegrator creates attention task with scene description."""
    integrator = TherapyTaskIntegrator()
    task = integrator.create_attention_task(
        scene_description="A door and a stairway.",
        target_objects=["door", "stairs"],
        difficulty=0.6,
    )
    assert task["task_type"] == TherapyTaskType.ATTENTION_TRAINING
    assert "door" in task["instructions"] and "stairs" in task["instructions"]
    assert task["difficulty"] == 0.6
    assert 30 <= task["duration"] <= 60


def test_therapy_integrator_contrast_task():
    """TherapyTaskIntegrator creates contrast recognition task."""
    integrator = TherapyTaskIntegrator()
    task = integrator.create_contrast_task(
        scene_description="Mixed lighting.",
        contrast_levels=[0.3, 0.5, 0.8],
        difficulty=0.5,
    )
    assert task["task_type"] == TherapyTaskType.CONTRAST_RECOGNITION
    assert task["contrast_levels"] == [0.3, 0.5, 0.8]
    assert "contrast" in task["instructions"].lower()


def test_therapy_integrator_edge_task():
    """TherapyTaskIntegrator creates edge detection task."""
    integrator = TherapyTaskIntegrator()
    task = integrator.create_edge_task(
        scene_description="Room with furniture.",
        edge_types=["door_edge", "stair_edge"],
        difficulty=0.4,
    )
    assert task["task_type"] == TherapyTaskType.EDGE_DETECTION
    assert "door_edge" in task["edge_types"]
    assert "Identify edges" in task["instructions"]


def test_therapy_integrator_spatial_task():
    """TherapyTaskIntegrator creates spatial awareness task."""
    integrator = TherapyTaskIntegrator()
    task = integrator.create_spatial_task(
        scene_description="Objects in a room.",
        spatial_relationships=["left_of", "near", "above"],
        difficulty=0.5,
    )
    assert task["task_type"] == TherapyTaskType.SPATIAL_AWARENESS
    assert "left_of" in task["instructions"]


def test_therapy_integrator_generate_task_from_scene():
    """TherapyTaskIntegrator generates tasks from scene detections."""
    integrator = TherapyTaskIntegrator()
    detections = [
        {"class_name": "door", "contrast": 0.5},
        {"class_name": "person"},
    ]
    scene = "A door and a person."

    for task_type in TherapyTaskType:
        task = integrator.generate_task_from_scene(
            detections=detections,
            scene_description=scene,
            task_type=task_type,
            difficulty=0.5,
        )
        assert "task_type" in task
        assert "scene_description" in task
        assert "difficulty" in task
        assert "duration" in task


def test_create_therapy_integrator_factory():
    """Factory returns a TherapyTaskIntegrator instance."""
    integrator = create_therapy_integrator()
    assert isinstance(integrator, TherapyTaskIntegrator)


# ---------------------------------------------------------------------------
# TherapyStateHead (forward pass)
# ---------------------------------------------------------------------------

@pytest.fixture
def therapy_head():
    from ml.models.heads.therapy_state_head import TherapyStateHead
    return TherapyStateHead(
        eye_dim=4,
        motion_dim=256,
        temporal_dim=128,
        hidden_dim=64,
        in_channels_depth=256,
        in_channels_contrast=256,
        use_lstm=True,
        lstm_hidden_size=32,
        lstm_num_layers=1,
        use_depth_multi_scale=False,
        use_edge_aware=True,
    )


def test_therapy_state_head_forward(therapy_head):
    """TherapyStateHead forward returns expected keys and shapes."""
    B, H, W = 2, 14, 14
    eye = torch.randn(B, 4)
    motion_2d = torch.randn(B, 256)
    motion_4d = torch.randn(B, 256, H, W)
    depth_feat = torch.randn(B, 256, H, W)
    contrast_feat = torch.randn(B, 256, H, W)

    out = therapy_head(eye, motion_4d, depth_feat, contrast_feat)

    assert "fatigue_score" in out
    assert "blink_rate" in out
    assert "fixation_stability" in out
    assert "depth_map" in out
    assert "uncertainty" in out
    assert "zones" in out
    assert "contrast_map" in out
    assert "edge_map" in out

    assert out["fatigue_score"].shape == (B, 1)
    assert out["blink_rate"].shape == (B, 1)
    assert out["fixation_stability"].shape == (B, 1)
    assert out["depth_map"].shape == (B, H, W)
    assert out["uncertainty"].shape == (B, H, W)
    assert out["zones"].shape == (B, 3)
    assert out["contrast_map"].shape == (B, H, W)
    assert out["edge_map"].shape == (B, H, W)


def test_therapy_state_head_forward_motion_2d(therapy_head):
    """TherapyStateHead accepts 2D motion features (fatigue path)."""
    B, H, W = 2, 14, 14
    eye = torch.randn(B, 4)
    motion_2d = torch.randn(B, 128)  # temporal_dim=128
    depth_feat = torch.randn(B, 256, H, W)
    contrast_feat = torch.randn(B, 256, H, W)

    out = therapy_head(eye, motion_2d, depth_feat, contrast_feat)
    assert out["fatigue_score"].shape == (B, 1)
    assert out["depth_map"].shape == (B, H, W)
