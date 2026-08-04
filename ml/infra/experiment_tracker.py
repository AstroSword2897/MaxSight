"""Experiment and run tracking for the MaxSight ML lifecycle.

Writes structured run records locally under runs/<run_id>/ and optionally
syncs to SageMaker Experiments when the SDK is available.

Each run captures:
  - Hyperparameters
  - Metrics (loss, mAP, latency) logged per step/epoch
  - Artefact paths (checkpoints, exports, reports)
  - Dataset provenance (gold index hash)
  - Git commit SHA

Usage
-----
from ml.infra.experiment_tracker import RunTracker

with RunTracker(run_id="t5_finetune_20260301", experiment="maxsight-detection") as run:
    run.log_params({"lr": 1e-4, "epochs": 30, "tier": "T5"})
    for epoch in range(30):
        run.log_metric("train_loss", 0.45 - epoch * 0.01, step=epoch)
    run.log_artefact(Path("checkpoints/best.pt"))
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO / "runs"


# ── Run record ────────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    run_id: str
    experiment: str = "default"
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"
    params: dict[str, Any] = field(default_factory=dict)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    artefacts: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    git_sha: str = ""
    gold_index_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "params": self.params,
            "metrics": self.metrics,
            "artefacts": self.artefacts,
            "tags": self.tags,
            "git_sha": self.git_sha,
            "gold_index_hash": self.gold_index_hash,
        }


# ── Tracker ───────────────────────────────────────────────────────────────────


class RunTracker:
    """Context-manager run tracker with local persistence and optional SM Experiments."""

    def __init__(
        self,
        run_id: str | None = None,
        experiment: str = "maxsight",
        runs_dir: Path = DEFAULT_RUNS_DIR,
        *,
        sm_experiment: bool = False,
        s3_client=None,
    ) -> None:
        self.run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.experiment = experiment
        self.runs_dir = Path(runs_dir)
        self._sm_experiment = sm_experiment
        self._s3_client = s3_client
        self._sm_run = None

        self.record = RunRecord(run_id=self.run_id, experiment=experiment)
        self._run_dir = self.runs_dir / experiment / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> RunTracker:
        self.record.started_at = _now()
        self.record.git_sha = _git_sha()
        if self._sm_experiment:
            self._start_sm_run()
        self._save()
        logger.info("Run started: %s / %s", self.experiment, self.run_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.record.status = "failed" if exc_type else "completed"
        self.record.finished_at = _now()
        self._save()
        if self._sm_run:
            self._end_sm_run()
        if self._s3_client and self.record.status == "completed":
            self._upload_run_dir()
        logger.info("Run %s: %s", self.record.status, self.run_id)

    # ── Logging API ───────────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        self.record.params.update(params)
        if self._sm_run:
            for k, v in params.items():
                _sm_log_param(self._sm_run, k, str(v))
        self._save()

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        entry: dict[str, Any] = {"name": name, "value": float(value), "ts": _now()}
        if step is not None:
            entry["step"] = step
        if epoch is not None:
            entry["epoch"] = epoch
        self.record.metrics.append(entry)
        if self._sm_run:
            _sm_log_metric(self._sm_run, name, value, step)
        # Append-only line to a streaming metrics file for live monitoring.
        metrics_file = self._run_dir / "metrics.jsonl"
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_artefact(self, path: Path, *, tag: str = "") -> None:
        entry = str(path)
        if tag:
            entry = f"{tag}:{entry}"
        self.record.artefacts.append(entry)
        self._save()

    def log_dataset_provenance(self, gold_index_path: Path) -> None:
        import hashlib

        if gold_index_path.exists():
            h = hashlib.sha256(gold_index_path.read_bytes()).hexdigest()[:16]
            self.record.gold_index_hash = h
            self._save()

    def set_tag(self, key: str, value: str) -> None:
        self.record.tags[key] = value
        self._save()

    # ── Summary helpers ───────────────────────────────────────────────────────

    def best_metric(self, name: str, mode: str = "min") -> float | None:
        vals = [e["value"] for e in self.record.metrics if e["name"] == name]
        if not vals:
            return None
        return min(vals) if mode == "min" else max(vals)

    def summary(self) -> dict[str, Any]:
        return self.record.to_dict()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save(self) -> None:
        out = self._run_dir / "run.json"
        out.write_text(json.dumps(self.record.to_dict(), indent=2), encoding="utf-8")

    def _upload_run_dir(self) -> None:
        s3_client = self._s3_client
        if s3_client is None:
            return
        try:
            s3_client.upload_run_artefacts(self._run_dir, self.run_id)
        except Exception as exc:
            logger.warning("Run upload failed: %s", exc)

    def _start_sm_run(self) -> None:
        try:
            from sagemaker.experiments.run import Run  # type: ignore

            self._sm_run = Run(
                experiment_name=self.experiment,
                run_name=self.run_id,
            )
            self._sm_run.__enter__()
        except Exception as exc:
            logger.warning("SageMaker Experiments unavailable: %s", exc)
            self._sm_run = None

    def _end_sm_run(self) -> None:
        sm_run = self._sm_run
        if sm_run is None:
            return
        try:
            sm_run.__exit__(None, None, None)
        except Exception:
            pass


# ── Leaderboard (local run comparison) ───────────────────────────────────────


def load_all_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Load every run.json under runs_dir into a list, newest first."""
    records = []
    for p in sorted(runs_dir.rglob("run.json"), reverse=True):
        try:
            records.append(json.loads(p.read_text()))
        except Exception:
            pass
    return records


def leaderboard(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    metric: str = "val_map",
    mode: str = "max",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Return the top-N completed runs sorted by a metric."""

    def _best(record: dict[str, Any]) -> float:
        vals = [e["value"] for e in record.get("metrics", []) if e["name"] == metric]
        if not vals:
            return float("-inf") if mode == "max" else float("inf")
        return max(vals) if mode == "max" else min(vals)

    runs = [r for r in load_all_runs(runs_dir) if r.get("status") == "completed"]
    runs.sort(key=_best, reverse=(mode == "max"))
    return runs[:top_n]


# ── Convenience context manager ───────────────────────────────────────────────


@contextmanager
def track_run(run_id: str | None = None, experiment: str = "maxsight", **kwargs):
    """Shorthand context manager for quick usage."""
    with RunTracker(run_id=run_id, experiment=experiment, **kwargs) as run:
        yield run


# ── Helpers ────────────────────────────────────────────────────────────────────


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _git_sha() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _sm_log_param(run, key: str, value: str) -> None:
    try:
        run.log_parameter(key, value)
    except Exception:
        pass


def _sm_log_metric(run, name: str, value: float, step: int | None) -> None:
    try:
        run.log_metric(name, value, step=step)
    except Exception:
        pass
