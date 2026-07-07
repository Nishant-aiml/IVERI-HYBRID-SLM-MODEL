# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for Phase 3.6 Experimental Campaign orchestration.

Verifies SQLite experiment database schemas, scheduler dependency resolution,
failure replays, golden check points, regression detectors, dashboards,
and paper artifact manifests.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from research.experiment_registry import ExperimentRegistry
from research.experiment_scheduler import ExperimentScheduler
from research.compare_runs import RunComparator
from research.regression_detector import RegressionDetector
from research.golden import GoldenCheckpointManager
from research.failure_replay import FailureReplayEngine
from research.artifacts_graph import ArtifactsGraphManager
from research.hypothesis import ResearchHypothesisEngine
from research.scorecard import ResearchScorecard
from research.dashboard import ResearchDashboard
from research.paper_artifact_generator import PaperArtifactGenerator


# Helper dummy model matching interfaces
class DummyModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(256, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 256)
        self.param = nn.Parameter(torch.ones(1, hidden_dim))

    def forward(self, x, **kwargs):
        # shape: (B, S) -> (B, S, 256)
        embeds = self.embedding(x)
        logits = self.lm_head(embeds)
        return {"logits": logits, "loss": logits.mean()}


# ── Test 1-5: SQLite Experiment Registry ──────────────────────────────
def test_registry_initialization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        assert db.exists()


def test_registry_register_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment(
            experiment_id="exp_01",
            purpose="test baseline",
            hypothesis="H1",
            config_hash="h123",
            git_sha="commit_sha",
            git_branch="main",
            random_seed=42,
            tags=["baseline", "paper"],
        )
        runs = reg.get_experiments_by_tag("baseline")
        assert len(runs) == 1
        assert runs[0]["experiment_id"] == "exp_01"


def test_registry_status_update():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])
        reg.update_experiment_status("exp_01", "RUNNING")
        reg.update_experiment_status("exp_01", "SUCCESS")
        runs = reg.get_experiments_by_tag("")
        assert runs[0]["status"] == "SUCCESS"


def test_registry_metrics_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [], status="COMPLETED")
        reg.log_metrics("exp_01", step=1, train_loss=2.5, val_loss=2.6, perplexity=13.4, accuracy=0.82)
        
        conn = reg._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT val_loss FROM metrics WHERE experiment_id = ?", ("exp_01",))
            row = cursor.fetchone()
            assert row[0] == 2.6
        finally:
            conn.close()


def test_registry_hardware_and_assets():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [], status="COMPLETED")
        reg.register_paper_asset("fig_01", "exp_01", "figure", "Caption test", "fig:label", "fig.png")
        
        conn = reg._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT type FROM paper_assets WHERE asset_id = ?", ("fig_01",))
            row = cursor.fetchone()
            assert row[0] == "figure"
        finally:
            conn.close()


# ── Test 6-10: Experiment Scheduler ──────────────────────────────────
def test_scheduler_add_task():
    sched = ExperimentScheduler()
    sched.add_task("task_01", lambda: True)
    assert "task_01" in sched.tasks


def test_scheduler_dependencies_sorting():
    sched = ExperimentScheduler()
    sched.add_task("task_01", lambda: True, depends_on=["task_02"], priority=1)
    sched.add_task("task_02", lambda: True, priority=2)
    order = sched.resolve_topological_order()
    assert order == ["task_02", "task_01"]


def test_scheduler_priority_queue():
    sched = ExperimentScheduler()
    sched.add_task("task_01", lambda: True, priority=1)
    sched.add_task("task_02", lambda: True, priority=10)
    order = sched.resolve_topological_order()
    assert order == ["task_02", "task_01"]


def test_scheduler_interruption_recovery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("task_01", "purpose", "H1", "h", "g", "b", 42, [], status="SUCCESS")
        
        timeline = Path(tmp_dir) / "timeline.md"
        sched = ExperimentScheduler(registry=reg, timeline_path=str(timeline))
        
        executed = False
        def trigger():
            nonlocal executed
            executed = True

        sched.add_task("task_01", trigger)
        results = sched.execute_campaign()
        assert results["task_01"] == "SKIPPED"
        assert not executed


def test_scheduler_log_event():
    with tempfile.TemporaryDirectory() as tmp_dir:
        timeline = Path(tmp_dir) / "timeline.md"
        sched = ExperimentScheduler(timeline_path=str(timeline))
        sched.log_event("Campaign initiated")
        assert timeline.exists()
        content = timeline.read_text(encoding="utf-8")
        assert "Campaign initiated" in content


# ── Test 11-15: Run Comparator & Diff ────────────────────────────────
def test_run_comparator_delta_calculation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_a", "purpose", "H1", "h", "g", "b", 42, [], status="COMPLETED")
        reg.register_experiment("exp_b", "purpose", "H1", "h", "g", "b", 43, [], status="COMPLETED")
        reg.log_metrics("exp_a", step=1, train_loss=2.0, val_loss=2.0, perplexity=7.3, accuracy=0.8)
        reg.log_metrics("exp_a", step=2, train_loss=1.8, val_loss=1.8, perplexity=6.0, accuracy=0.82)
        reg.log_metrics("exp_b", step=1, train_loss=1.6, val_loss=1.6, perplexity=4.9, accuracy=0.85)
        reg.log_metrics("exp_b", step=2, train_loss=1.4, val_loss=1.4, perplexity=4.0, accuracy=0.88)

        comparator = RunComparator(registry=reg)
        res = comparator.compare_two_runs("exp_a", "exp_b")
        assert res["status"] == "SUCCESS"
        assert res["deltas"]["val_loss_absolute"] < 0.0
        assert res["statistics"]["cohens_d"] != 0.0


def test_run_comparator_insufficient_data():
    reg = ExperimentRegistry()
    comparator = RunComparator(registry=reg)
    res = comparator.compare_two_runs("nonexistent_a", "nonexistent_b")
    assert res["status"] == "INSUFFICIENT_DATA"


def test_run_comparator_insufficient_steps():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_a", "purpose", "H1", "h", "g", "b", 42, [], status="COMPLETED")
        reg.register_experiment("exp_b", "purpose", "H1", "h", "g", "b", 43, [], status="COMPLETED")
        reg.log_metrics("exp_a", step=1, train_loss=2.0, val_loss=2.0, perplexity=7.3, accuracy=0.8)
        reg.log_metrics("exp_b", step=1, train_loss=1.6, val_loss=1.6, perplexity=4.9, accuracy=0.85)

        comparator = RunComparator(registry=reg)
        res = comparator.compare_two_runs("exp_a", "exp_b")
        assert res["status"] == "INSUFFICIENT_STEPS"


# ── Test 16-18: Regression Detector ───────────────────────────────────
def test_regression_detector_info():
    gold = {"loss": 1.5, "perplexity": 4.5}
    new_metrics = {"loss": 1.51, "perplexity": 4.52} # slight increase < 2%
    det = RegressionDetector()
    res = det.check_for_regression(new_metrics, gold)
    assert res["highest_severity"] == "INFO"


def test_regression_detector_warning_critical():
    gold = {"loss": 1.5}
    new_warn = {"loss": 1.55} # increase between 2% and 5%
    new_crit = {"loss": 1.62} # increase between 5% and 10%
    det = RegressionDetector()
    assert det.check_for_regression(new_warn, gold)["highest_severity"] == "WARNING"
    assert det.check_for_regression(new_crit, gold)["highest_severity"] == "CRITICAL"


def test_regression_detector_fatal():
    gold = {"loss": 1.5, "humaneval_pass_rate": 0.8}
    new_fatal_loss = {"loss": 1.8} # increase > 10%
    new_fatal_accuracy = {"loss": 1.5, "humaneval_pass_rate": 0.5} # drop > 25%
    det = RegressionDetector()
    assert det.check_for_regression(new_fatal_loss, gold)["highest_severity"] == "FATAL"
    assert det.check_for_regression(new_fatal_accuracy, gold)["highest_severity"] == "FATAL"


# ── Test 19-21: Golden Checkpoint Manager ─────────────────────────────
def test_golden_checkpoint_set_get():
    config = IVERIConfig()
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])
        reg.register_checkpoint("ckpt_01", "exp_01", step=10, path="ckpt.pt", chk_hash="h123", parameters_count=1000)

        golden_mgr = GoldenCheckpointManager(config, registry=reg)
        golden_mgr.set_golden("ckpt_01")
        golden = golden_mgr.get_golden()
        assert golden is not None
        assert golden["checkpoint_id"] == "ckpt_01"


def test_golden_checkpoint_compare():
    config = IVERIConfig()
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])
        reg.register_checkpoint("ckpt_01", "exp_01", step=10, path="ckpt.pt", chk_hash="h123", parameters_count=1000)
        reg.log_metrics("exp_01", step=10, train_loss=1.5, val_loss=1.5, perplexity=4.5, accuracy=0.8)
        
        golden_mgr = GoldenCheckpointManager(config, registry=reg)
        golden_mgr.set_golden("ckpt_01")

        comparison = golden_mgr.compare_to_golden({"loss": 1.51, "perplexity": 4.52, "accuracy": 0.8})
        assert comparison["highest_severity"] == "INFO"


# ── Test 22-25: Failure Replay Engine ─────────────────────────────────
def test_failure_replay_serialization():
    config = IVERIConfig()
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])

        inputs = torch.randint(0, 256, (1, 8))
        error = RuntimeError("Fake OOM simulated")
        
        replay = FailureReplayEngine(registry=reg)
        file_path = replay.serialize_failure_payload(
            experiment_id="exp_01",
            step=10,
            failure_type="OOM",
            error=error,
            input_tensor=inputs,
            config=config,
            payload_dir=tmp_dir
        )
        assert file_path.exists()


def test_failure_replay_execution():
    config = IVERIConfig()
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])

        inputs = torch.randint(0, 256, (1, 8))
        error = RuntimeError("Fake OOM simulated")
        
        replay = FailureReplayEngine(registry=reg)
        file_path = replay.serialize_failure_payload(
            experiment_id="exp_01",
            step=10,
            failure_type="OOM",
            error=error,
            input_tensor=inputs,
            config=config,
            payload_dir=tmp_dir
        )

        res = replay.replay_failure(model, file_path)
        assert not res["reproduced"]


# ── Test 26-28: Artifacts Dependency Graph ────────────────────────────
def test_artifacts_graph_verify():
    graph = ArtifactsGraphManager()
    with tempfile.TemporaryDirectory() as tmp_dir:
        f = Path(tmp_dir) / "file.txt"
        f.write_text("content", encoding="utf-8")
        
        graph.add_artifact("file", f)
        res = graph.verify_integrity()
        assert res["ok"]


def test_artifacts_graph_missing():
    graph = ArtifactsGraphManager()
    graph.add_artifact("missing", "nonexistent.txt")
    res = graph.verify_integrity()
    assert not res["ok"]
    assert "missing" in res["missing_files"]


# ── Test 29-30: Extended Hypothesis Engine ───────────────────────────
def test_hypothesis_supported():
    engine = ResearchHypothesisEngine()
    res = engine.evaluate_hypothesis("H1", delta_pct=0.05, p_value=0.01, num_seeds=5)
    assert res["outcome"] == "SUPPORTED"


def test_hypothesis_refuted():
    engine = ResearchHypothesisEngine()
    res = engine.evaluate_hypothesis("H1", delta_pct=-0.08, p_value=0.02, num_seeds=5)
    assert res["outcome"] == "REFUTED"


# ── Test 31-33: Scorecard & Dashboard ────────────────────────────────
def test_scorecard_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        
        sc_path = Path(tmp_dir) / "Research_Scorecard.md"
        sc = ResearchScorecard(registry=reg, output_path=str(sc_path))
        
        hyp_evals = [{"hypothesis_label": "H1", "description": "Routing", "null_hypothesis": "Null", "outcome": "SUPPORTED", "evidence": "p<0.05"}]
        path = sc.generate_scorecard(hyp_evals, {"Baseline Sweeps Run": True}, calibration_ece=0.04)
        assert path.exists()


def test_dashboard_rendering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])

        dash = ResearchDashboard(registry=reg)
        ascii_repr = dash.render_dashboard()
        assert "exp_01" in ascii_repr


# ── Test 34-35: Paper Artifacts Manifest ──────────────────────────────
def test_paper_artifact_generator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Path(tmp_dir) / "experiments.db"
        reg = ExperimentRegistry(db_path=str(db))
        reg.register_experiment("exp_01", "purpose", "H1", "h", "g", "b", 42, [])

        gen = PaperArtifactGenerator(registry=reg, output_dir=tmp_dir)
        ckpt_path = Path(tmp_dir) / "ckpt.pt"
        ckpt_path.write_text("weights", encoding="utf-8")
        reg.register_checkpoint("ckpt_01", "exp_01", step=10, path=str(ckpt_path), chk_hash="h123", parameters_count=1000)

        manifest = gen.generate_and_verify_all_assets("exp_01", "git_sha_abc", "ckpt_hash_xyz", 42)
        assert "Figure_1_loss_curves" in manifest["assets"]
