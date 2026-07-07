# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.1B database integrity, transaction, and publication-source tests."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from research.experiment_registry import ExperimentRegistry
from research.publication_manager import PublicationManager
from research.registry_integrity import RegistryIntegrityError, validate_schema
from tests.provenance_helpers import CONFIG_HASH, GIT_SHA, seed_measured_experiment


@pytest.fixture
def registry() -> ExperimentRegistry:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    reg = ExperimentRegistry(db_path=db_path)
    yield reg
    Path(db_path).unlink(missing_ok=True)


def test_schema_validation_on_init(registry: ExperimentRegistry) -> None:
    conn = registry._get_connection()
    try:
        validate_schema(conn)
    finally:
        conn.close()


def test_duplicate_run_uuid_blocked(registry: ExperimentRegistry) -> None:
    registry.register_experiment(
        "exp_dup", "p", "H1", CONFIG_HASH, GIT_SHA, "main", 42, []
    )
    with pytest.raises(RegistryIntegrityError, match="Duplicate Run UUID"):
        registry.register_experiment(
            "exp_dup", "p2", "H2", CONFIG_HASH, GIT_SHA, "main", 43, []
        )


def test_failed_cannot_become_completed(registry: ExperimentRegistry) -> None:
    registry.register_experiment(
        "exp_fail", "p", "H1", CONFIG_HASH, GIT_SHA, "main", 42, []
    )
    registry.update_experiment_status("exp_fail", "RUNNING")
    registry.update_experiment_status("exp_fail", "FAILED")
    with pytest.raises(RegistryIntegrityError, match="FAILED cannot become COMPLETED"):
        registry.update_experiment_status("exp_fail", "COMPLETED")


def test_pending_cannot_become_completed(registry: ExperimentRegistry) -> None:
    registry.register_experiment(
        "exp_pending", "p", "H1", CONFIG_HASH, GIT_SHA, "main", 42, []
    )
    with pytest.raises(RegistryIntegrityError, match="PENDING cannot become COMPLETED"):
        registry.update_experiment_status("exp_pending", "COMPLETED")


def test_pending_to_running_to_completed_allowed(registry: ExperimentRegistry) -> None:
    registry.register_experiment(
        "exp_ok", "p", "H1", CONFIG_HASH, GIT_SHA, "main", 42, []
    )
    registry.update_experiment_status("exp_ok", "RUNNING")
    registry.update_experiment_status("exp_ok", "COMPLETED")
    rows = registry.get_experiments_by_tag("")
    assert rows[0]["status"] == "COMPLETED"


def test_mock_metrics_cannot_overwrite_measured(registry: ExperimentRegistry) -> None:
    seed_measured_experiment(registry, "exp_metric_guard", metrics_steps=[(10, 1.0, 0.9, 3.0)])
    with pytest.raises(RegistryIntegrityError, match="Metric overwrite blocked"):
        registry.log_metrics(
            "exp_metric_guard",
            step=10,
            train_loss=9.9,
            val_loss=9.8,
            perplexity=99.0,
            accuracy=0.1,
            provenance_label="SYNTHETIC",
        )


def test_metric_requires_existing_experiment(registry: ExperimentRegistry) -> None:
    with pytest.raises(RegistryIntegrityError, match="does not exist"):
        registry.log_metrics(
            "missing_exp", 1, 1.0, 1.0, 2.0, 0.5, provenance_label="MEASURED"
        )


def test_metric_row_unique_per_experiment_step(registry: ExperimentRegistry) -> None:
    seed_measured_experiment(registry, "exp_unique", metrics_steps=[(5, 1.0, 0.9, 3.0)])
    with pytest.raises(RegistryIntegrityError, match="overwrite blocked"):
        registry.log_metrics(
            "exp_unique", 5, 1.1, 1.0, 3.1, 0.5, provenance_label="UNKNOWN"
        )


def test_benchmark_measured_overwrite_guard(registry: ExperimentRegistry) -> None:
    seed_measured_experiment(registry, "exp_bench", with_benchmark=True)
    with pytest.raises(RegistryIntegrityError, match="Benchmark overwrite blocked"):
        registry.log_benchmark_run(
            run_id="bench_exp_bench",
            experiment_id="exp_bench",
            benchmark_id="HumanEval",
            step=100,
            score=0.01,
            provenance_label="SYNTHETIC",
        )


def test_write_audit_trail_populated(registry: ExperimentRegistry) -> None:
    before = registry.count_write_audit_entries()
    registry.register_experiment(
        "exp_audit", "p", "H1", CONFIG_HASH, GIT_SHA, "main", 42, []
    )
    registry.update_experiment_status("exp_audit", "RUNNING")
    after = registry.count_write_audit_entries()
    assert after >= before + 2


def test_foreign_keys_enforced(registry: ExperimentRegistry) -> None:
    registry.register_benchmark(
        benchmark_id="HEval",
        name="HumanEval",
        version="v1",
        source="HF",
        dataset_revision="main",
        prompt_suite_version="3A-v1",
        hash_sha256="abc123",
        num_prompts=1,
        evaluation_parameters={},
    )
    with pytest.raises(RegistryIntegrityError, match="does not exist"):
        registry.log_benchmark_run(
            "run_x", "missing_exp", "HEval", 1, 0.5, provenance_label="MEASURED"
        )


def test_benchmark_registry_generated_from_db_only(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    reg = ExperimentRegistry(db_path=str(db))
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="benchmark_registry table is empty"):
        pub.generate_benchmark_registry()

    seed_measured_experiment(reg, "exp_pub", with_benchmark=True)
    pub.generate_benchmark_registry()
    content = (tmp_path / "integrity" / "Benchmark_Registry.md").read_text(encoding="utf-8")
    assert "experiments.db" in content
    assert "4a7b5d2e3f8a9c1d0b6e" not in content


def test_reports_use_measured_db_rows_only(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    reg = ExperimentRegistry(db_path=str(db))
    seed_measured_experiment(reg, "exp_report", with_benchmark=True)
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path))
    pub.compile_reports_from_db(
        campaign_id="CAMP_TEST",
        git_sha=GIT_SHA,
        dataset_manifest_hash="d" * 64,
        pub_manifest_hash="e" * 64,
    )
    training_report = (tmp_path / "experiments" / "Training_Report.md").read_text(encoding="utf-8")
    assert "generated strictly from measured database values" in training_report
    assert "exp_report" in training_report
    assert "0.850000" not in training_report  # no hardcoded benchmark score injection


def test_metrics_reference_single_experiment(registry: ExperimentRegistry) -> None:
    seed_measured_experiment(registry, "exp_ref_a")
    seed_measured_experiment(registry, "exp_ref_b")
    conn = sqlite3.connect(registry.db_path)
    try:
        orphan = conn.execute(
            """
            SELECT COUNT(*) FROM metrics m
            LEFT JOIN experiments e ON e.experiment_id = m.experiment_id
            WHERE e.experiment_id IS NULL
            """
        ).fetchone()[0]
        assert orphan == 0
        multi = conn.execute(
            """
            SELECT experiment_id, step, COUNT(*) AS c
            FROM metrics
            GROUP BY experiment_id, step
            HAVING c > 1
            """
        ).fetchall()
        assert multi == []
    finally:
        conn.close()
