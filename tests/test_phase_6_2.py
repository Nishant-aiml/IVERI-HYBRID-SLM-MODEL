# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Automated test suite verifying Phase 6.2 Database schemas, Integrity checks, and card generation."""

from __future__ import annotations

import json
import tempfile
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import numpy as np
import torch

from research.experiment_registry import ExperimentRegistry
from research.benchmark_integrity import BenchmarkIntegrityFramework
from research.publication_manager import PublicationManager
from tests.provenance_helpers import CONFIG_HASH, GIT_SHA, seed_measured_experiment


@pytest.fixture
def temp_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Clean up
    p = Path(db_path)
    if p.exists():
        p.unlink()


def test_experiment_registry_schema_migrations(temp_db: str) -> None:
    """Verifies that the new schema tables are created and readable."""
    registry = ExperimentRegistry(db_path=temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        # Verify tables exist
        tables = [
            "benchmark_registry",
            "benchmark_runs",
            "benchmark_integrity",
            "benchmark_artifacts",
            "publication_runs",
            "release_manifests",
        ]
        for t in tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (t,),
            )
            assert cursor.fetchone() is not None, f"Table '{t}' is missing from schema!"
    finally:
        conn.close()


def test_experiment_registry_logging_helpers(temp_db: str) -> None:
    """Verifies write and read of relational integrity data."""
    registry = ExperimentRegistry(db_path=temp_db)
    registry.register_experiment(
        experiment_id="exp_1",
        purpose="test",
        hypothesis="H1",
        config_hash=CONFIG_HASH,
        git_sha=GIT_SHA,
        git_branch="main",
        random_seed=42,
        tags=["iveri"],
        provenance_label="MEASURED",
        status="COMPLETED",
    )

    # 1. Register benchmark
    registry.register_benchmark(
        benchmark_id="HEval",
        name="HumanEval",
        version="v1",
        source="HF",
        dataset_revision="main",
        prompt_suite_version="3A-v1",
        hash_sha256="abc123hash",
        num_prompts=17,
        evaluation_parameters={"temp": 0.8},
    )

    # 2. Log benchmark run
    registry.log_benchmark_run(
        run_id="run_1",
        experiment_id="exp_1",
        benchmark_id="HEval",
        step=1000,
        score=0.85,
        provenance_label="MEASURED",
    )

    # 3. Log benchmark integrity
    registry.log_benchmark_integrity(
        run_id="run_1",
        prompt_hash_ok=True,
        template_hash_ok=True,
        system_prompt_hash_ok=True,
        fewshot_hash_ok=True,
        generation_params_hash_ok=True,
        dataset_hash_ok=True,
        reproducibility_ok=True,
        integrity_report_path="reports/integrity.md",
    )

    # 4. Log benchmark artifact
    registry.log_benchmark_artifact(
        artifact_id="art_1",
        run_id="run_1",
        name="Benchmark_Registry.md",
        path="reports/Benchmark_Registry.md",
        hash_val="registry_hash",
    )

    # Verify database contents
    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM benchmark_runs WHERE run_id='run_1'")
        assert cursor.fetchone()[0] == 0.85

        cursor.execute("SELECT prompt_hash_ok FROM benchmark_integrity WHERE run_id='run_1'")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT hash FROM benchmark_artifacts WHERE artifact_id='art_1'")
        assert cursor.fetchone()[0] == "registry_hash"
    finally:
        conn.close()


def test_benchmark_integrity_hashing(temp_db: str) -> None:
    """Verifies file and pipeline asset hashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "pretrain.bin"
        test_file.write_bytes(b"hello_pretrain_data")

        framework = BenchmarkIntegrityFramework(data_dir=tmp_path, db_path=temp_db)

        # File hash check
        h = framework.compute_file_sha256(test_file)
        assert h == "4c54d693f6e0188823eb075075eb55925a555709d72b782b286c4c105e5be65c"

        # Lock checks
        locks = framework.lock_dataset_revisions()
        assert locks[test_file.as_posix()] == h

        # Pipeline asset checks
        prompts = [{"prompt_id": "p1", "instruction": "write binary search"}]
        eval_cfg = {"max_len": 512}
        gen_params = {"temp": 0.8}
        hashes = framework.hash_pipeline_assets(
            benchmark_name="HumanEval",
            prompts=prompts,
            template_func=lambda x: f"Prompt: {x}",
            system_prompt="You are coding assistant",
            evaluation_config=eval_cfg,
            few_shot=None,
            generation_params=gen_params,
        )
        assert "prompt_hash" in hashes
        assert "template_hash" in hashes
        assert "system_prompt_hash" in hashes


def test_env_locking_and_reproducibility(temp_db: str) -> None:
    """Verifies that environment locks and reproducibility audits run successfully."""
    framework = BenchmarkIntegrityFramework(db_path=temp_db)
    env = framework.get_env_info()
    assert "os" in env
    assert "python_version" in env
    assert "pytorch_version" in env

    # Verify audit comparison
    audit_res = framework.audit_reproducibility(
        run_config={"max_steps": 100},
        expected_env={"pytorch_version": torch.__version__},
    )
    assert audit_res["reproducibility_ok"] is True
    assert len(audit_res["mismatches"]) == 0


def test_publication_manager_cards_generation(temp_db: str) -> None:
    """Verifies that card, manifest, and certificate files compile correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        registry = ExperimentRegistry(db_path=temp_db)
        experiment_id = seed_measured_experiment(registry, "IVERI_Run001", with_benchmark=True)
        pub = PublicationManager(registry=registry, output_dir=str(tmp_path))

        # Invoke generators
        pub.generate_model_card(checkpoint_id="ckpt_IVERI_Run001")
        pub.generate_dataset_cards()
        pub.generate_benchmark_registry()
        pub.generate_release_manifest(
            experiment_id=experiment_id,
            release_id="rel_IVERI_Run001",
            checkpoint_path="checkpoints/final.pt",
            env_info={"git_sha": GIT_SHA, "git_branch": "main"},
        )
        pub.generate_phase_certificate(campaign_id="IVERI_CAMPAIGN_VERIFY")

        # Verify file presence
        assert (tmp_path / "cards/Model_Card.md").exists()
        assert (tmp_path / "cards/Dataset_Cards/FineWeb.md").exists()
        assert (tmp_path / "cards/Dataset_Cards/Wikipedia.md").exists()
        assert (tmp_path / "integrity/Benchmark_Registry.md").exists()
        assert (tmp_path / "integrity/benchmark_registry.json").exists()
        assert (tmp_path / "integrity/release_manifest.json").exists()
        assert (tmp_path / "reviewer/Phase_6_3_Certificate.md").exists()
