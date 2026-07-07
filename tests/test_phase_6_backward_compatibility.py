# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Automated regression suite verifying backward compatibility for Phase 4 and Phase 5 operations."""

from __future__ import annotations

import tempfile
import sqlite3
from pathlib import Path

import pytest

from research.experiment_registry import ExperimentRegistry
from research.publication_manager import PublicationManager


@pytest.fixture
def temp_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    p = Path(db_path)
    if p.exists():
        p.unlink()


def test_phase_5_table_schema_backward_compatibility(temp_db: str) -> None:
    """Verifies that pre-existing schemas (experiments, metrics, hardware, checkpoints, failures) remain fully operational."""
    registry = ExperimentRegistry(db_path=temp_db)

    # 1. Write standard pretraining experiment entry
    registry.register_experiment(
        experiment_id="IVERI_Phase5_pretrain_Seed42",
        purpose="Pretraining baseline comparison",
        hypothesis="IVERI converges faster than Transformer",
        config_hash="conf_abc_123",
        git_sha="git_sha_abc",
        git_branch="main",
        random_seed=42,
        tags=["pretrain", "baseline"],
    )

    # 2. Write metric entries
    registry.log_metrics(
        experiment_id="IVERI_Phase5_pretrain_Seed42",
        step=100,
        train_loss=1.5,
        val_loss=1.6,
        perplexity=4.9,
        accuracy=0.45,
    )

    # 3. Log hardware telemetry
    registry.log_hardware(
        experiment_id="IVERI_Phase5_pretrain_Seed42",
        cpu=15.0,
        gpu=85.0,
        ram_mb=4096.0,
        vram_mb=2048.0,
        wattage=120.0,
        energy_j=12000.0,
        cost_usd=0.01,
    )

    # 4. Register a checkpoint
    registry.register_checkpoint(
        checkpoint_id="ckpt_step100",
        experiment_id="IVERI_Phase5_pretrain_Seed42",
        step=100,
        path="checkpoints/final.pt",
        chk_hash="sha256_checkpoint",
        parameters_count=10000000,
        is_golden=True,
    )

    # Validate database state
    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT train_loss FROM metrics WHERE experiment_id='IVERI_Phase5_pretrain_Seed42'")
        assert cursor.fetchone()[0] == 1.5

        cursor.execute("SELECT gpu_utilization FROM hardware WHERE experiment_id='IVERI_Phase5_pretrain_Seed42'")
        assert cursor.fetchone()[0] == 85.0

        cursor.execute("SELECT path FROM checkpoints WHERE checkpoint_id='ckpt_step100'")
        assert cursor.fetchone()[0] == "checkpoints/final.pt"
    finally:
        conn.close()


def test_publication_manager_backward_compatibility(temp_db: str) -> None:
    """Verifies that PublicationManager can still compile report artifacts on top of updated schemas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        registry = ExperimentRegistry(db_path=temp_db)
        exp_id = "IVERI_CAMPAIGN_VERIFY_EXP"
        registry.register_experiment(
            experiment_id=exp_id,
            purpose="Backward compatibility verification",
            hypothesis="Publication pipeline remains compatible",
            config_hash="conf_verify",
            git_sha="abc123",
            git_branch="main",
            random_seed=42,
            tags=["verify"],
            provenance_label="MEASURED",
            status="COMPLETED",
        )
        registry.log_metrics(
            experiment_id=exp_id,
            step=100,
            train_loss=1.0,
            val_loss=1.1,
            perplexity=3.0,
            accuracy=0.5,
            provenance_label="MEASURED",
        )
        pub = PublicationManager(registry=registry, output_dir=str(tmp_path))

        # Check standard report indexing files compile successfully
        pub.generate_final_report(campaign_id="IVERI_CAMPAIGN_VERIFY")
        final_report_file = tmp_path / "FINAL_REPORT.md"
        assert final_report_file.exists()

        content = final_report_file.read_text(encoding="utf-8")
        assert "Campaign ID:" in content
        assert "Protocol Version:** Phase6.3" in content or "Protocol Version:** Phase6.3-v2.0" in content
