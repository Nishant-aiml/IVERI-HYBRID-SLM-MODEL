# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 9 pretraining campaign orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from scripts.run_phase9_campaign import CampaignOrchestrator


def test_environment_lock(tmp_path: Path) -> None:
    """Test locking the environment and writing manifests."""
    orchestrator = CampaignOrchestrator(recipe_name="iveri_v1")
    
    # Redirect outputs to temp dir
    orchestrator.reports_dir = tmp_path / "reports"
    orchestrator.reports_dir.mkdir()
    orchestrator.failures_dir = orchestrator.reports_dir / "failures"
    orchestrator.failures_dir.mkdir()

    assert orchestrator.lock_environment()
    assert orchestrator.environment_locked

    # Check files exist
    assert (orchestrator.reports_dir / "campaign_manifest.json").exists()
    assert (orchestrator.reports_dir / "env_report.md").exists()
    assert (orchestrator.reports_dir / "hardware_report.md").exists()
    assert (orchestrator.reports_dir / "config_snapshot.json").exists()
    assert (orchestrator.reports_dir / "training_manifest.json").exists()


def test_model_init_validation(tmp_path: Path) -> None:
    """Test profiling model parameter count and sub-module activation checks."""
    orchestrator = CampaignOrchestrator(recipe_name="iveri_v1")
    orchestrator.reports_dir = tmp_path / "reports"
    orchestrator.reports_dir.mkdir()
    orchestrator.failures_dir = orchestrator.reports_dir / "failures"
    orchestrator.failures_dir.mkdir()

    # Pre-requisite
    orchestrator.environment_locked = True
    orchestrator.dataset_verified = True

    # Reduce model dimensions for fast CPU test
    orchestrator.base_config.model.hidden_dim = 64
    orchestrator.base_config.model.num_layers = 1
    orchestrator.base_config.model.num_experts = 2
    orchestrator.base_config.model.num_active_experts = 1
    orchestrator.base_config.model.titans_memory_dim = 16
    orchestrator.base_config.hardware.device = "cpu"

    assert orchestrator.validate_model_init()
    assert orchestrator.model_initialized

    # Verify reports
    assert (orchestrator.reports_dir / "model_init_report.json").exists()
    assert (orchestrator.reports_dir / "model_init_report.md").exists()


def test_stability_early_warnings_and_failure_db(tmp_path: Path) -> None:
    """Test generating early warning alerts and writing to the failure database."""
    orchestrator = CampaignOrchestrator(recipe_name="iveri_v1")
    orchestrator.reports_dir = tmp_path / "reports"
    orchestrator.reports_dir.mkdir()
    orchestrator.failures_dir = orchestrator.reports_dir / "failures"
    orchestrator.failures_dir.mkdir()

    # Check warnings
    warnings = orchestrator.check_stability_warnings(
        step=50,
        loss=16.5,
        grad_norm=12.4,
        expert_entropy=0.05,
    )
    assert len(warnings) == 3
    assert any("Exploding gradient" in w for w in warnings)
    assert any("entropy collapse" in w for w in warnings)

    # Check failure db logging
    fail_file = orchestrator.save_failure_db_log(
        step=120,
        error_msg="Injected NaN Loss error",
        stack_trace="Traceback: line 42...",
    )
    assert fail_file.exists()
    with open(fail_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["step"] == 120
    assert "NaN Loss" in data["error_message"]
