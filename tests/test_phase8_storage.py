# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 8 Storage Manager and dynamic budgeting."""

from __future__ import annotations

from pathlib import Path
import pytest

from configs.phase8_config import Phase8Config, DatasetBudget
from data.pipeline.storage import StorageManager, EvalContaminationError


def test_storage_budget_calculation() -> None:
    """Test dynamic budget calculation handles reserve fraction and min free space."""
    config = Phase8Config(
        min_free_gb=15.0,
        reserve_fraction=0.20,
        max_total_budget_gb=50.0,
    )
    # Add dummy spec to budgets
    spec = DatasetBudget(
        name="test_ds",
        hf_id="test",
        hf_revision="main",
        hf_config=None,
        license="MIT",
        license_url="",
        source_url="",
        text_field="text",
        priority=1,
        stage="pretrain",
        max_storage_gb=8.0,
        curriculum_phases=["A"],
    )
    config.dataset_budgets = {"test_ds": spec}

    manager = StorageManager(config)
    budget = manager.compute_budget()

    assert budget.usable_gb >= 0.0
    assert budget.reserved_gb >= 15.0
    # Usable gb is capped at max_total_budget_gb
    assert budget.usable_gb <= 50.0


def test_storage_isolation_guard(tmp_path: Path) -> None:
    """Test validate_path_isolation raises error on eval-only datasets under certified."""
    config = Phase8Config(
        certified_data_dir="certified",
    )
    spec = DatasetBudget(
        name="test_eval",
        hf_id="test",
        hf_revision="main",
        hf_config=None,
        license="MIT",
        license_url="",
        source_url="",
        text_field="text",
        priority=1,
        stage="eval_only",
        max_storage_gb=0.1,
        curriculum_phases=[],
    )
    config.dataset_budgets = {"test_eval": spec}

    manager = StorageManager(config, base_dir=tmp_path)

    # Pretrain dataset path (safe)
    safe_path = tmp_path / "certified" / "test_pretrain"
    manager.validate_path_isolation(safe_path, "test_pretrain")  # should not raise

    # Eval-only dataset under certified (contamination!)
    unsafe_path = tmp_path / "certified" / "test_eval"
    with pytest.raises(EvalContaminationError) as exc_info:
        manager.validate_path_isolation(unsafe_path, "test_eval")
    assert "contamination" in str(exc_info.value).lower()
