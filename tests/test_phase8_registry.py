# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 8 Dynamic Dataset Registry."""

from __future__ import annotations

from pathlib import Path
import pytest

from configs.phase8_config import Phase8Config, DatasetBudget
from data.pipeline.registry import DatasetRegistry


def test_registry_discovery(tmp_path: Path) -> None:
    """Test registry discovers specs from temporary spec YAMLs."""
    specs_dir = tmp_path / "datasets"
    specs_dir.mkdir()

    # Create dummy specs
    dummy_spec_1 = """
name: test_dataset_1
hf_id: test/dataset-1
hf_revision: main
license: MIT
license_url: https://opensource.org/licenses/MIT
priority: 1
stage: pretrain
max_storage_gb: 1.5
curriculum_phases: [A, B]
enabled: true
"""
    dummy_spec_2 = """
name: test_dataset_2
hf_id: test/dataset-2
hf_revision: pinned-sha
license: Apache-2.0
license_url: https://opensource.org/licenses/Apache-2.0
priority: 2
stage: pretrain
max_storage_gb: 2.5
curriculum_phases: [B, C]
enabled: false
"""

    (specs_dir / "test1.yaml").write_text(dummy_spec_1, encoding="utf-8")
    (specs_dir / "test2.yaml").write_text(dummy_spec_2, encoding="utf-8")

    registry = DatasetRegistry(specs_dir)
    discovered = registry.discover()

    # test_dataset_2 is disabled, so only 1 discovered
    assert len(discovered) == 1
    assert "test_dataset_1" in discovered
    assert "test_dataset_2" not in discovered

    spec = registry.get("test_dataset_1")
    assert isinstance(spec, DatasetBudget)
    assert spec.hf_id == "test/dataset-1"
    assert spec.license == "MIT"
    assert spec.max_storage_gb == 1.5
    assert spec.curriculum_phases == ["A", "B"]


def test_registry_schema_validation(tmp_path: Path) -> None:
    """Test registry validates spec YAML files and reports schema errors."""
    specs_dir = tmp_path / "datasets"
    specs_dir.mkdir()

    bad_spec = """
hf_id: test/bad
priority: not-an-integer
"""
    (specs_dir / "bad.yaml").write_text(bad_spec, encoding="utf-8")

    registry = DatasetRegistry(specs_dir)
    errors = registry.validate_all()

    assert "bad" in errors
    assert len(errors["bad"]) > 0
    assert any("name" in e for e in errors["bad"])
    assert any("license" in e for e in errors["bad"])
    assert any("priority" in e for e in errors["bad"])
