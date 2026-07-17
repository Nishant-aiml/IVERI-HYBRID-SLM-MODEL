# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for CurriculumBuilder and CurriculumByteDataset."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from configs.phase8_config import Phase8Config
from data.pipeline.curriculum import CurriculumBuilder
from data.pipeline.dataloader import CurriculumByteDataset


def test_curriculum_manifest_build(tmp_path: Path) -> None:
    """Test building curriculum manifests from certified sources."""
    config = Phase8Config()
    certified_dir = tmp_path / "certified"
    curriculum_dir = tmp_path / "curriculum"

    # Create dummy certified folders
    (certified_dir / "tinystories").mkdir(parents=True)
    # Write a dummy train.jsonl
    (certified_dir / "tinystories" / "train.jsonl").write_text(
        json.dumps({"text": "Story 1"}) + "\n" + json.dumps({"text": "Story 2"}) + "\n",
        encoding="utf-8",
    )

    (certified_dir / "fineweb_edu").mkdir(parents=True)
    (certified_dir / "fineweb_edu" / "train.jsonl").write_text(
        json.dumps({"text": "Education text"}) + "\n",
        encoding="utf-8",
    )

    builder = CurriculumBuilder(config, certified_dir, curriculum_dir)
    weights = {"tinystories": 0.5, "fineweb_edu": 0.5, "missing_ds": 0.2}

    result = builder.build_phase(
        phase="test",
        weights=weights,
        dataset_specs={},
    )

    assert result.success
    assert "tinystories" in result.available_datasets
    assert "fineweb_edu" in result.available_datasets
    assert "missing_ds" in result.missing_datasets

    # Weights for available datasets should be renormalized (0.5 / (0.5+0.5) = 0.5)
    manifest = result.manifest
    assert manifest.weights["tinystories"] == 0.5
    assert manifest.weights["fineweb_edu"] == 0.5
    assert "missing_ds" not in manifest.weights

    # Verify manifest file exists
    assert result.manifest_path.exists()

    # Test loading via CurriculumByteDataset
    dataset = CurriculumByteDataset(result.manifest_path, seq_len=4)
    assert len(dataset) > 0
    assert isinstance(dataset.data, np.ndarray)
