# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable streaming downloads with budget stops."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from configs.phase8_config import Phase8Config, DatasetBudget
from data.pipeline.streaming import StreamingDownloader, OfflineModeError


def test_offline_mode_error(tmp_path: Path) -> None:
    """Test OfflineModeError raised when dataset cache and staging are empty."""
    config = Phase8Config(
        offline_mode=True,
        cache_dir=str(tmp_path / "cache"),
        raw_data_dir=str(tmp_path / "raw"),
    )
    spec = DatasetBudget(
        name="not_cached",
        hf_id="test",
        hf_revision="main",
        hf_config=None,
        license="MIT",
        license_url="",
        source_url="",
        text_field="text",
        priority=1,
        stage="pretrain",
        max_storage_gb=1.0,
        curriculum_phases=["A"],
    )

    downloader = StreamingDownloader(config)
    output_dir = tmp_path / "raw" / "not_cached"

    with pytest.raises(OfflineModeError) as exc_info:
        downloader.download_to_budget(
            spec=spec,
            output_dir=output_dir,
            max_storage_gb=1.0,
        )
    assert "offline mode is active" in str(exc_info.value).lower()


def test_download_local_with_budget(tmp_path: Path) -> None:
    """Test that local downloader streams and stops when budget is reached."""
    # Create local source mock data
    raw_dir = tmp_path / "raw"
    source_dir = raw_dir / "local_test"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "local_test.jsonl"

    # Write 10 large documents
    docs = [{"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 100} for _ in range(10)]
    with open(source_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    config = Phase8Config(
        raw_data_dir=str(raw_dir),
        offline_mode=False,
    )
    # Budget cap at 1 KB (will fit only 1 document)
    spec = DatasetBudget(
        name="local_test",
        hf_id="",
        hf_revision="main",
        hf_config=None,
        license="MIT",
        license_url="",
        source_url="",
        text_field="text",
        priority=1,
        stage="pretrain",
        max_storage_gb=0.000001, # ~1 KB
        curriculum_phases=["A"],
        source_type="local",
    )

    downloader = StreamingDownloader(config)
    output_dir = tmp_path / "staging" / "local_test"

    result = downloader.download_to_budget(
        spec=spec,
        output_dir=output_dir,
        max_storage_gb=0.000001, # ~1 KB
        source_type="local",
    )

    # Should hit budget and stop early
    assert result.budget_hit
    assert result.num_docs < 10
    assert result.num_docs > 0
    assert result.output_path.exists()
