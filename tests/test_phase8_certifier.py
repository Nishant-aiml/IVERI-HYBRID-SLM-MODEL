# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the 24-stage certification pipeline."""

from __future__ import annotations

from pathlib import Path
import pytest

from configs.phase8_config import Phase8Config
from data.pipeline.certification import CertificationPipeline


def test_certification_pipeline_run(tmp_path: Path) -> None:
    """Test that all 24 stages of the pipeline run on a mock dataset."""
    config = Phase8Config(
        min_doc_chars=5,
        max_doc_chars=1000,
        offline_mode=True,
    )

    docs = [
        {"text": "Hello world, this is a clean English document."},
        {"text": "Hello world, this is a clean English document."},  # exact duplicate
        {"text": ""},  # empty
        {"text": "Short"},  # too short
        {"text": "HTML <p>tags</p> should be stripped from this text."},
    ]

    pipeline = CertificationPipeline(config)
    result = pipeline.certify(
        docs=docs,
        dataset_name="test_certify",
        output_dir=tmp_path / "certified",
    )

    assert result.success
    assert result.input_count == len(docs)
    # The duplicate, empty and too short docs should be filtered
    assert result.certified_count > 0
    assert result.certified_count < len(docs)
    assert result.total_removed > 0

    # Verify certified output directory has train.jsonl
    train_file = tmp_path / "certified" / "test_certify" / "train.jsonl"
    assert train_file.exists()
