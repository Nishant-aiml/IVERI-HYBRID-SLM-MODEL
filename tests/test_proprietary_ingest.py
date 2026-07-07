# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for Stage 3B proprietary ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.pipeline.proprietary_ingest import (
    ProprietaryIngestError,
    count_proprietary_records,
    ingest_stage3b,
    validate_proprietary_record,
)


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_validate_proprietary_record_qa_format() -> None:
    rec = validate_proprietary_record(
        {
            "id": "qa-001",
            "license": "PROPRIETARY",
            "language": "en",
            "question": "What is a binary search tree?",
            "answer": "A BST orders nodes so left < parent < right.",
        },
        source="placement_qa",
    )
    assert "binary search" in rec.text
    assert rec.source == "placement_qa"


def test_validate_rejects_non_proprietary_license() -> None:
    with pytest.raises(ProprietaryIngestError, match="PROPRIETARY"):
        validate_proprietary_record(
            {"id": "x", "license": "MIT", "content": "A" * 30},
            source="university_papers",
        )


def test_ingest_stage3b_writes_manifest(tmp_path: Path) -> None:
    proprietary = tmp_path / "proprietary"
    _write_record(
        proprietary / "university_papers" / "au_2024.json",
        {
            "id": "paper-1",
            "license": "PROPRIETARY",
            "language": "en",
            "content": "Explain time complexity of merge sort with a detailed example.",
        },
    )
    _write_record(
        proprietary / "placement_qa" / "ds.json",
        {
            "id": "place-1",
            "license": "PROPRIETARY",
            "language": "en",
            "question": "Implement LRU cache?",
            "answer": "Use a hash map plus doubly linked list for O(1) operations.",
        },
    )

    out = tmp_path / "processed"
    report = ingest_stage3b(proprietary_dir=proprietary, output_dir=out, seed=7)
    assert report.record_count == 2
    assert (out / "manifest.json").exists()
    assert (out / "stage3b_train.json").exists()
    counts = count_proprietary_records(proprietary)
    assert sum(counts.values()) == 2


def test_ingest_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(ProprietaryIngestError, match="No proprietary records"):
        ingest_stage3b(proprietary_dir=tmp_path / "empty", output_dir=tmp_path / "out")
