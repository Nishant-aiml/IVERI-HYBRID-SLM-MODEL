# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset lineage tracking."""

from __future__ import annotations

from pathlib import Path
import pytest

from data.pipeline.lineage import LineageTracker, DocumentLineage


def test_lineage_tracker_save_load(tmp_path: Path) -> None:
    """Test recording, querying, and serializing document lineage."""
    tracker = LineageTracker()

    entry_1 = DocumentLineage(
        doc_id="uuid-1",
        source_dataset="tinystories",
        source_split="train",
        source_file="file_0.jsonl",
        source_row=42,
        download_date="2026-07-11T00:00:00",
        hf_revision="main",
        certification_stages_passed=["utf8", "lang"],
        certification_stages_failed=[],
        quality_score=0.9,
        byte_entropy=5.4,
        pii_replacements=0,
        safety_flags=[],
        curriculum_phases=["A"],
        final_text_hash="hash-1",
    )

    entry_2 = DocumentLineage(
        doc_id="uuid-2",
        source_dataset="fineweb_edu",
        source_split="train",
        source_file="file_1.jsonl",
        source_row=99,
        download_date="2026-07-11T00:00:00",
        hf_revision="main",
        certification_stages_passed=["utf8"],
        certification_stages_failed=["quality"],
        quality_score=0.3,
        byte_entropy=4.1,
        pii_replacements=2,
        safety_flags=["explicit_adult"],
        curriculum_phases=["B"],
        final_text_hash="hash-2",
    )

    tracker.record(entry_1.doc_id, entry_1)
    tracker.record(entry_2.doc_id, entry_2)

    # Test queries
    ts_records = tracker.query_by_source("tinystories")
    assert len(ts_records) == 1
    assert ts_records[0].doc_id == "uuid-1"

    phase_b_records = tracker.query_by_curriculum("B")
    assert len(phase_b_records) == 1
    assert phase_b_records[0].doc_id == "uuid-2"

    # Test save and load
    jsonl_path = tmp_path / "lineage.jsonl"
    tracker.save(jsonl_path)
    assert jsonl_path.exists()

    new_tracker = LineageTracker()
    new_tracker.load(jsonl_path)
    assert len(new_tracker.records) == 2
    assert new_tracker.get("uuid-1") is not None
    assert new_tracker.get("uuid-1").quality_score == 0.9
