# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Document-level lineage (provenance) tracking for IVERI CORE datasets.

Allows every document to be traced back to its source, split, file, and row
including the certification filters it passed and any modifications.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentLineage:
    """Provenance and lineage metadata for a single certified document."""

    doc_id: str
    source_dataset: str
    source_split: str
    source_file: str
    source_row: int | None
    download_date: str
    hf_revision: str
    certification_stages_passed: list[str]
    certification_stages_failed: list[str]
    quality_score: float
    byte_entropy: float
    pii_replacements: int
    safety_flags: list[str]
    curriculum_phases: list[str]
    final_text_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return {
            "doc_id": self.doc_id,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_file": self.source_file,
            "source_row": self.source_row,
            "download_date": self.download_date,
            "hf_revision": self.hf_revision,
            "certification_stages_passed": self.certification_stages_passed,
            "certification_stages_failed": self.certification_stages_failed,
            "quality_score": round(self.quality_score, 4),
            "byte_entropy": round(self.byte_entropy, 4),
            "pii_replacements": self.pii_replacements,
            "safety_flags": self.safety_flags,
            "curriculum_phases": self.curriculum_phases,
            "final_text_hash": self.final_text_hash,
        }


class LineageTracker:
    """Tracks and queries document lineage across datasets."""

    def __init__(self) -> None:
        self.records: dict[str, DocumentLineage] = {}

    def record(self, doc_id: str, lineage: DocumentLineage) -> None:
        """Record a lineage entry."""
        self.records[doc_id] = lineage

    def get(self, doc_id: str) -> DocumentLineage | None:
        """Get lineage for a specific document ID."""
        return self.records.get(doc_id)

    def save(self, output_path: Path) -> None:
        """Save lineage registry as JSONL (one JSON dict per line)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for record in self.records.values():
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            logger.info("Saved %d lineage records to %s", len(self.records), output_path)
        except OSError as exc:
            logger.error("Failed to write lineage records to %s: %s", output_path, exc)

    def load(self, path: Path) -> None:
        """Load lineage registry from a JSONL file."""
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    doc_id = data["doc_id"]
                    self.records[doc_id] = DocumentLineage(**data)
            logger.info("Loaded %d lineage records from %s", len(self.records), path)
        except Exception as exc:
            logger.error("Failed to load lineage registry from %s: %s", path, exc)

    def query_by_source(self, source_dataset: str) -> list[DocumentLineage]:
        """Query records by source dataset."""
        return [r for r in self.records.values() if r.source_dataset == source_dataset]

    def query_by_curriculum(self, phase: str) -> list[DocumentLineage]:
        """Query records by curriculum phase."""
        return [r for r in self.records.values() if phase in r.curriculum_phases]

    def generate_lineage_report(self, output_path: Path) -> None:
        """Generate a text summary report of lineage metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            datasets: dict[str, int] = {}
            for r in self.records.values():
                datasets[r.source_dataset] = datasets.get(r.source_dataset, 0) + 1

            lines = [
                "# IVERI CORE — Phase 8 Dataset Lineage Report",
                "",
                f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Tracked Documents:** {len(self.records):,}",
                "",
                "## Distribution by Source Dataset",
                "",
                "| Dataset | Documents |",
                "|---|---|",
            ]
            for ds, count in sorted(datasets.items(), key=lambda x: -x[1]):
                lines.append(f"| {ds} | {count:,} |")

            output_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to write lineage report: %s", exc)
