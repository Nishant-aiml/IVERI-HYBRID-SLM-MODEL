# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Document-level provenance and audit-trail system for dataset samples.

Records the lineage, processing steps, license, and intermediate statistics
of every document/sample through the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "3.0.0"


@dataclass(frozen=False, slots=True)
class ProcessingStep:
    """A single transformation or filtering step applied to a document."""

    step_name: str
    timestamp: str
    parameters: dict[str, Any]
    result_count: int
    removed_count: int


@dataclass(frozen=False, slots=True)
class ProvenanceRecord:
    """lineage metadata for a document/sample."""

    source_dataset: str
    document_hash: str  # SHA-256 of the text
    license: str
    download_time: str
    pipeline_version: str
    processing_steps: list[ProcessingStep] = field(default_factory=list)
    stage: str = "unknown"
    url: str | None = None
    byte_count: int = 0
    is_pii_cleaned: bool = False
    language: str | None = None
    quality_score: float | None = None


class ProvenanceTracker:
    """Helpers to construct, edit, and serialize ProvenanceRecord objects."""

    def __init__(self, pipeline_version: str = PIPELINE_VERSION) -> None:
        self.pipeline_version = pipeline_version

    def compute_document_hash(self, text: str) -> str:
        """Compute the SHA-256 hash of a string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def create_record(
        self,
        text: str,
        source_dataset: str,
        license: str,
        stage: str,
        url: str | None = None,
        language: str | None = None,
    ) -> ProvenanceRecord:
        """Create a fresh lineage record for a document."""
        doc_hash = self.compute_document_hash(text)
        return ProvenanceRecord(
            source_dataset=source_dataset,
            document_hash=doc_hash,
            license=license,
            download_time=datetime.now().isoformat(),
            pipeline_version=self.pipeline_version,
            processing_steps=[],
            stage=str(stage),
            url=url,
            byte_count=len(text.encode("utf-8")),
            is_pii_cleaned=False,
            language=language,
            quality_score=None,
        )

    def add_step(
        self,
        record: ProvenanceRecord,
        step_name: str,
        parameters: dict[str, Any],
        result_count: int = 1,
        removed_count: int = 0,
    ) -> ProvenanceRecord:
        """Append a processing step history to a record."""
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            result_count=result_count,
            removed_count=removed_count,
        )
        record.processing_steps.append(step)
        return record

    def to_dict(self, record: ProvenanceRecord) -> dict[str, Any]:
        """Convert a ProvenanceRecord to a serialize-ready dictionary."""
        steps = []
        for step in record.processing_steps:
            steps.append(
                {
                    "step_name": step.step_name,
                    "timestamp": step.timestamp,
                    "parameters": step.parameters,
                    "result_count": step.result_count,
                    "removed_count": step.removed_count,
                }
            )
        return {
            "source_dataset": record.source_dataset,
            "document_hash": record.document_hash,
            "license": record.license,
            "download_time": record.download_time,
            "pipeline_version": record.pipeline_version,
            "processing_steps": steps,
            "stage": record.stage,
            "url": record.url,
            "byte_count": record.byte_count,
            "is_pii_cleaned": record.is_pii_cleaned,
            "language": record.language,
            "quality_score": record.quality_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        """Construct a ProvenanceRecord object from a dictionary."""
        steps_raw = d.get("processing_steps", [])
        steps = []
        for s in steps_raw:
            steps.append(
                ProcessingStep(
                    step_name=s["step_name"],
                    timestamp=s["timestamp"],
                    parameters=s["parameters"],
                    result_count=s["result_count"],
                    removed_count=s["removed_count"],
                )
            )

        return ProvenanceRecord(
            source_dataset=d["source_dataset"],
            document_hash=d["document_hash"],
            license=d["license"],
            download_time=d["download_time"],
            pipeline_version=d["pipeline_version"],
            processing_steps=steps,
            stage=d.get("stage", "unknown"),
            url=d.get("url"),
            byte_count=d.get("byte_count", 0),
            is_pii_cleaned=d.get("is_pii_cleaned", False),
            language=d.get("language"),
            quality_score=d.get("quality_score"),
        )

    def save_batch(self, records: list[ProvenanceRecord], output_path: Path) -> None:
        """Save a batch of records to a JSON Lines (JSONL) file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(self.to_dict(r)) + "\n")
        logger.info(f"Wrote {len(records)} provenance records to {output_path}")

    def load_batch(self, path: Path) -> list[ProvenanceRecord]:
        """Load a batch of records from a JSON Lines (JSONL) file."""
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(self.from_dict(json.loads(line)))
        return records
