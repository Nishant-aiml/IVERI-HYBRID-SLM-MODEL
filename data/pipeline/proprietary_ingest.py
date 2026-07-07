# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Stage 3B proprietary dataset ingestion, validation, and processing.

Loads JSON records from ``data/proprietary/{source}/``, validates schema,
applies PII cleaning, splits 90/5/5 for small datasets, and writes processed
artifacts to ``data/processed/stage3b/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.pipeline.pii_remover import PIIRemover
from data.pipeline.splitter import DatasetSplitter

logger = logging.getLogger(__name__)

STAGE3B_SOURCES: tuple[str, ...] = (
    "university_papers",
    "gate_questions",
    "placement_qa",
    "subject_explanations",
)

EXCLUDED_JSON_NAMES: frozenset[str] = frozenset({"SCHEMA.example.json"})


@dataclass(frozen=True)
class ProprietaryRecord:
    """Normalized proprietary training record."""

    id: str
    source: str
    license: str
    language: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProprietaryIngestReport:
    """Summary of a Stage 3B ingest run."""

    record_count: int
    sources: dict[str, int]
    output_dir: str
    manifest_path: str
    split_report: dict[str, Any]
    timestamp_utc: str


class ProprietaryIngestError(ValueError):
    """Raised when proprietary JSON fails validation."""


def _is_data_json(path: Path) -> bool:
    if path.name in EXCLUDED_JSON_NAMES:
        return False
    if path.name.startswith("."):
        return False
    return path.suffix.lower() == ".json"


def discover_raw_json_files(proprietary_dir: Path) -> list[Path]:
    """Return proprietary JSON files across all Stage 3B source folders."""
    files: list[Path] = []
    for source in STAGE3B_SOURCES:
        source_dir = proprietary_dir / source
        if not source_dir.is_dir():
            continue
        files.extend(p for p in source_dir.rglob("*.json") if _is_data_json(p))
    return sorted(files)


def count_proprietary_records(proprietary_dir: Path | str) -> dict[str, int]:
    """Count valid records per source without writing processed output."""
    root = Path(proprietary_dir)
    counts: dict[str, int] = {s: 0 for s in STAGE3B_SOURCES}
    for path in discover_raw_json_files(root):
        source = path.parent.name if path.parent.name in STAGE3B_SOURCES else "unknown"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for item in records:
            validate_proprietary_record(item, source=source)
            if source in counts:
                counts[source] += 1
    return counts


def validate_proprietary_record(data: dict[str, Any], *, source: str) -> ProprietaryRecord:
    """Validate a single proprietary JSON object and return normalized text."""
    if not isinstance(data, dict):
        raise ProprietaryIngestError(f"Record must be an object, got {type(data).__name__}")

    record_id = str(data.get("id", "")).strip()
    if not record_id:
        raise ProprietaryIngestError("Record missing required field: id")

    license_name = str(data.get("license", "")).strip().upper()
    if license_name != "PROPRIETARY":
        raise ProprietaryIngestError(
            f"Record {record_id}: license must be PROPRIETARY, got {license_name!r}"
        )

    language = str(data.get("language", "en")).strip().lower()
    if not language:
        raise ProprietaryIngestError(f"Record {record_id}: language must be non-empty")

    text = _extract_text(data)
    if len(text.strip()) < 20:
        raise ProprietaryIngestError(f"Record {record_id}: text too short (< 20 chars)")

    metadata = data.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise ProprietaryIngestError(f"Record {record_id}: metadata must be an object")

    return ProprietaryRecord(
        id=record_id,
        source=source,
        license=license_name,
        language=language,
        text=text.strip(),
        metadata=dict(metadata or {}),
    )


def _extract_text(data: dict[str, Any]) -> str:
    if "content" in data and data["content"]:
        return str(data["content"])
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if question and answer:
        return f"### Question:\n{question}\n\n### Answer:\n{answer}"
    if question:
        return question
    raise ProprietaryIngestError(
        f"Record {data.get('id', '?')}: require 'content' or 'question'+'answer'"
    )


def load_proprietary_records(proprietary_dir: Path | str) -> list[ProprietaryRecord]:
    """Load and validate all proprietary JSON records."""
    root = Path(proprietary_dir)
    records: list[ProprietaryRecord] = []
    for path in discover_raw_json_files(root):
        source = path.parent.name if path.parent.name in STAGE3B_SOURCES else "unknown"
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            records.append(validate_proprietary_record(item, source=source))
    return records


def ingest_stage3b(
    proprietary_dir: Path | str = "data/proprietary",
    output_dir: Path | str = "data/processed/stage3b",
    *,
    seed: int = 42,
    clean_pii: bool = True,
) -> ProprietaryIngestReport:
    """Validate, clean, split, and persist Stage 3B proprietary data."""
    root = Path(proprietary_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_proprietary_records(root)
    if not records:
        raise ProprietaryIngestError(
            "No proprietary records found. Add JSON files under "
            f"{root}/{{{','.join(STAGE3B_SOURCES)}}}/"
        )

    pii = PIIRemover() if clean_pii else None
    normalized: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {s: 0 for s in STAGE3B_SOURCES}

    for rec in records:
        text = rec.text
        if pii is not None:
            text = pii.remove(text)
        source_counts[rec.source] = source_counts.get(rec.source, 0) + 1
        normalized.append(
            {
                "id": rec.id,
                "source": rec.source,
                "license": rec.license,
                "language": rec.language,
                "text": text,
                "metadata": rec.metadata,
            }
        )

    splitter = DatasetSplitter(seed=seed)
    train, val, test = splitter.split_small_dataset(normalized)
    splitter.save_splits(train, val, test, out, "stage3b")
    split_report = asdict(splitter.generate_report(train, val, test, seed))

    manifest = {
        "dataset": "stage3b_proprietary",
        "record_count": len(normalized),
        "sources": source_counts,
        "seed": seed,
        "pii_cleaned": clean_pii,
        "split": split_report,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "Stage 3B ingest complete: %d records -> %s",
        len(normalized),
        manifest_path,
    )
    return ProprietaryIngestReport(
        record_count=len(normalized),
        sources=source_counts,
        output_dir=str(out),
        manifest_path=str(manifest_path),
        split_report=split_report,
        timestamp_utc=manifest["timestamp_utc"],
    )
