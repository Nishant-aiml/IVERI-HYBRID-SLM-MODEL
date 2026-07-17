# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace-style dataset card generator for IVERI CORE certified datasets.

Generates structured Markdown dataset cards with YAML frontmatter and full
certification pipeline statistics, mirroring the HuggingFace Hub card format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class DatasetCardMetadata:
    """All metadata fields required to generate a dataset card."""

    name: str
    version: str
    license: str
    license_url: str
    source_url: str
    created_at: str
    num_documents: int
    num_bytes: int
    avg_doc_length: float
    median_doc_length: float
    byte_entropy_mean: float
    dedup_removal_rate: float
    pii_removal_rate: float
    safety_removal_rate: float
    quality_removal_rate: float
    overall_retention_rate: float
    sha256_manifest_hash: str
    freeze_certificate_hash: str
    curriculum_phases: list[str] = field(default_factory=lambda: ["A"])
    certification_stages_passed: int = 24  # out of 24

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to plain dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "license_url": self.license_url,
            "source_url": self.source_url,
            "created_at": self.created_at,
            "num_documents": self.num_documents,
            "num_bytes": self.num_bytes,
            "avg_doc_length": round(self.avg_doc_length, 4),
            "median_doc_length": round(self.median_doc_length, 4),
            "byte_entropy_mean": round(self.byte_entropy_mean, 4),
            "dedup_removal_rate": round(self.dedup_removal_rate, 6),
            "pii_removal_rate": round(self.pii_removal_rate, 6),
            "safety_removal_rate": round(self.safety_removal_rate, 6),
            "quality_removal_rate": round(self.quality_removal_rate, 6),
            "overall_retention_rate": round(self.overall_retention_rate, 6),
            "sha256_manifest_hash": self.sha256_manifest_hash,
            "freeze_certificate_hash": self.freeze_certificate_hash,
            "curriculum_phases": self.curriculum_phases,
            "certification_stages_passed": self.certification_stages_passed,
        }

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


class DatasetCardGenerator:
    """Generates HuggingFace-style dataset cards in Markdown."""

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        metadata: DatasetCardMetadata,
        output_path: Path,
    ) -> str:
        """Generate Markdown dataset card, write to output_path, return markdown string.

        Format:
        - YAML frontmatter compatible with HuggingFace Hub
        - Sections: Overview, Dataset Details, Certification Pipeline, Statistics, License, Citation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = self._render_card(metadata)
        output_path.write_text(md, encoding="utf-8")

        # Also write JSON metadata alongside Markdown card
        json_path = output_path.with_name(f"{output_path.stem}.json")
        json_path.write_text(metadata.to_json(), encoding="utf-8")

        logger.info(f"Dataset card written to {output_path} and {json_path}")
        return md

    def generate_combined_card(
        self,
        cards: list[DatasetCardMetadata],
        output_path: Path,
    ) -> str:
        """Generate a combined card for all certified datasets."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON counterpart
        import json
        json_path = output_path.with_name("combined_cards.json")
        combined_data = {
            "generated_at": datetime.now().isoformat(),
            "cards": [c.to_dict() for c in cards]
        }
        json_path.write_text(json.dumps(combined_data, indent=2), encoding="utf-8")

        lines: list[str] = []
        lines += [
            "---",
            "dataset_name: iveri_certified_combined",
            "license: cc-by-4.0",
            "language: en",
            "task_categories: text-generation",
            "tags:",
            "  - iveri-certified",
            "  - phase-8",
            "  - combined",
            "---",
            "",
            "# IVERI Combined Dataset Card",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Certified Datasets",
            "",
            "| Dataset | Documents | Size (bytes) | Retention Rate | Stages Passed |",
            "|---|---|---|---|---|",
        ]
        for m in cards:
            lines.append(
                f"| {m.name} | {m.num_documents:,} | {m.num_bytes:,} "
                f"| {m.overall_retention_rate:.1%} | {m.certification_stages_passed}/24 |"
            )

        lines += [
            "",
            "---",
            "",
        ]

        for m in cards:
            lines.append(self._render_card(m))
            lines.append("")
            lines.append("---")
            lines.append("")

        md = "\n".join(lines)
        output_path.write_text(md, encoding="utf-8")
        logger.info(f"Combined dataset card written to {output_path} and {json_path}")
        return md


    # ── Internal rendering ────────────────────────────────────────────────────

    def _render_card(self, m: DatasetCardMetadata) -> str:
        """Render a single DatasetCardMetadata as a Markdown string."""
        safe_name = m.name.lower().replace(" ", "_")
        phases_str = ", ".join(m.curriculum_phases) if m.curriculum_phases else "A"
        num_gb = m.num_bytes / (1024 ** 3)

        lines: list[str] = [
            "---",
            f"dataset_name: {safe_name}_certified_v{m.version}",
            f"license: {m.license}",
            "language: en",
            "task_categories: text-generation",
            "tags:",
            "  - iveri-certified",
            "  - phase-8",
            "---",
            "",
            f"# IVERI Dataset Card — {m.name}",
            "",
            "## Overview",
            "",
            f"- **Dataset Name**: `{m.name}`",
            f"- **Version**: `{m.version}`",
            f"- **License**: [{m.license}]({m.license_url})" if m.license_url else f"- **License**: `{m.license}`",
            f"- **Created**: {m.created_at}",
            f"- **Source URL**: {m.source_url}" if m.source_url else "- **Source URL**: —",
            f"- **Curriculum Phases**: {phases_str}",
            "",
            "## Dataset Details",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total Documents | {m.num_documents:,} |",
            f"| Total Size | {m.num_bytes:,} bytes ({num_gb:.4f} GB) |",
            f"| Average Document Length | {m.avg_doc_length:,.1f} bytes |",
            f"| Median Document Length | {m.median_doc_length:,.1f} bytes |",
            f"| Byte Entropy (mean) | {m.byte_entropy_mean:.4f} bits/byte |",
            "",
            "## Certification Pipeline",
            "",
            f"This dataset passed the **IVERI 24-stage deterministic certification pipeline** "
            f"({m.certification_stages_passed}/24 stages).",
            "",
            "### Removal Statistics",
            "",
            "| Stage | Removal Rate |",
            "|---|---|",
            f"| Deduplication | {m.dedup_removal_rate:.2%} |",
            f"| PII Detection | {m.pii_removal_rate:.2%} |",
            f"| Safety Filtering | {m.safety_removal_rate:.2%} |",
            f"| Quality Filtering | {m.quality_removal_rate:.2%} |",
            f"| **Overall Retention** | **{m.overall_retention_rate:.2%}** |",
            "",
            "### Integrity Hashes",
            "",
            f"- **SHA-256 Manifest Hash**: `{m.sha256_manifest_hash or 'N/A'}`",
            f"- **Freeze Certificate Hash**: `{m.freeze_certificate_hash or 'N/A'}`",
            "",
            "## Statistics",
            "",
            "Dataset statistics are stored alongside the certified data in `certified_statistics.json` "
            "and `certified_statistics.md`.",
            "",
            "## License",
            "",
            f"This dataset is released under the [{m.license}]({m.license_url}) license."
            if m.license_url else f"This dataset is released under the `{m.license}` license.",
            "",
            "## Citation",
            "",
            "```bibtex",
            "@misc{iveri_core_2026,",
            f"  title   = {{IVERI CORE Dataset — {m.name}}},",
            "  author  = {IVERI Project},",
            "  year    = {2026},",
            f"  version = {{{m.version}}},",
            "  url     = {https://github.com/iveri-project/iveri-core},",
            "}",
            "```",
        ]
        return "\n".join(lines)
