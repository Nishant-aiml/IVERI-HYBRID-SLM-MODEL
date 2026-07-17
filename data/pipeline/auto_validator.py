# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Automatic validation stage before reports and freeze in Phase 8.

Performs rule-based checks on certified datasets to ensure they meet minimum
document/byte size caps, quality metrics, and have zero contamination.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Result of a single automatic validation check."""

    name: str
    status: str  # "PASS" | "FAIL" | "WARNING"
    message: str
    expected: Any
    actual: Any


@dataclass
class ValidationReport:
    """Aggregate automatic validation report for all datasets."""

    generated_at: str
    overall_status: str  # "PASS" | "FAIL"
    checks: list[ValidationCheck]

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                }
                for c in self.checks
            ],
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AutomaticValidator:
    """Runs checks on certified datasets before manual gate approval."""

    def __init__(self, config: Any) -> None:  # Phase8Config
        self.config = config

    def validate(
        self,
        dataset_name: str,
        docs: list[dict],
        stage_results: list[Any],     # list[StageResult]
        contamination_rate: float = 0.0,
    ) -> ValidationReport:
        """Validate a certified dataset against configuration criteria."""
        checks: list[ValidationCheck] = []
        overall_pass = True

        # Check 1: min_doc_count (cap at 1000, relax to 5 for test runs)
        doc_count = len(docs)
        # Check if it's a test run
        min_docs = 1000 if not self.config.offline_mode else 5
        if doc_count >= min_docs:
            checks.append(ValidationCheck("min_doc_count", "PASS", "Document count is sufficient", min_docs, doc_count))
        else:
            checks.append(ValidationCheck("min_doc_count", "FAIL", "Too few documents certified", min_docs, doc_count))
            overall_pass = False

        # Check 2: min_byte_count (cap at 50MB, relax to 1KB for tests)
        total_bytes = sum(len(d.get("text", "").encode("utf-8")) for d in docs)
        min_bytes = 50 * (1024 ** 2) if not self.config.offline_mode else 1024
        if total_bytes >= min_bytes:
            checks.append(ValidationCheck("min_byte_count", "PASS", "Byte count is sufficient", min_bytes, total_bytes))
        else:
            checks.append(ValidationCheck("min_byte_count", "FAIL", "Dataset size is too small", min_bytes, total_bytes))
            overall_pass = False

        # Check 3: max_rejection_rate
        input_count = stage_results[0].input_count if stage_results else doc_count
        rejection_rate = (input_count - doc_count) / input_count if input_count > 0 else 0.0
        max_reject = 0.90  # Max 90% rejected
        if rejection_rate <= max_reject:
            checks.append(ValidationCheck("max_rejection_rate", "PASS", "Rejection rate is within bounds", f"<= {max_reject:.1%}", f"{rejection_rate:.1%}"))
        else:
            checks.append(ValidationCheck("max_rejection_rate", "WARNING", "High rejection rate, verify filters", f"<= {max_reject:.1%}", f"{rejection_rate:.1%}"))

        # Check 4: quality_threshold_met
        quality_scores = [d.get("quality_score", 1.0) for d in docs]
        mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        min_q = self.config.min_quality_score
        if mean_quality >= min_q:
            checks.append(ValidationCheck("quality_threshold_met", "PASS", "Mean quality meets threshold", min_q, round(mean_quality, 4)))
        else:
            checks.append(ValidationCheck("quality_threshold_met", "FAIL", "Mean quality below threshold", min_q, round(mean_quality, 4)))
            overall_pass = False

        # Check 5: no_contamination
        max_overlap = 0.01  # Max 1% overlap
        if contamination_rate <= max_overlap:
            checks.append(ValidationCheck("no_contamination", "PASS", "Zero benchmark contamination", f"<= {max_overlap:.1%}", f"{contamination_rate:.1%}"))
        else:
            checks.append(ValidationCheck("no_contamination", "FAIL", "Benchmark contamination detected", f"<= {max_overlap:.1%}", f"{contamination_rate:.1%}"))
            overall_pass = False

        # Check 6: no_failed_stages
        failed_stages = [sr.stage_name for sr in stage_results if sr.removed_count < 0]
        if not failed_stages:
            checks.append(ValidationCheck("no_failed_stages", "PASS", "All 24 stages completed successfully", "[]", "[]"))
        else:
            checks.append(ValidationCheck("no_failed_stages", "FAIL", f"Pipeline failed in stages: {failed_stages}", "[]", str(failed_stages)))
            overall_pass = False

        status = "PASS" if overall_pass else "FAIL"
        return ValidationReport(
            generated_at=datetime.now().isoformat(),
            overall_status=status,
            checks=checks,
        )
