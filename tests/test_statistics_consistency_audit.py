# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.1G statistics pipeline consistency tests."""

from __future__ import annotations

from pathlib import Path

from research.statistics import CANONICAL_STATISTICS_METHODS, ResearchStatisticalValidator
from research.statistics_consistency_audit import (
    run_statistics_consistency_audit,
    write_statistics_consistency_report,
)

REPORT_PATH = Path("reports/scientific_integrity_audit/Statistics_Consistency_Report.md")


def test_canonical_methods_count() -> None:
    assert len(CANONICAL_STATISTICS_METHODS) == 7


def test_compute_paired_hypothesis_statistics_bundle() -> None:
    validator = ResearchStatisticalValidator()
    baseline = [1.2, 1.1, 1.15, 1.08, 1.12]
    treatment = [1.0, 1.05, 0.98, 1.02, 1.01]
    bundle = validator.compute_paired_hypothesis_statistics(
        baseline, treatment, metric_name="val_loss"
    )
    assert bundle["status"] == "OK"
    assert bundle["pipeline_version"] == "Phase-6.3.1G"
    assert "shapiro_wilk" in bundle
    assert "holm_adjusted_p_value" in bundle
    assert bundle["holm_adjusted_p_value"] is not None


def test_statistics_consistency_audit_pass() -> None:
    result = run_statistics_consistency_audit()
    assert result.production_verdict == "PASS"
    assert result.bundle_covers_all_methods
    assert not result.duplicate_calculation_violations


def test_write_statistics_consistency_report() -> None:
    result = write_statistics_consistency_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Phase 6.3.1G" in text
    assert result.production_verdict == "PASS"
