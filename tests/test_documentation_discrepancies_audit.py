# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.1H documentation discrepancies audit tests."""

from __future__ import annotations

from pathlib import Path

from research.documentation_discrepancies_audit import (
    run_documentation_discrepancies_audit,
    write_documentation_discrepancies_report,
)

REPORT_PATH = Path("reports/scientific_integrity_audit/Documentation_Discrepancies.md")


def test_documentation_discrepancies_audit_runs() -> None:
    result = run_documentation_discrepancies_audit()
    assert result.protocol_version == "Phase-6.3.1H"
    assert len(result.documents_scanned) >= 5
    assert len(result.items) >= 1


def test_write_documentation_discrepancies_report() -> None:
    result = write_documentation_discrepancies_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Phase 6.3.1H" in text
    assert "VERIFIED" in text or "PENDING" in text or "TODO" in text
    assert result.production_verdict in {"PASS", "FAIL"}
