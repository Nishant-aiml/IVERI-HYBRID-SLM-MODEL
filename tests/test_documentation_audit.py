# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ8 documentation sync runtime tests."""

from __future__ import annotations

from pathlib import Path

from core.constants import ARCHITECTURE_VERSION, CURRENT_PHASE
from research.documentation_audit import run_documentation_audit, write_documentation_sync_report

REPORT_PATH = Path("reports/scientific_integrity_audit/Documentation_Sync_Report.md")


def test_readme_documents_phase_632() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "6.3.2" in text
    assert "🔲 Not Started" not in text
    assert ARCHITECTURE_VERSION in text


def test_changelog_documents_632() -> None:
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert "[1.5.0]" in text
    assert "Phase 6.3.2" in text


def test_master_doc_not_phase_0() -> None:
    text = Path("IVERI_PROJECT_MASTER.md").read_text(encoding="utf-8")
    assert "Build starting Phase 0" not in text
    assert "IVERIModel" in text


def test_phases_index_exists() -> None:
    index = Path("docs/phases/INDEX.md")
    assert index.exists()
    assert "Documentation_Sync_Report.md" in index.read_text(encoding="utf-8")


def test_current_phase_constant() -> None:
    assert CURRENT_PHASE >= 6


def test_documentation_audit_pass() -> None:
    result = run_documentation_audit()
    assert result.production_verdict == "PASS"
    assert all(g.passed for g in result.gates)


def test_write_documentation_sync_report(tmp_path: Path) -> None:
    out = tmp_path / "Documentation_Sync_Report.md"
    data = write_documentation_sync_report(out)
    assert out.exists()
    assert "Phase 6.3.2 OBJ8" in out.read_text(encoding="utf-8")
    assert data["production_verdict"] == "PASS"


def test_regenerate_repo_documentation_report() -> None:
    data = write_documentation_sync_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    assert data["production_verdict"] == "PASS"
