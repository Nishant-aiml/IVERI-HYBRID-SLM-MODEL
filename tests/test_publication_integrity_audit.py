# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ5 publication integrity runtime tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.experiment_registry import ExperimentRegistry
from research.publication_audit import (
    run_publication_audit,
    write_publication_integrity_report,
)
from research.publication_manager import PublicationManager
from tests.provenance_helpers import GIT_SHA, seed_measured_experiment

REPORT_PATH = Path("reports/scientific_integrity_audit/Publication_Integrity_Report.md")


def test_publication_blocks_failure_table_rows(tmp_path: Path) -> None:
    reg = ExperimentRegistry(db_path=str(tmp_path / "f.db"))
    seed_measured_experiment(reg, "exp_x")
    reg.log_failure("exp_x", 0, "TRAINING", "err", "", {}, "")
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="failure record"):
        pub._assert_integrity_for_publication()


def test_publication_audit_pass(tmp_path: Path) -> None:
    result = run_publication_audit(tmp_path)
    assert result.production_verdict == "PASS"
    assert result.mock_metrics_path_removed is True
    assert all(g.passed for g in result.gates)


def test_write_publication_integrity_report(tmp_path: Path) -> None:
    out = tmp_path / "Publication_Integrity_Report.md"
    result = write_publication_integrity_report(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Phase 6.3.2 OBJ5" in text
    assert result.production_verdict == "PASS"


def test_regenerate_repo_publication_report() -> None:
    result = write_publication_integrity_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    assert result.production_verdict == "PASS"
