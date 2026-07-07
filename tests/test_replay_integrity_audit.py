# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ6 replay integrity runtime tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.experiment_registry import ExperimentRegistry
from research.replay_audit import run_replay_audit, write_replay_integrity_report
from research.replay_integrity import (
    verify_claim_provenance_chain,
    verify_replay_figures,
    verify_replay_registry_integrity,
)
from tests.provenance_helpers import GIT_SHA, seed_measured_experiment, seed_provenance_chain_experiments

REPORT_PATH = Path("reports/scientific_integrity_audit/Replay_Integrity_Report.md")


def test_registry_blocks_failure_rows(tmp_path: Path) -> None:
    reg = ExperimentRegistry(db_path=str(tmp_path / "f.db"))
    seed_measured_experiment(reg, "exp_x", with_benchmark=True)
    reg.log_failure("exp_x", 0, "TRAINING", "err", "", {}, "")
    ok, errors = verify_replay_registry_integrity(str(tmp_path / "f.db"))
    assert not ok
    assert any("failure" in e.lower() for e in errors)


def test_registry_blocks_pilot_tag(tmp_path: Path) -> None:
    db = tmp_path / "pilot.db"
    reg = ExperimentRegistry(db_path=str(db))
    reg.register_experiment(
        experiment_id="exp_pilot",
        purpose="pilot",
        hypothesis="H1",
        config_hash="c" * 40,
        git_sha=GIT_SHA,
        git_branch="main",
        random_seed=42,
        tags=["pilot"],
        provenance_label="PILOT",
        status="COMPLETED",
    )
    ok, errors = verify_replay_registry_integrity(str(db))
    assert not ok
    assert any("pilot" in e.lower() or "provenance" in e.lower() for e in errors)


def test_claim_chain_verbose_broken(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "reports"
    (out_dir / "statistics").mkdir(parents=True)
    (out_dir / "publication").mkdir(parents=True)
    ok, _ = verify_claim_provenance_chain(
        str(tmp_path / "empty.db"), str(out_dir), verbose=True
    )
    assert not ok
    captured = capsys.readouterr()
    assert "BROKEN" in captured.out


def test_claim_chain_passes_with_seed(tmp_path: Path) -> None:
    db = tmp_path / "good.db"
    out_dir = tmp_path / "reports"
    (out_dir / "statistics").mkdir(parents=True)
    (out_dir / "publication").mkdir(parents=True)
    reg = ExperimentRegistry(db_path=str(db))
    seed_provenance_chain_experiments(reg)
    exp_mentions = "\n".join(
        [f"IVERI_Phase5_pretrain_Seed42_IVERI_Run{i:03d}" for i in range(1, 11)]
    )
    (out_dir / "statistics" / "Hypothesis_Report.md").write_text(
        f"Hypothesis Report\n{exp_mentions}", encoding="utf-8"
    )
    evidence_mentions = "\n".join(
        [f"Hypothesis H{i} is SUPPORTED with p < 0.05" for i in range(1, 11)]
    )
    (out_dir / "publication" / "Evidence_Index.md").write_text(
        f"Evidence Index\n{evidence_mentions}", encoding="utf-8"
    )
    ok, errors = verify_claim_provenance_chain(str(db), str(out_dir))
    assert ok, errors


def test_figures_reject_placeholder(tmp_path: Path) -> None:
    fig_dir = tmp_path / "publication" / "Paper_Figures"
    fig_dir.mkdir(parents=True)
    (fig_dir / "loss_convergence_comparison.txt").write_text(
        "mock figure placeholder", encoding="utf-8"
    )
    ok, errors = verify_replay_figures(str(tmp_path))
    assert not ok
    assert any("placeholder" in e for e in errors)


def test_replay_audit_pass() -> None:
    data = run_replay_audit()
    assert data["production_verdict"] == "PASS"
    assert data["full_chain_passes"] is True
    assert data["blocks_failure_rows"] is True
    assert data["blocks_pilot_provenance"] is True


def test_write_replay_integrity_report(tmp_path: Path) -> None:
    out = tmp_path / "Replay_Integrity_Report.md"
    data = write_replay_integrity_report(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Phase 6.3.2 OBJ6" in text
    assert data["production_verdict"] == "PASS"


def test_regenerate_repo_replay_report() -> None:
    data = write_replay_integrity_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    assert data["production_verdict"] == "PASS"
