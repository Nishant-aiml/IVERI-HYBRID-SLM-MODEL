# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ2 runtime Titans memory audit tests."""

from __future__ import annotations

from pathlib import Path

import torch

from research.titans_audit import (
    TitansInstrumentor,
    audit_backbone_production,
    build_mini_config,
    run_titans_audit,
    write_titans_verification_report,
)
from model.titans.memory import TitansMemory

REPORT_PATH = Path("reports/scientific_integrity_audit/Titans_Verification.md")


def test_isolated_forward_performs_online_updates() -> None:
    cfg = build_mini_config()
    memory = TitansMemory(cfg)
    instrumentor = TitansInstrumentor(memory)
    x = torch.randn(1, 5, cfg.model.hidden_dim)
    memory.forward(x)
    assert instrumentor.counts["updater.update"] == 5
    assert memory.telemetry.get("update_count", 0) == 5
    instrumentor.restore()


def test_production_path_forward_with_online_writes() -> None:
    result = audit_backbone_production()
    assert result.forward_calls >= 1
    assert result.updater_calls > 0
    assert result.inject_calls == 0
    assert result.online_weight_delta_after > 0.0
    assert result.telemetry_reports_writes is True


def test_production_lifecycle_snapshots_recorded() -> None:
    result = audit_backbone_production()
    stages = [s.stage for s in result.snapshots]
    assert "before_forward" in stages
    assert "after_forward" in stages
    assert "after_backward" in stages
    assert "after_optimizer" in stages
    assert "after_second_forward" in stages


def test_full_audit_production_pass() -> None:
    result = run_titans_audit()
    assert result.production_verdict == "PASS"
    assert result.writes_occur_in_production is True
    assert len(result.write_presence_proof) >= 5
    assert result.isolated_forward_updates > 0


def test_generate_titans_verification_report() -> None:
    result = write_titans_verification_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Phase 6.3.2 OBJ2" in text
    assert "forward" in text
    assert result.writes_occur_in_production is True
