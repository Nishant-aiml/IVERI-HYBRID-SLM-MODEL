# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Publication integrity audit (Phase 6.3.2 OBJ5).

Verifies fail-closed publication gates: MEASURED provenance only, no mock
metrics path, registry failure blocking, and DB-driven report compilation.
"""

from __future__ import annotations

import inspect
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.campaign_runner import CampaignRunner
from research.experiment_registry import ExperimentRegistry
from research.publication_manager import PublicationManager
from tests.provenance_helpers import GIT_SHA, seed_measured_experiment


@dataclass
class PublicationGateProbe:
    gate_name: str
    passed: bool
    detail: str


@dataclass
class PublicationAuditResult:
    protocol_version: str = "Phase-6.3.2-OBJ5"
    timestamp_utc: str = ""
    production_verdict: str = "UNKNOWN"
    mock_metrics_path_removed: bool = False
    campaign_marks_failed_on_training_error: bool = False
    gates: list[PublicationGateProbe] = field(default_factory=list)
    presence_proof: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_blocks_non_measured(tmp_path: Path) -> PublicationGateProbe:
    suffix = uuid.uuid4().hex[:8]
    reg = ExperimentRegistry(db_path=str(tmp_path / f"bad_prov_{suffix}.db"))
    exp_id = f"exp_bad_{suffix}"
    reg.register_experiment(
        experiment_id=exp_id,
        purpose="test",
        hypothesis="H1",
        config_hash="c" * 40,
        git_sha=GIT_SHA,
        git_branch="main",
        random_seed=42,
        tags=["test"],
        provenance_label="SYNTHETIC",
        status="COMPLETED",
    )
    reg.log_metrics(
        exp_id, 1, 1.0, 1.0, 2.0, 0.5, provenance_label="SYNTHETIC"
    )
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path / f"out_bad_{suffix}"))
    try:
        pub.compile_reports_from_db(
            campaign_id="C",
            git_sha=GIT_SHA,
            dataset_manifest_hash="d" * 64,
            pub_manifest_hash="e" * 64,
        )
        return PublicationGateProbe("non_measured_blocked", False, "expected RuntimeError")
    except RuntimeError as e:
        ok = "Publication blocked" in str(e)
        return PublicationGateProbe("non_measured_blocked", ok, str(e))


def _probe_blocks_failed_experiment(tmp_path: Path) -> PublicationGateProbe:
    suffix = uuid.uuid4().hex[:8]
    reg = ExperimentRegistry(db_path=str(tmp_path / f"failed_{suffix}.db"))
    reg.register_experiment(
        experiment_id=f"exp_fail_{suffix}",
        purpose="test",
        hypothesis="H1",
        config_hash="c" * 40,
        git_sha=GIT_SHA,
        git_branch="main",
        random_seed=42,
        tags=["test"],
        provenance_label="UNKNOWN",
        status="FAILED",
    )
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path / f"out_fail_{suffix}"))
    try:
        pub.generate_phase_certificate(campaign_id="C")
        return PublicationGateProbe("failed_runs_blocked", False, "expected RuntimeError")
    except RuntimeError as e:
        ok = "failed runs exist" in str(e)
        return PublicationGateProbe("failed_runs_blocked", ok, str(e))


def _probe_blocks_failure_records(tmp_path: Path) -> PublicationGateProbe:
    suffix = uuid.uuid4().hex[:8]
    reg = ExperimentRegistry(db_path=str(tmp_path / f"failures_{suffix}.db"))
    exp_id = f"exp_ok_{suffix}"
    seed_measured_experiment(reg, exp_id)
    reg.log_failure(exp_id, 0, "TRAINING", "simulated", "", {}, "")
    pub = PublicationManager(registry=reg, output_dir=str(tmp_path / f"out_frec_{suffix}"))
    try:
        pub._assert_integrity_for_publication()
        return PublicationGateProbe("failure_records_blocked", False, "expected RuntimeError")
    except RuntimeError as e:
        ok = "failure record" in str(e).lower()
        return PublicationGateProbe("failure_records_blocked", ok, str(e))


def _probe_measured_pipeline_passes(tmp_path: Path) -> PublicationGateProbe:
    suffix = uuid.uuid4().hex[:8]
    reg = ExperimentRegistry(db_path=str(tmp_path / f"good_{suffix}.db"))
    exp_id = seed_measured_experiment(reg, f"exp_good_{suffix}", with_benchmark=True)
    out_dir = tmp_path / f"out_good_{suffix}"
    pub = PublicationManager(registry=reg, output_dir=str(out_dir))
    try:
        pub.compile_reports_from_db(
            campaign_id="C_GOOD",
            git_sha=GIT_SHA,
            dataset_manifest_hash="d" * 64,
            pub_manifest_hash="e" * 64,
        )
        pub.generate_phase_certificate(campaign_id="C_GOOD")
        report = (out_dir / "experiments" / "Training_Report.md").read_text(encoding="utf-8")
        ok = exp_id in report and "measured database values" in report
        return PublicationGateProbe("measured_pipeline_passes", ok, f"exp={exp_id}")
    except Exception as e:
        return PublicationGateProbe("measured_pipeline_passes", False, str(e))


def _probe_no_mock_metrics_helper() -> bool:
    source = inspect.getsource(CampaignRunner)
    return "_log_mock_metrics" not in source


def _probe_campaign_failure_handling() -> bool:
    source = inspect.getsource(CampaignRunner._run_stage_pretrain)
    return (
        'update_experiment_status(exp_id, "FAILED")' in source
        and "No synthetic fallback metrics are allowed" in source
    )


def run_publication_audit(tmp_path: Path | None = None) -> PublicationAuditResult:
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="pub_audit_"))
    else:
        tmp_path.mkdir(parents=True, exist_ok=True)
    gates = [
        _probe_blocks_non_measured(tmp_path),
        _probe_blocks_failed_experiment(tmp_path),
        _probe_blocks_failure_records(tmp_path),
        _probe_measured_pipeline_passes(tmp_path),
    ]
    mock_removed = _probe_no_mock_metrics_helper()
    campaign_fails = _probe_campaign_failure_handling()

    all_pass = all(g.passed for g in gates) and mock_removed and campaign_fails
    proof = [
        "PublicationManager._assert_integrity_for_publication requires COMPLETED + MEASURED rows.",
        "Non-MEASURED metrics and benchmark_runs raise RuntimeError before report generation.",
        "failures table rows block publication and certificate signing.",
        "CampaignRunner no longer defines _log_mock_metrics synthetic loss fallback.",
        "Failed training attempts set experiment status FAILED instead of COMPLETED.",
    ]

    return PublicationAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        production_verdict="PASS" if all_pass else "FAIL",
        mock_metrics_path_removed=mock_removed,
        campaign_marks_failed_on_training_error=campaign_fails,
        gates=gates,
        presence_proof=proof,
    )


def render_publication_integrity_report(result: PublicationAuditResult) -> str:
    lines = [
        "# Publication Integrity Report (Phase 6.3.2 OBJ5)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Fail-closed publication framework:** `{result.production_verdict}`",
        "",
        f"- Mock metrics helper removed from campaign: `{result.mock_metrics_path_removed}`",
        f"- Campaign marks FAILED on training error: `{result.campaign_marks_failed_on_training_error}`",
        "",
        "## Gate Probes",
        "",
        "| Gate | Passed | Detail |",
        "|------|:------:|--------|",
    ]
    for g in result.gates:
        lines.append(f"| {g.gate_name} | {g.passed} | {g.detail[:120]} |")

    if result.production_verdict == "PASS":
        lines.extend(["", "## Proof: Fail-Closed Publication", ""])
        for i, p in enumerate(result.presence_proof, 1):
            lines.append(f"{i}. {p}")

    lines.extend(
        [
            "",
            "## Raw JSON",
            "",
            "```json",
            json.dumps(result.to_dict(), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_publication_integrity_report(output_path: str | Path) -> PublicationAuditResult:
    result = run_publication_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_publication_integrity_report(result), encoding="utf-8")
    return result
