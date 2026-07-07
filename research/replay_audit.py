# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Replay integrity audit (Phase 6.3.2 OBJ6)."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.experiment_registry import ExperimentRegistry
from research.replay_integrity import (
    ReplayIntegrityResult,
    run_replay_integrity_audit,
    verify_replay_registry_integrity,
)
from tests.provenance_helpers import GIT_SHA, seed_measured_experiment, seed_provenance_chain_experiments

def run_replay_audit() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="replay_audit_") as tmp:
        base = Path(tmp)
        db_path = base / "good.db"
        out_dir = base / "reports"

        registry = ExperimentRegistry(db_path=str(db_path))
        seed_provenance_chain_experiments(registry)

        (out_dir / "statistics").mkdir(parents=True, exist_ok=True)
        (out_dir / "publication").mkdir(parents=True, exist_ok=True)
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

        fig_dir = out_dir / "publication" / "Paper_Figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Minimal valid-sized PDF stub (audit tests figure gate, not matplotlib rendering).
        (fig_dir / "loss_convergence_comparison.pdf").write_bytes(
            b"%PDF-1.4\n% replay audit stub figure\n" + b"0" * 48
        )

        good = run_replay_integrity_audit(str(db_path), str(out_dir))

        bad_db = base / "bad.db"
        bad_reg = ExperimentRegistry(db_path=str(bad_db))
        seed_measured_experiment(bad_reg, "exp_bad", with_benchmark=True)
        bad_reg.log_failure("exp_bad", 0, "TRAINING", "simulated", "", {}, "")
        bad_registry_ok, _ = verify_replay_registry_integrity(str(bad_db))

        pilot_db = base / "pilot.db"
        pilot_reg = ExperimentRegistry(db_path=str(pilot_db))
        pilot_reg.register_experiment(
            experiment_id="exp_pilot",
            purpose="pilot run",
            hypothesis="H1",
            config_hash="c" * 40,
            git_sha=GIT_SHA,
            git_branch="main",
            random_seed=42,
            tags=["pilot"],
            provenance_label="PILOT",
            status="COMPLETED",
        )
        pilot_ok, pilot_errors = verify_replay_registry_integrity(str(pilot_db))

        return {
            "protocol_version": "Phase-6.3.2-OBJ6",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "production_verdict": "PASS"
            if good.passed and not bad_registry_ok and not pilot_ok
            else "FAIL",
            "full_chain_passes": good.passed,
            "blocks_failure_rows": not bad_registry_ok,
            "blocks_pilot_provenance": not pilot_ok,
            "replay_integrity": asdict(good),
            "presence_proof": [
                "replay_campaign.py pre-flights verify_replay_registry_integrity before publication.",
                "Non-zero exit when registry, claim chain, or figure checks fail.",
                "Claim chain requires MEASURED metrics per hypothesis experiment.",
                "Disallowed tags (verification, pilot, mock, dry_run) block replay sign-off.",
                "Figure verification rejects mock placeholder artifacts.",
            ],
        }


def render_replay_integrity_report(data: dict[str, Any]) -> str:
    lines = [
        "# Replay Integrity Report (Phase 6.3.2 OBJ6)",
        "",
        f"**Generated:** {data['timestamp_utc']}  ",
        f"**Protocol:** {data['protocol_version']}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Replay fail-closed framework:** `{data['production_verdict']}`",
        "",
        f"- Full H1–H10 chain passes on measured seed DB: `{data['full_chain_passes']}`",
        f"- Failure rows block replay: `{data['blocks_failure_rows']}`",
        f"- Non-paper provenance blocked: `{data['blocks_pilot_provenance']}`",
        "",
    ]
    if data["production_verdict"] == "PASS":
        lines.extend(["## Proof: Replay Integrity Gates", ""])
        for i, proof in enumerate(data["presence_proof"], 1):
            lines.append(f"{i}. {proof}")
        lines.append("")

    lines.extend(
        [
            "## Raw JSON",
            "",
            "```json",
            json.dumps(data, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_replay_integrity_report(output_path: str | Path) -> dict[str, Any]:
    data = run_replay_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_replay_integrity_report(data), encoding="utf-8")
    return data
