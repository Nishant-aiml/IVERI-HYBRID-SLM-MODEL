# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Replay integrity verification (Phase 6.3.2 OBJ6).

Fail-closed checks for campaign replay: registry provenance, measured metrics,
disallowed profile tags, and hypothesis claim chains.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DISALLOWED_REPLAY_TAGS = frozenset({"verification", "pilot", "mock", "dry_run"})


@dataclass
class ReplayIntegrityResult:
    protocol_version: str = "Phase-6.3.2-OBJ6"
    registry_ok: bool = False
    claims_ok: bool = False
    figures_ok: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.registry_ok and self.claims_ok and self.figures_ok and not self.errors


def _parse_tags(tags_raw: str | None) -> list[str]:
    if not tags_raw:
        return []
    return [t.strip().lower() for t in str(tags_raw).split(",") if t.strip()]


def verify_replay_registry_integrity(db_path: str) -> tuple[bool, list[str]]:
    """Verify DB is eligible for paper-profile replay (MEASURED only, no failures)."""
    errors: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        failure_count = int(cur.execute("SELECT COUNT(*) FROM failures").fetchone()[0])
        if failure_count > 0:
            errors.append(f"failures table has {failure_count} row(s)")

        cur.execute("SELECT experiment_id, status, provenance_label, tags FROM experiments")
        for exp_id, status, prov, tags_raw in cur.fetchall():
            if str(status).upper() != "COMPLETED":
                errors.append(f"experiment {exp_id} status={status}")
            if str(prov).upper() != "MEASURED":
                errors.append(f"experiment {exp_id} provenance={prov}")
            for tag in _parse_tags(tags_raw):
                if tag in DISALLOWED_REPLAY_TAGS:
                    errors.append(f"experiment {exp_id} has disallowed tag '{tag}'")

            metric_count = int(
                cur.execute(
                    """
                    SELECT COUNT(*) FROM metrics
                    WHERE experiment_id = ? AND provenance_label = 'MEASURED'
                    """,
                    (exp_id,),
                ).fetchone()[0]
            )
            if metric_count == 0:
                errors.append(f"experiment {exp_id} has no MEASURED metrics")

        bad_metrics = int(
            cur.execute(
                "SELECT COUNT(*) FROM metrics WHERE provenance_label != 'MEASURED'"
            ).fetchone()[0]
        )
        if bad_metrics > 0:
            errors.append(f"{bad_metrics} non-MEASURED metric row(s)")

        bad_bench = int(
            cur.execute(
                "SELECT COUNT(*) FROM benchmark_runs WHERE provenance_label != 'MEASURED'"
            ).fetchone()[0]
        )
        if bad_bench > 0:
            errors.append(f"{bad_bench} non-MEASURED benchmark row(s)")

        exp_count = int(cur.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
        if exp_count == 0:
            errors.append("no experiments in registry")
    finally:
        conn.close()

    return len(errors) == 0, errors


def verify_claim_provenance_chain(
    db_path: str,
    output_dir: str,
    *,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Verify H1–H10 structural claim chains including measured metrics."""
    errors: list[str] = []
    hypotheses = [f"H{i}" for i in range(1, 11)]

    if verbose:
        print("\nClaim Provenance Chains:")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for hyp in hypotheses:
        def fail(msg: str) -> None:
            errors.append(f"{hyp}: {msg}")
            if verbose:
                print(f"  {hyp:<34} [BROKEN — {msg}]")

        try:
            cursor.execute(
                """
                SELECT experiment_id, random_seed, config_hash, git_sha, status, provenance_label
                FROM experiments
                WHERE hypothesis = ?
                ORDER BY experiment_id
                LIMIT 1
                """,
                (hyp,),
            )
            row = cursor.fetchone()
            if not row:
                fail("missing experiment")
                continue
            exp_id, seed, config_hash, git_sha, status, provenance_label = row
            if status != "COMPLETED":
                fail(f"{exp_id} status={status}")
                continue
            if provenance_label != "MEASURED":
                fail(f"{exp_id} provenance={provenance_label}")
                continue
            if not seed or not config_hash or not git_sha:
                fail(f"incomplete metadata for {exp_id}")
                continue

            metric_count = int(
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM metrics
                    WHERE experiment_id = ? AND provenance_label = 'MEASURED'
                    """,
                    (exp_id,),
                ).fetchone()[0]
            )
            if metric_count == 0:
                fail(f"no MEASURED metrics for {exp_id}")
                continue

            cursor.execute(
                "SELECT checkpoint_id, hash FROM checkpoints WHERE experiment_id = ? LIMIT 1",
                (exp_id,),
            )
            row_ckpt = cursor.fetchone()
            if not row_ckpt or not row_ckpt[1]:
                fail(f"missing checkpoint hash for {exp_id}")
                continue

            cursor.execute(
                """
                SELECT br.run_id, bi.dataset_hash_ok, bi.prompt_hash_ok, br.benchmark_id,
                       br.provenance_label
                FROM benchmark_runs br
                JOIN benchmark_integrity bi ON br.run_id = bi.run_id
                WHERE br.experiment_id = ? LIMIT 1
                """,
                (exp_id,),
            )
            row_bench = cursor.fetchone()
            if not row_bench:
                fail(f"missing benchmark integrity for {exp_id}")
                continue
            run_id, ds_ok, prompt_ok, bench_id, bench_provenance = row_bench
            if not ds_ok or not prompt_ok:
                fail(f"integrity flags failed for {run_id}")
                continue
            if bench_provenance != "MEASURED":
                fail(f"benchmark {run_id} provenance={bench_provenance}")
                continue

            cursor.execute(
                "SELECT version FROM benchmark_registry WHERE benchmark_id = ? LIMIT 1",
                (bench_id,),
            )
            if not cursor.fetchone():
                fail(f"missing benchmark registry for {bench_id}")
                continue

            hyp_report_path = Path(output_dir) / "statistics" / "Hypothesis_Report.md"
            if not hyp_report_path.exists():
                fail("Hypothesis_Report.md missing")
                continue
            report_content = hyp_report_path.read_text(encoding="utf-8")
            if exp_id not in report_content:
                fail(f"{exp_id} not in Hypothesis_Report.md")
                continue

            evidence_index_path = Path(output_dir) / "publication" / "Evidence_Index.md"
            if not evidence_index_path.exists():
                fail("Evidence_Index.md missing")
                continue
            evidence_content = evidence_index_path.read_text(encoding="utf-8")
            if hyp not in evidence_content:
                fail("not in Evidence_Index.md")
                continue
            if "PENDING" in evidence_content:
                fail("PENDING in Evidence_Index.md")
                continue

            if verbose:
                print(f"  {hyp:<34} [OK]")
        except Exception as exc:
            fail(str(exc))

    conn.close()
    return len(errors) == 0, errors


def verify_replay_figures(output_dir: str) -> tuple[bool, list[str]]:
    """Ensure generated figures exist and are not mock placeholders."""
    errors: list[str] = []
    fig_dir = Path(output_dir) / "publication" / "Paper_Figures"
    candidates = list(fig_dir.glob("loss_convergence_comparison.*"))
    if not candidates:
        errors.append("no loss_convergence_comparison figure found")
        return False, errors

    for path in candidates:
        if path.suffix.lower() == ".pdf":
            if path.stat().st_size < 64:
                errors.append(f"figure too small: {path.name}")
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                if "mock figure placeholder" in text:
                    errors.append(f"placeholder figure: {path.name}")
            except OSError as exc:
                errors.append(f"cannot read figure {path.name}: {exc}")

    return len(errors) == 0, errors


def run_replay_integrity_audit(
    db_path: str,
    output_dir: str,
) -> ReplayIntegrityResult:
    registry_ok, reg_errors = verify_replay_registry_integrity(db_path)
    claims_ok, claim_errors = verify_claim_provenance_chain(db_path, output_dir)
    figures_ok, fig_errors = verify_replay_figures(output_dir)
    return ReplayIntegrityResult(
        registry_ok=registry_ok,
        claims_ok=claims_ok,
        figures_ok=figures_ok,
        errors=reg_errors + claim_errors + fig_errors,
    )
