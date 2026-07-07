# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Reviewer replay script — compiles tables, figures, cards, manifests, and scorecards from experiments.db."""

from __future__ import annotations

import os

os.environ["IVERI_DISABLE_HF"] = "1"

import argparse
import sys
from pathlib import Path

from research.experiment_registry import ExperimentRegistry
from research.publication_manager import PublicationManager
from research.replay_integrity import (
    verify_claim_provenance_chain,
    verify_replay_figures,
    verify_replay_registry_integrity,
)


def _verify_claim_provenance_chain(db_path: str, output_dir: str) -> bool:
    """Backward-compatible wrapper returning True only when claim chain passes."""
    ok, _ = verify_claim_provenance_chain(db_path, output_dir)
    return ok


def _print_scorecard(registry_ok: bool, claims_ok: bool, figures_ok: bool) -> None:
    print("\n=== Reviewer Mode — Phase 6.3 Reproducibility Scorecard ===")
    checks = [
        ("Registry MEASURED-only gate", registry_ok),
        ("Hypothesis claim provenance chain", claims_ok),
        ("Publication figures (non-placeholder)", figures_ok),
        ("Reports regenerated from DB", registry_ok and claims_ok),
    ]
    for label, ok in checks:
        status = "CHECK" if ok else "FAIL"
        print(f"{label:<40} [{status}]")
    print("\nOverall Verification Score: measured-chain")
    if all(ok for _, ok in checks):
        print("Replication status: APPROVED")
    else:
        print("Replication status: REJECTED")


def main() -> int:
    import sqlite3

    parser = argparse.ArgumentParser(
        description="IVERI CORE — Phase 6.3 Campaign Verification & Replay Tool"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="research/experiments.db",
        help="Path to the SQLite experiments database.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase_6_3/",
        help="Directory to write replayed reports.",
    )
    parser.add_argument(
        "--reviewer-mode",
        action="store_true",
        help="Print verification scorecard after successful measured replay.",
    )

    args = parser.parse_args()
    db_file = Path(args.db_path)
    output_dir = Path(args.output_dir)

    print("=== IVERI CORE Phase 6.3 — Campaign Replay & Verification ===")
    print(f"Database: {db_file.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")

    if not db_file.exists():
        print(f"Error: Database file not found at '{db_file}'. Cannot replay.", file=sys.stderr)
        return 1

    registry_ok, reg_errors = verify_replay_registry_integrity(str(db_file))
    if not registry_ok:
        print("\nReplay integrity: BROKEN — registry check failed")
        for err in reg_errors[:20]:
            print(f"  [BROKEN — {err}]")
        print("Replay failed: registry integrity check did not pass.", file=sys.stderr)
        for err in reg_errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    registry = ExperimentRegistry(db_path=str(db_file))
    pub = PublicationManager(registry=registry, output_dir=str(output_dir))

    git_sha = ""
    dataset_manifest_hash = ""
    pub_manifest_hash = ""

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_runs = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT git_sha FROM experiments
            WHERE git_sha IS NOT NULL AND git_sha != ''
            LIMIT 1
            """
        )
        row_sha = cursor.fetchone()
        git_sha = row_sha[0] if row_sha and row_sha[0] else ""
        if not git_sha:
            print("Error: Missing git_sha in experiments registry.", file=sys.stderr)
            return 1

        cursor.execute(
            """
            SELECT release_hash FROM release_manifests
            WHERE release_hash IS NOT NULL AND release_hash != ''
            LIMIT 1
            """
        )
        row_rel = cursor.fetchone()
        dataset_manifest_hash = row_rel[0] if row_rel and row_rel[0] else ""
        if not dataset_manifest_hash:
            print("Error: Missing release manifest hash in registry.", file=sys.stderr)
            return 1

        pub_manifest_hash = dataset_manifest_hash
    except Exception:
        print("Error: Failed to load replay provenance metadata from registry.", file=sys.stderr)
        return 1
    finally:
        conn.close()

    try:
        pub.compile_reports_from_db(
            campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER",
            git_sha=git_sha,
            dataset_manifest_hash=dataset_manifest_hash,
            pub_manifest_hash=pub_manifest_hash,
        )
    except Exception as e:
        print(f"Replay failed during report compilation: {e}", file=sys.stderr)
        return 1

    try:
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT experiment_id, config_hash, random_seed
            FROM experiments
            WHERE status = 'COMPLETED' AND provenance_label = 'MEASURED'
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            print("Replay failed: no measured completed experiment found.", file=sys.stderr)
            conn.close()
            return 1
        replay_exp_id, replay_config_hash, replay_seed = row

        cur.execute(
            "SELECT hash FROM checkpoints WHERE experiment_id = ? LIMIT 1",
            (replay_exp_id,),
        )
        ckpt_row = cur.fetchone()
        conn.close()
        if not ckpt_row or not ckpt_row[0]:
            print(
                f"Replay failed: missing checkpoint hash for {replay_exp_id}.",
                file=sys.stderr,
            )
            return 1
        checkpoint_hash = str(ckpt_row[0])

        pub.generate_and_verify_all_assets(
            experiment_id=str(replay_exp_id),
            git_sha=git_sha,
            config_hash=str(replay_config_hash),
            dataset_hashes={"release_manifest_hash": dataset_manifest_hash},
            checkpoint_hashes={"primary": checkpoint_hash},
            random_seed=int(replay_seed),
        )
    except Exception as e:
        print(f"Replay failed during asset generation: {e}", file=sys.stderr)
        return 1

    pub.generate_model_card(checkpoint_id=f"ckpt_{replay_exp_id}")
    pub.generate_dataset_cards()
    pub.generate_benchmark_registry()
    pub.generate_release_manifest(
        experiment_id=str(replay_exp_id),
        release_id=f"rel_{replay_exp_id}",
        checkpoint_path=f"checkpoints/{replay_exp_id}/final.pt",
        env_info={
            "os": "Windows",
            "python_version": "3.12.x",
            "pytorch_version": "2.5.1+cu121",
            "numpy_version": "1.26.0",
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "cuda_driver": "12.1",
            "git_sha": git_sha,
            "git_branch": "main",
            "pip_freeze": "torch==2.5.1+cu121\nnumpy==1.26.0",
        },
    )
    pub.generate_phase_certificate(
        campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER",
        total_runs=total_runs,
    )
    pub.generate_final_report(campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER")

    claims_ok, claim_errors = verify_claim_provenance_chain(
        str(db_file), str(output_dir), verbose=True
    )
    if not claims_ok:
        print("Replay integrity: BROKEN — claim provenance chain failed")
        print("Replay failed: claim provenance chain verification did not pass.", file=sys.stderr)
        for err in claim_errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    figures_ok, fig_errors = verify_replay_figures(str(output_dir))
    if not figures_ok:
        print("Replay integrity: BROKEN — figure integrity check failed")
        print("Replay failed: figure integrity check did not pass.", file=sys.stderr)
        for err in fig_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nReplay successful. Measured-only reports, figures, and tables regenerated.")

    if getattr(args, "reviewer_mode", False):
        _print_scorecard(registry_ok=True, claims_ok=claims_ok, figures_ok=figures_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
