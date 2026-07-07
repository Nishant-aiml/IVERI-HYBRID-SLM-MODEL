# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point to launch Phase 6.3 Production Empirical Training & Scientific Validation Campaign."""

from __future__ import annotations

import os
os.environ["IVERI_DISABLE_HF"] = "1"

import argparse
import logging
import sys

from research.campaign_runner import CampaignRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="IVERI CORE — Phase 6.3 Campaign Execution Runner")
    parser.add_argument(
        "--profile",
        type=str,
        default="verification",
        choices=["verification", "pilot", "full", "paper"],
        help="Campaign declarative profile setting scale of steps and seeds.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "rtx3050", "kaggle", "colab", "vast", "lambda"],
        help="Hardware tier adapter configuration.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="research/experiments.db",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase_6_3/",
        help="Output reports folder location.",
    )
    parser.add_argument(
        "--skip-integrity-halt",
        action="store_true",
        help="Skip hard halt upon benchmark integrity failure (warning-only mode).",
    )
    parser.add_argument(
        "--resume-strategy",
        type=str,
        default="AUTO",
        choices=["AUTO", "FROM_LAST", "FROM_BEST", "FROM_GOLDEN", "FROM_CHECKPOINT_ID"],
        help="Strategy to resume campaign checkpoints.",
    )
    # ── Phase C: Ablation Studies ─────────────────────────────────────────────
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_titans", "no_blt", "no_mor", "no_moe", "no_entropy_routing"],
        help=(
            "Disable an architectural component for ablation. "
            "Applied via config override only — no source edits."
        ),
    )
    # ── Phase D: Downstream Specialization Stage ──────────────────────────────
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft", "coding", "alignment", "all"],
        help=(
            "Training stage to execute. "
            "'sft', 'coding', 'alignment' each initialize from the promoted Phase B checkpoint. "
            "'all' runs pretrain → sft → coding → alignment sequentially."
        ),
    )
    # ── Phase E: Benchmarks Only ──────────────────────────────────────────────
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        help=(
            "Skip training. Load the promoted golden checkpoint and execute "
            "the full benchmark suite only (Phase E)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform pre-flight checks without starting training.",
    )

    args = parser.parse_args()

    try:
        runner = CampaignRunner(
            profile_name=args.profile,
            backend_name=args.backend,
            db_path=args.db_path,
            output_dir=args.output_dir,
            resume_strategy=args.resume_strategy,
            ablation=args.ablation,
            stage=args.stage,
            benchmarks_only=args.benchmarks_only,
            dry_run=args.dry_run,
            skip_integrity_halt=args.skip_integrity_halt,
        )
        res = runner.run_campaign()
        if res["status"] in ("SUCCESS", "SKIPPED", "DRY_RUN_COMPLETED"):
            logger.info("Campaign completed successfully.")
            return 0
        else:
            logger.error(f"Campaign ended with status: {res['status']}. Details: {res}")
            return 1
    except Exception as e:
        logger.exception(f"Fatal error executing campaign: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
