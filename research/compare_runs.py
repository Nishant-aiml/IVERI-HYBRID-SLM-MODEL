# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run Comparator comparing metrics and statistical significance between runs with connection closing."""

from __future__ import annotations

import logging
from typing import Any

from research.experiment_registry import ExperimentRegistry
from research.statistics import ResearchStatisticalValidator

logger = logging.getLogger(__name__)


class RunComparator:
    """Queries SQLite metrics and runs statistical comparison deltas between runs."""

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()
        self.stats = ResearchStatisticalValidator()

    def compare_two_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two runs by querying SQLite database.

        Args:
            run_id_a: Baseline run ID.
            run_id_b: Model run ID.

        Returns:
            dict[str, Any]: Comparison variables and significance scores.
        """
        # Fetch metrics from SQLite database
        conn = self.registry._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step, train_loss, val_loss, perplexity, accuracy FROM metrics WHERE experiment_id = ? ORDER BY step",
                (run_id_a,),
            )
            metrics_a = cursor.fetchall()

            cursor.execute(
                "SELECT step, train_loss, val_loss, perplexity, accuracy FROM metrics WHERE experiment_id = ? ORDER BY step",
                (run_id_b,),
            )
            metrics_b = cursor.fetchall()
        finally:
            conn.close()

        if not metrics_a or not metrics_b:
            logger.warning("One of the runs has no metric logs in the registry database. Returning stub deltas.")
            return {"status": "INSUFFICIENT_DATA"}

        # Match steps to form paired datasets
        dict_a = {row[0]: (row[1], row[2], row[3], row[4]) for row in metrics_a}
        dict_b = {row[0]: (row[1], row[2], row[3], row[4]) for row in metrics_b}

        common_steps = sorted(list(set(dict_a.keys()) & set(dict_b.keys())))
        if len(common_steps) < 2:
            logger.warning("Fewer than 2 matching steps found. Returning baseline comparisons.")
            return {"status": "INSUFFICIENT_STEPS"}

        losses_a = [dict_a[s][1] for s in common_steps]  # val_loss
        losses_b = [dict_b[s][1] for s in common_steps]

        ppl_a = [dict_a[s][2] for s in common_steps]
        ppl_b = [dict_b[s][2] for s in common_steps]

        # Calculate means
        mean_loss_a = sum(losses_a) / len(losses_a)
        mean_loss_b = sum(losses_b) / len(losses_b)
        mean_ppl_a = sum(ppl_a) / len(ppl_a)
        mean_ppl_b = sum(ppl_b) / len(ppl_b)

        # Calculate percentage changes
        loss_delta_pct = (mean_loss_b - mean_loss_a) / mean_loss_a if mean_loss_a > 0 else 0.0
        ppl_delta_pct = (mean_ppl_b - mean_ppl_a) / mean_ppl_a if mean_ppl_a > 0 else 0.0

        # Canonical paired statistics (Phase 6.3.1G — single pipeline)
        stats_bundle = self.stats.compute_paired_hypothesis_statistics(
            losses_a, losses_b, metric_name="val_loss"
        )
        if stats_bundle.get("status") != "OK":
            return {"status": "INSUFFICIENT_STEPS"}

        t_test = stats_bundle["paired_t_test"]
        wilcoxon = stats_bundle["wilcoxon"]
        cohens_d = stats_bundle["cohens_d"]
        bootstrap_ci = (
            stats_bundle["bootstrap_95_ci"]["lower"],
            stats_bundle["bootstrap_95_ci"]["upper"],
        )
        is_significant = stats_bundle["primary_p_value"] < 0.05 and loss_delta_pct < 0.0

        return {
            "status": "SUCCESS",
            "common_steps_count": len(common_steps),
            "run_a": {
                "experiment_id": run_id_a,
                "mean_val_loss": mean_loss_a,
                "mean_perplexity": mean_ppl_a,
            },
            "run_b": {
                "experiment_id": run_id_b,
                "mean_val_loss": mean_loss_b,
                "mean_perplexity": mean_ppl_b,
            },
            "deltas": {
                "val_loss_absolute": mean_loss_b - mean_loss_a,
                "val_loss_percentage": loss_delta_pct,
                "perplexity_absolute": mean_ppl_b - mean_ppl_a,
                "perplexity_percentage": ppl_delta_pct,
            },
            "statistics": {
                "pipeline_version": stats_bundle["pipeline_version"],
                "selected_test": stats_bundle["selected_test"],
                "shapiro_wilk": stats_bundle["shapiro_wilk"],
                "t_statistic": t_test["t_statistic"],
                "t_test_p_value": t_test["p_value"],
                "wilcoxon_w_statistic": wilcoxon["w_statistic"],
                "wilcoxon_p_value": wilcoxon["p_value"],
                "cohens_d": cohens_d,
                "cliffs_delta": stats_bundle["cliffs_delta"],
                "bootstrap_95_ci": bootstrap_ci,
                "primary_p_value": stats_bundle["primary_p_value"],
                "is_statistically_significant": is_significant,
            }
        }
