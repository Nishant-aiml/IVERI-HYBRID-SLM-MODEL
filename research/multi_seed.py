# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Multi-seed runner for statistical analysis of validation metrics across 5 seeds."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from research.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class MultiSeedRunner:
    """Orchestrates 5-seed validation sweeps.

    Calculates mean, variance, standard deviation, and 95% confidence bounds.
    """

    def __init__(self, config: IVERIConfig, runner: ExperimentRunner | None = None) -> None:
        self.config = config
        self.runner = runner or ExperimentRunner(config)
        self.seeds = config.research.random_seeds

    def run_multi_seed(
        self,
        model_builder: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_steps: int = 100,
    ) -> dict[str, dict[str, float]]:
        """Run training and validation loops across all configured seeds.

        Args:
            model_builder: Callback function that builds a fresh model instance.
            train_loader: Dataloader with training sequences.
            val_loader: Dataloader with validation sequences.
            max_steps: Steps budget.

        Returns:
            dict[str, dict[str, float]]: Aggregated statistical metrics per key.
        """
        all_run_metrics: list[dict[str, Any]] = []

        for seed in self.seeds:
            logger.info(f"Triggering seed run {seed} out of {self.seeds}...")
            # Always build a fresh model instance per seed run to reset parameters
            model = model_builder()
            metrics = self.runner.run_experiment(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                max_steps=max_steps,
                seed=seed
            )
            all_run_metrics.append(metrics)

        # Aggregate keys
        aggregate_keys = ["train_loss", "val_loss", "perplexity", "runtime_seconds"]
        aggregated_stats: dict[str, dict[str, float]] = {}

        n = len(all_run_metrics)
        for key in aggregate_keys:
            values = [run[key] for run in all_run_metrics]
            mean_val = sum(values) / n
            variance_val = sum((v - mean_val) ** 2 for v in values) / max(1, n - 1)
            std_val = math.sqrt(variance_val)
            # 95% confidence interval using Student's t distribution approximation (t_val = 2.776 for df=4)
            # or z-score approximation of 1.96
            margin_error = 1.96 * (std_val / math.sqrt(n)) if n > 1 else 0.0

            aggregated_stats[key] = {
                "mean": mean_val,
                "variance": variance_val,
                "std": std_val,
                "ci_lower": mean_val - margin_error,
                "ci_upper": mean_val + margin_error,
                "min": min(values),
                "max": max(values),
            }

        logger.info(f"Aggregated multi-seed stats complete: {aggregated_stats}")
        return aggregated_stats
