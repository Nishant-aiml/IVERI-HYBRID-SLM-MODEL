# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Cost Estimator module predicting runtime, storage, energy, and cloud dollar costs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CostEstimator:
    """Pre-flight estimator calculating resources footprint before validation runs."""

    def __init__(self, backend_name: str = "local") -> None:
        self.backend_name = backend_name.lower()
        # Constants per backend: (step_duration_sec, gpu_wattage, hourly_cost_usd)
        self.backend_params = {
            "local": (0.05, 100.0, 0.0),
            "rtx3050": (0.06, 75.0, 0.0),
            "colab": (0.02, 150.0, 0.0),
            "kaggle": (0.015, 200.0, 0.0),
            "vast": (0.008, 250.0, 0.45),
            "lambda": (0.005, 300.0, 1.20)
        }

    def estimate_costs(
        self,
        num_steps: int,
        num_seeds: int,
        num_models: int,
        checkpoint_interval: int = 1000,
    ) -> dict[str, Any]:
        """Estimate resources footprint based on campaign variables.

        Returns:
            dict[str, Any]: Estimations payload.
        """
        step_sec, wattage, price = self.backend_params.get(
            self.backend_name, (0.05, 100.0, 0.0)
        )

        total_steps = num_steps * num_seeds * num_models
        total_time_sec = total_steps * step_sec
        gpu_hours = total_time_sec / 3600.0

        # Energy footprint
        energy_kwh = gpu_hours * (wattage / 1000.0)

        # Cloud dollars
        cloud_cost = gpu_hours * price

        # Checkpoints storage: 10M model checkpoint is approx 0.04 GB (40MB)
        checkpoints_per_run = max(1, num_steps // checkpoint_interval)
        total_checkpoints = checkpoints_per_run * num_seeds * num_models
        disk_space_gb = total_checkpoints * 0.04

        # Orchestration reports — PublicationManager generates exactly 17 Markdown reports
        report_count = 17

        return {
            "backend": self.backend_name,
            "estimated_gpu_hours": gpu_hours,
            "estimated_wall_time_hours": gpu_hours,
            "estimated_energy_kwh": energy_kwh,
            "estimated_cloud_cost_usd": cloud_cost,
            "estimated_checkpoints_saved": total_checkpoints,
            "estimated_disk_space_gb": disk_space_gb,
            "estimated_report_count": report_count,
        }
