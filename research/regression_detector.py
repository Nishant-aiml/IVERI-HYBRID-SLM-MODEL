# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Regression Detector validating new runs against the Golden Checkpoint."""

from __future__ import annotations

import logging
from typing import Any

from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class RegressionDetector:
    """Monitors metrics against golden reference configurations, logging severity grades."""

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def check_for_regression(
        self,
        new_metrics: dict[str, float],
        golden_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Audits evaluation metrics against golden references.

        Checks: Loss, Perplexity, TTFT, Decode speed, VRAM, Energy/token, ECE, HumanEval, SFT.
        Assigns: INFO, WARNING, CRITICAL, or FATAL severity.
        """
        results: dict[str, Any] = {}
        highest_severity = "INFO"

        # List of tracked metrics and their preferred change direction
        # (lower_is_better=True: regression is when value increases)
        metric_configs = {
            "loss": {"lower_is_better": True, "warn_thresh": 0.02, "crit_thresh": 0.05, "fatal_thresh": 0.10},
            "perplexity": {"lower_is_better": True, "warn_thresh": 0.02, "crit_thresh": 0.05, "fatal_thresh": 0.10},
            "ttft_sec": {"lower_is_better": True, "warn_thresh": 0.10, "crit_thresh": 0.25, "fatal_thresh": 0.50},
            "decode_speed_tps": {"lower_is_better": False, "warn_thresh": -0.05, "crit_thresh": -0.15, "fatal_thresh": -0.25},
            "vram_peak_mb": {"lower_is_better": True, "warn_thresh": 0.05, "crit_thresh": 0.15, "fatal_thresh": 0.30},
            "energy_per_token_j": {"lower_is_better": True, "warn_thresh": 0.05, "crit_thresh": 0.15, "fatal_thresh": 0.30},
            "calibration_ece": {"lower_is_better": True, "warn_thresh": 0.05, "crit_thresh": 0.15, "fatal_thresh": 0.30},
            "humaneval_pass_rate": {"lower_is_better": False, "warn_thresh": -0.05, "crit_thresh": -0.15, "fatal_thresh": -0.25},
            "instruction_score": {"lower_is_better": False, "warn_thresh": -0.05, "crit_thresh": -0.15, "fatal_thresh": -0.25},
        }

        for key, config in metric_configs.items():
            new_val = new_metrics.get(key)
            gold_val = golden_metrics.get(key)

            if new_val is None or gold_val is None:
                results[key] = {"status": "SKIPPED_MISSING_DATA", "severity": "INFO"}
                continue

            # Compute relative change
            if gold_val == 0:
                change = 0.0
            else:
                change = (new_val - gold_val) / gold_val

            # Determine if it's a regression based on lower_is_better
            is_regression = False
            rel_change = change
            if config["lower_is_better"]:
                if rel_change > 0:
                    is_regression = True
            else:
                # If higher is better, relative change should be positive.
                # If negative, it is a regression.
                if rel_change < 0:
                    is_regression = True
                # Invert change representation for threshold check
                rel_change = -rel_change

            severity = "INFO"
            if is_regression:
                if rel_change >= config["fatal_thresh"]:
                    severity = "FATAL"
                elif rel_change >= config["crit_thresh"]:
                    severity = "CRITICAL"
                elif rel_change >= config["warn_thresh"]:
                    severity = "WARNING"

            # Update highest severity
            severity_ranks = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "FATAL": 3}
            if severity_ranks[severity] > severity_ranks[highest_severity]:
                highest_severity = severity

            results[key] = {
                "golden_value": gold_val,
                "new_value": new_val,
                "percentage_change": change * 100.0,
                "is_regression": is_regression,
                "severity": severity,
            }

        # Log or alert based on highest severity
        if highest_severity == "FATAL":
            logger.error("Catastrophic regression detected! Severity level: FATAL.")
        elif highest_severity == "CRITICAL":
            logger.error("Significant regression detected. Severity level: CRITICAL.")
        elif highest_severity == "WARNING":
            logger.warning("Mild regression detected. Severity level: WARNING.")

        return {
            "highest_severity": highest_severity,
            "metrics": results,
        }
