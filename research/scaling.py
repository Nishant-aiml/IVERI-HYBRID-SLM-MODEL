# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Scaling Law Analysis module fitting power laws for parameters, compute, and validation loss."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class ScalingAnalyzer:
    """Estimates and fits scaling law exponents for parameters and training FLOPs."""

    def __init__(self) -> None:
        pass

    def fit_power_law(self, x_values: list[float], y_values: list[float]) -> dict[str, float]:
        """Fit a power-law relationship: Y = a * X^(-b).

        Uses linear least squares regression on log-transformed variables:
        log(Y) = log(a) - b * log(X)
        """
        if len(x_values) < 2:
            logger.warning("At least 2 data points required to fit scaling laws. Returning default mock fits.")
            return {"a": 3.5, "b": 0.08, "r_squared": 0.0}

        # Log transform
        log_x = [math.log(x) for x in x_values]
        log_y = [math.log(y) for y in y_values]

        n = len(x_values)
        sum_x = sum(log_x)
        sum_y = sum(log_y)
        sum_xx = sum(x ** 2 for x in log_x)
        sum_xy = sum(x * y for x, y in zip(log_x, log_y, strict=True))

        # Linear regression slope (m = -b) and intercept (c = log(a))
        # y = m * x + c
        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-9:
            return {"a": 3.5, "b": 0.08, "r_squared": 0.0}

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # Extract parameters
        a = math.exp(intercept)
        b = -slope

        # Compute R-squared correlation coefficient
        mean_y = sum(log_y) / n
        total_ss = sum((y - mean_y) ** 2 for y in log_y)
        residual_ss = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y, strict=True))

        r_squared = 1.0 - (residual_ss / total_ss) if total_ss > 0.0 else 1.0

        return {
            "a": a,
            "b": b,
            "r_squared": r_squared,
        }

    def predict_loss(self, size: float, fit_params: dict[str, float]) -> float:
        """Predict model validation loss given size (parameters or FLOPs) using fitted parameters."""
        a = fit_params.get("a", 3.5)
        b = fit_params.get("b", 0.08)
        return a * (size ** (-b))

    def validate_predictions(
        self,
        actual_sizes: list[float],
        actual_losses: list[float],
        fit_params: dict[str, float],
    ) -> dict[str, float]:
        """Compare actual measurements vs scaling law predictions.

        Calculates MAPE and RMSE validation errors.
        """
        predictions = [self.predict_loss(size, fit_params) for size in actual_sizes]

        # Compute root mean squared error
        mse = sum((act - pred) ** 2 for act, pred in zip(actual_losses, predictions, strict=True)) / len(actual_losses)
        rmse = math.sqrt(mse)

        # Compute mean absolute percentage error
        mape = sum(abs(act - pred) / act for act, pred in zip(actual_losses, predictions, strict=True)) / len(actual_losses)

        return {
            "rmse": rmse,
            "mape": mape,
        }
