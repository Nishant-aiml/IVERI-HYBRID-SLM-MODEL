# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Confidence Calibration module for Stage 5 scientific validations.

Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier score, and compiles reliability diagrams.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Computes Expected Calibration Error (ECE) and Brier scores for byte-level outputs."""

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins

    def compute_calibration_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute calibration parameters across vocabulary predictions.

        Args:
            logits: Unnormalized logits of shape (N, V) or (B, S, V).
            labels: Ground truth token indices of shape (N,) or (B, S).

        Returns:
            dict[str, Any]: ECE, MCE, Brier, NLL, and reliability diagram coordinates.
        """
        # Flatten tensors
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Softmax to get probability distribution
        probs = torch.softmax(logits_flat, dim=-1)

        # Max prediction confidence and predicted class
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == labels_flat).float()

        # Expected Calibration Error (ECE) calculation
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_data = []

        n_samples = logits_flat.size(0)

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Samples that fall into the current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean().item()
                confidence_in_bin = confidences[in_bin].mean().item()
                bin_diff = abs(accuracy_in_bin - confidence_in_bin)

                ece += prop_in_bin * bin_diff
                mce = max(mce, bin_diff)

                bin_data.append({
                    "bin_idx": i,
                    "bin_range": (float(bin_lower), float(bin_upper)),
                    "sample_count": int(in_bin.sum().item()),
                    "accuracy": accuracy_in_bin,
                    "confidence": confidence_in_bin,
                })
            else:
                bin_data.append({
                    "bin_idx": i,
                    "bin_range": (float(bin_lower), float(bin_upper)),
                    "sample_count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                })

        # Multi-class Brier Score: Mean squared error of probabilities
        # Brier = (1/N) * sum_i sum_c (prob_ic - target_ic)^2
        # Create one-hot targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, labels_flat.unsqueeze(1), 1.0)
        brier_score = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1)).item()

        # Negative Log Likelihood (NLL)
        loss_fn = torch.nn.CrossEntropyLoss()
        nll = loss_fn(logits_flat, labels_flat).item()

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": brier_score,
            "negative_log_likelihood": nll,
            "reliability_diagram": bin_data,
        }
