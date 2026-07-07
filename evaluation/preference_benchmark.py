# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Preference benchmark suite for Phase 3.4 Preference Optimization.

Runs offline evaluations over UltraFeedback or Tulu preference validation subsets,
computing win rates, average reward margins, and preference accuracy.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.preference_loss import compute_logps

logger = logging.getLogger(__name__)


def compute_histogram_quantiles(values: list[float]) -> dict[str, float]:
    """Compute standard histogram quantiles for diagnostics (Feedback-#3)."""
    if not values:
        return {
            "min": 0.0, "10%": 0.0, "25%": 0.0, "50%": 0.0,
            "75%": 0.0, "90%": 0.0, "max": 0.0, "std": 0.0, "mean": 0.0
        }
    arr = np.array(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "10%": float(np.percentile(arr, 10)),
        "25%": float(np.percentile(arr, 25)),
        "50%": float(np.percentile(arr, 50)),
        "75%": float(np.percentile(arr, 75)),
        "90%": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "mean": float(np.mean(arr)),
    }


class PreferenceBenchmarkRunner:
    """Orchestrates validation evaluations on preference test/val datasets."""

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module | None,
        beta: float = 0.1,
        average_log_prob: bool = False,
    ) -> None:
        self.model = model
        self.reference_model = reference_model
        self.beta = beta
        self.average_log_prob = average_log_prob

    def run_evaluation(
        self,
        dataloader: DataLoader,
        device: torch.device,
        precision_handler: Any,
    ) -> dict[str, Any]:
        """Execute evaluation over the dataloader.

        Returns a dictionary of aggregated preference metrics and reward histograms.
        """
        self.model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        chosen_logps_list = []
        rejected_logps_list = []
        ref_chosen_logps_list = []
        ref_rejected_logps_list = []

        chosen_rewards_list = []
        rejected_rewards_list = []
        margins_list = []

        n_wins = 0
        n_ref_wins = 0
        n_pref_correct = 0
        total_samples = 0

        # Run under no_grad context
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch (6 elements returned by PreferenceByteDataset)
                c_x, c_y, c_mask, r_x, r_y, r_mask = batch

                c_x = c_x.to(device, non_blocking=True)
                c_y = c_y.to(device, non_blocking=True)
                c_mask = c_mask.to(device, non_blocking=True)
                r_x = r_x.to(device, non_blocking=True)
                r_y = r_y.to(device, non_blocking=True)
                r_mask = r_mask.to(device, non_blocking=True)

                B = c_x.size(0)
                total_samples += B

                # Concatenate inputs for a single forward pass
                input_ids = torch.cat([c_x, r_x], dim=0)
                
                with precision_handler.autocast_context():
                    outputs = self.model(input_ids, return_dict=True)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    
                    # Split back to chosen and rejected
                    chosen_logits, rejected_logits = logits.chunk(2, dim=0)

                    # Compute log probabilities under policy
                    c_logps = compute_logps(chosen_logits, c_y, c_mask, average_log_prob=self.average_log_prob)
                    r_logps = compute_logps(rejected_logits, r_y, r_mask, average_log_prob=self.average_log_prob)

                # Reference log probabilities
                if self.reference_model is not None:
                    # Load reference model device
                    ref_device = next(self.reference_model.parameters()).device
                    ref_input_ids = input_ids.to(ref_device)
                    
                    with precision_handler.autocast_context():
                        ref_outputs = self.reference_model(ref_input_ids, return_dict=True)
                        ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs
                        ref_chosen_logits, ref_rejected_logits = ref_logits.chunk(2, dim=0)

                        ref_c_logps = compute_logps(ref_chosen_logits, c_y.to(ref_device), c_mask.to(ref_device), average_log_prob=self.average_log_prob).to(device)
                        ref_r_logps = compute_logps(ref_rejected_logits, r_y.to(ref_device), r_mask.to(ref_device), average_log_prob=self.average_log_prob).to(device)
                else:
                    ref_c_logps = torch.zeros_like(c_logps)
                    ref_r_logps = torch.zeros_like(r_logps)

                # Compute rewards and margins
                c_rewards = self.beta * (c_logps - ref_c_logps)
                r_rewards = self.beta * (r_logps - ref_r_logps)
                margins = c_rewards - r_rewards

                # Accumulate lists
                chosen_logps_list.extend(c_logps.cpu().tolist())
                rejected_logps_list.extend(r_logps.cpu().tolist())
                ref_chosen_logps_list.extend(ref_c_logps.cpu().tolist())
                ref_rejected_logps_list.extend(ref_r_logps.cpu().tolist())

                chosen_rewards_list.extend(c_rewards.cpu().tolist())
                rejected_rewards_list.extend(r_rewards.cpu().tolist())
                margins_list.extend(margins.cpu().tolist())

                # Win counters
                # Policy preferred choice over rejection
                n_wins += (c_logps > r_logps).sum().item()
                # Reference preferred choice over rejection
                n_ref_wins += (ref_c_logps > ref_r_logps).sum().item()
                # Relative preference (policy margin > reference margin) matches human label
                n_pref_correct += ((c_logps - ref_c_logps) > (r_logps - ref_r_logps)).sum().item()

        if total_samples == 0:
            return {"benchmark/win_rate": 0.0}

        # Calculate statistics
        policy_win_rate = n_wins / total_samples
        reference_win_rate = n_ref_wins / total_samples
        preference_accuracy = n_pref_correct / total_samples
        avg_margin = float(np.mean(margins_list))

        # Reward Histograms (Feedback-#3 & Component 11)
        chosen_hist = compute_histogram_quantiles(chosen_rewards_list)
        rejected_hist = compute_histogram_quantiles(rejected_rewards_list)
        margin_hist = compute_histogram_quantiles(margins_list)

        # Average KL estimation: mean(policy_chosen_logps - reference_chosen_logps)
        avg_kl = float(np.mean([c - r for c, r in zip(chosen_logps_list, ref_chosen_logps_list)]))

        metrics = {
            "benchmark/samples": total_samples,
            "benchmark/win_rate": policy_win_rate,
            "benchmark/reference_win_rate": reference_win_rate,
            "benchmark/preference_accuracy": preference_accuracy,
            "benchmark/average_reward_margin": avg_margin,
            "benchmark/average_kl": avg_kl,
            "benchmark/chosen_avg_reward": float(np.mean(chosen_rewards_list)),
            "benchmark/rejected_avg_reward": float(np.mean(rejected_rewards_list)),
            
            # Diagnostic Histograms (Feedback-#3 & Component 11)
            "histograms/chosen": chosen_hist,
            "histograms/rejected": rejected_hist,
            "histograms/margin": margin_hist,
        }

        logger.info(
            "Preference Benchmark completed: Win Rate=%.4f (Ref: %.4f), Preference Acc=%.4f, Avg Margin=%.4f",
            policy_win_rate,
            reference_win_rate,
            preference_accuracy,
            avg_margin,
        )

        return metrics
