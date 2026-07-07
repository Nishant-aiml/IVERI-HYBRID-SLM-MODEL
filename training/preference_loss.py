# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Preference loss calculations for IVERI CORE preference optimization.

Implements standard DPO, IPO, Conservative DPO (label smoothing), and SimPO loss
functions with stable log-sigmoid implementation and NaN guards.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Compute the log probabilities of response bytes.

    Parameters
    ----------
    logits:
        Logits tensor of shape ``(B, S, Vocab)``
    labels:
        Target labels of shape ``(B, S)``
    mask:
        Bool loss mask of shape ``(B, S)`` identifying response segments
    average_log_prob:
        If ``True``, normalize the log probabilities by the response length.
        Required for SimPO loss.

    Returns
    -------
    logps:
        Log probabilities per sample of shape ``(B,)``
    """
    # Shift logits and labels if needed (in our case inputs/targets are already shifted at dataset level)
    # Gather token log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out non-response tokens
    masked = gathered * mask.float()
    
    summed = masked.sum(dim=-1)
    
    if average_log_prob:
        # Avoid division by zero
        lengths = mask.float().sum(dim=-1).clamp(min=1.0)
        return summed / lengths
        
    return summed


class PreferenceLoss(nn.Module):
    """Calculates preference optimization loss and rewards for chosen/rejected pairs.

    Supports standard DPO, Conservative DPO, IPO, and SimPO.

    Parameters
    ----------
    algorithm:
        One of ``"dpo"``, ``"conservative_dpo"``, ``"ipo"``, ``"simpo"``.
    beta:
        DPO preference temperature scale parameter. Default ``0.1``.
    label_smoothing:
        Smoothing parameter for conservative DPO. Default ``0.1``.
    ipo_gamma:
        Margin threshold parameter for IPO / SimPO. Default ``2.0``.
    """

    def __init__(
        self,
        algorithm: str = "dpo",
        beta: float = 0.1,
        label_smoothing: float = 0.1,
        ipo_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.algorithm = algorithm.lower()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo_gamma = ipo_gamma

        valid = {"dpo", "conservative_dpo", "ipo", "simpo"}
        if self.algorithm not in valid:
            raise ValueError(f"Unknown preference algorithm: {self.algorithm}")

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor | None = None,
        reference_rejected_logps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the preference loss and reward metrics.

        Parameters
        ----------
        policy_chosen_logps:
            Chosen response logps under the active policy model. Shape ``(B,)``.
        policy_rejected_logps:
            Rejected response logps under the active policy model. Shape ``(B,)``.
        reference_chosen_logps:
            Chosen response logps under the reference model. Shape ``(B,)`` or ``None`` (for SimPO).
        reference_rejected_logps:
            Rejected response logps under the reference model. Shape ``(B,)`` or ``None`` (for SimPO).

        Returns
        -------
        loss:
            Scalar preference loss tensor.
        chosen_rewards:
            Reward values for chosen responses of shape ``(B,)``.
        rejected_rewards:
            Reward values for rejected responses of shape ``(B,)``.
        """
        # Calculate log ratios
        policy_log_ratios = policy_chosen_logps - policy_rejected_logps

        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            reference_log_ratios = reference_chosen_logps - reference_rejected_logps
            log_ratios = policy_log_ratios - reference_log_ratios
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        else:
            # SimPO / Reference-free algorithms
            log_ratios = policy_log_ratios
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps

        # Compute specific loss functions
        if self.algorithm == "dpo":
            logits = self.beta * log_ratios
            loss = -F.logsigmoid(logits).mean()

        elif self.algorithm == "conservative_dpo":
            # DPO with label smoothing:
            # L = -[(1 - label_smoothing) * logsigmoid(beta * log_ratios) + label_smoothing * logsigmoid(-beta * log_ratios)]
            logits = self.beta * log_ratios
            loss = -(
                (1.0 - self.label_smoothing) * F.logsigmoid(logits)
                + self.label_smoothing * F.logsigmoid(-logits)
            ).mean()

        elif self.algorithm == "ipo":
            # IPO loss: (policy_log_ratios - reference_log_ratios - 1/(2*beta))**2
            # Here log_ratios = policy_log_ratios - reference_log_ratios
            # ipo_gamma can also act as the offset, but standard is 1 / (2 * beta)
            # To maintain compatibility, we use 1 / (2 * beta) as target offset if beta > 0
            offset = 1.0 / (2.0 * self.beta) if self.beta > 0 else 0.5
            loss = ((log_ratios - offset) ** 2).mean()

        elif self.algorithm == "simpo":
            # SimPO: -logsigmoid(beta * policy_log_ratios_normalized - gamma)
            # Note: log_ratios passed here should represent length-normalized chosen/rejected logprobs
            logits = self.beta * log_ratios - self.ipo_gamma
            loss = -F.logsigmoid(logits).mean()

        else:
            raise ValueError(f"Unknown preference algorithm: {self.algorithm}")

        # ── NaN / Inf Guard ──────────────────────────────────────────────────
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN or Inf detected in preference loss! Applying zero guard.")
            # Create a zero tensor matching the device of loss
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)

        return loss, chosen_rewards, rejected_rewards
