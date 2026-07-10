# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mixture of Experts Container.

Implements allocation, parallel sparse execution, and token capacity dropping
policies based on:

    *GShard: Scaling Giant Models with Conditional Computation*
    Lepikhin et al. (2020). arXiv:2006.16668
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from model.swiglu import SwiGLUFFN


@register("moe_experts")
class MoEExperts(BaseModule):
    """Mixture of Experts layer running a set of SwiGLUFFN networks.

    Splits and dispatches tokens to their target experts, enforcing capacity
    limits (tokens exceeding capacity bypass FFN computation).
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the expert container.

        Parameters
        ----------
        config : IVERIConfig
            Project configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.num_experts = config.model.num_experts
        self.num_active_experts = config.model.num_active_experts

        # Configurable capacity factor with fallback
        self.capacity_factor = float(getattr(config.model, "moe_capacity_factor", 1.25))

        # Instantiate expert networks (reusing SwiGLUFFN from Phase 1.1)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(dim=self.hidden_dim, bias=False) for _ in range(self.num_experts)]
        )

    def reset_parameters(self) -> None:
        """Reinitialize all expert subnetworks."""
        for expert in self.experts:
            expert_module = typing.cast(SwiGLUFFN, expert)
            expert_module.reset_parameters()

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run sparse execution over active experts with capacity checks.

        Parameters
        ----------
        x : torch.Tensor
            Input hidden representation tensor of shape ``(B, S, D)`` or ``(N, D)``.
        weights : torch.Tensor
            Softmax gating weights of shape ``(B, S, K)`` or ``(N, K)``.
        indices : torch.Tensor
            Target expert indices of shape ``(B, S, K)`` or ``(N, K)``.
        **kwargs : object
            Ignored; accepted for signature compatibility.

        Returns
        -------
        output : torch.Tensor
            Combined expert output tensor of shape ``(B, S, D)`` or ``(N, D)``.
        metrics : dict of str to float
            Quality indicators: capacity limits, token drops, and active expert counts.
        """
        orig_shape = x.shape
        if len(orig_shape) == 3:
            b, s, d = orig_shape
            # Use actual K from the inputs (not self.num_active_experts from config),
            # as callers may pass K != config default.
            actual_k = weights.shape[-1]
            x_flat = x.view(-1, d)
            weights_flat = weights.view(-1, actual_k)
            indices_flat = indices.view(-1, actual_k)
        else:
            actual_k = weights.shape[-1]
            x_flat = x
            weights_flat = weights
            indices_flat = indices

        num_tokens = x_flat.shape[0]
        # Use actual K from the flattened inputs, not self.num_active_experts from config.
        # The caller may pass more or fewer active experts than the config default.
        actual_k = indices_flat.shape[1]
        total_dispatches = num_tokens * actual_k

        # 1. Calculate expert capacity threshold
        capacity = int(
            math.ceil(
                (num_tokens * actual_k / self.num_experts) * self.capacity_factor
            )
        )

        # Output accumulation buffer
        output_flat = torch.zeros_like(x_flat)

        # Token & Rank index grids for alignment mapping
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(-1)
            .expand(-1, actual_k)
        )
        rank_indices = (
            torch.arange(actual_k, device=x.device)
            .unsqueeze(0)
            .expand(num_tokens, -1)
        )

        dropped_dispatches = 0
        active_experts_count = 0
        executed_tokens = 0

        # 2. Iterate through experts and run sparse computation
        for e in range(self.num_experts):
            mask = indices_flat == e
            tokens_e = token_indices[mask]
            ranks_e = rank_indices[mask]

            num_routed = tokens_e.shape[0]
            if num_routed > 0:
                active_experts_count += 1

                # Enforce capacity capping; excess tokens are dropped
                if num_routed > capacity:
                    dropped_dispatches += num_routed - capacity
                    tokens_e = tokens_e[:capacity]
                    ranks_e = ranks_e[:capacity]

                actual_routed = tokens_e.shape[0]
                executed_tokens += actual_routed

                if actual_routed > 0:
                    # Gather allocated input tokens
                    input_e = x_flat[tokens_e]  # Shape (actual_routed, D)

                    # Compute expert output
                    output_e = self.experts[e](input_e)  # Shape (actual_routed, D)

                    # Multiply by gating weights
                    weights_e = weights_flat[tokens_e, ranks_e].unsqueeze(-1)
                    weighted_output_e = (output_e * weights_e).to(output_flat.dtype)

                    # Accumulate outputs back
                    output_flat.index_add_(0, tokens_e, weighted_output_e)

        # 3. Reshape output back if input was 3D
        output = output_flat.view(b, s, -1) if len(orig_shape) == 3 else output_flat

        # 4. Formulate metrics
        overflow_pct = (
            100.0 * (dropped_dispatches / total_dispatches) if total_dispatches > 0 else 0.0
        )
        # FLOP savings: actual tokens executed vs. dense baseline executions (num_tokens * E)
        flop_savings_pct = (
            100.0 * (1.0 - (executed_tokens / (num_tokens * self.num_experts)))
            if num_tokens > 0
            else 0.0
        )

        metrics = {
            "capacity": float(capacity),
            "dropped_tokens": float(dropped_dispatches),
            "overflow_pct": overflow_pct,
            "active_experts": float(active_experts_count),
            "flop_savings_pct": flop_savings_pct,
        }

        return output, metrics

    def extra_repr(self) -> str:
        """Return representation details."""
        return (
            f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"capacity_factor={self.capacity_factor}"
        )
