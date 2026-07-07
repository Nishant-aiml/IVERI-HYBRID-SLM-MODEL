# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mixture of Recursions (MoR) Router.

Implements the recursion depth router mapping input signals (e.g. byte entropy
or projections of hidden states) to token-level computational depth assignments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseRouter
from core.registry import register
from utils.validation import validate_shape


@register("recursion_depth_router")
@register("mor_router")
class RecursionDepthRouter(BaseRouter):
    """Router for assigning recursion depth to sequence elements.

    In production mode (Option C), it directly maps patch entropy to depth
    values in [1, max_depth]. In research mode, it projects the hidden state x
    to logit scores over depths for ablation comparison.
    """

    def __init__(self, config: IVERIConfig, research_mode: bool = False) -> None:
        """Initialize the recursion depth router.

        Args:
            config: Configuration object containing model parameters.
            research_mode: Whether to enable optional learned routing for ablation studies.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.max_depth = config.model.max_recursion_depth
        self.research_mode = research_mode

        # Projection layer used ONLY in research mode for learned routing ablation
        if self.research_mode:
            self.wg = nn.Linear(self.hidden_dim, self.max_depth, bias=False)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learned routing weights (only active in research mode)."""
        if self.research_mode:
            nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)

    def route(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute recursion depth routing decisions.

        Args:
            x: Hidden representation tensor of shape (B, P, D).
            entropy: Optional patch entropy tensor of shape (B, P, 1) or (B, P).

        Returns:
            dispatch_weights: Routing weights of shape (B, P, 1).
            dispatch_indices: Assigned depth indices (0-indexed) of shape (B, P, 1).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="x")
        batch, seq_len, _ = x.shape

        if not self.research_mode:
            # Official Option C Production Mode: direct entropy-to-depth mapping
            if entropy is None:
                raise ValueError(
                    "Official Option C requires an entropy tensor. "
                    "Ensure the official IVERI entropy pipeline is active."
                )

            # Verify input entropy shape (can be B, P or B, P, 1)
            if entropy.ndim == 2:
                entropy = entropy.unsqueeze(-1)
            validate_shape(entropy, (batch, seq_len, 1), name="entropy")

            # Map E_p in [0.0, 1.0] -> depths in [1, max_depth]
            # Formula: 1 + floor(E_p * (max_depth - 1))
            depths = 1 + torch.floor(entropy * (self.max_depth - 1)).long()
            depths = torch.clamp(depths, min=1, max=self.max_depth)

            # Slices and weights conform to BaseRouter shape contract: (B, P, 1)
            dispatch_indices = depths - 1  # Convert to 0-indexed depths
            dispatch_weights = torch.ones_like(dispatch_indices, dtype=torch.float32)
        else:
            # Research Mode (Ablation Baseline): Learned routing projection
            logits = self.wg(x)  # (B, P, max_depth)
            probs = torch.softmax(logits, dim=-1)  # (B, P, max_depth)

            # Select top-1 argmax depth
            if self.training:
                # Add gumbel or simple noise if needed, default argmax for deterministic shapes
                dispatch_indices = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                dispatch_indices = torch.argmax(probs, dim=-1, keepdim=True)

            dispatch_weights = torch.gather(probs, dim=-1, index=dispatch_indices)

        validate_shape(dispatch_weights, (batch, seq_len, 1), name="dispatch_weights")
        validate_shape(dispatch_indices, (batch, seq_len, 1), name="dispatch_indices")
        return dispatch_weights, dispatch_indices

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Route input representations and return depth index tensor.

        Args:
            x: Input representations of shape (B, P, D).
            **kwargs: Extra parameters (e.g. `entropy` tensor).

        Returns:
            Assigned depth indices (0-indexed) of shape (B, P, 1).
        """
        entropy = kwargs.get("entropy")
        if entropy is not None and not isinstance(entropy, torch.Tensor):
            raise TypeError("entropy must be a torch.Tensor")

        _, dispatch_indices = self.route(x, entropy=entropy)
        return dispatch_indices
