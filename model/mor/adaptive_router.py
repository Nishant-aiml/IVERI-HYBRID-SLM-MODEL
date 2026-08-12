# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Adaptive Mixture of Recursions (MoR v0.92) Controller.

Implements adaptive depth gating:
  - Bypasses recursion (depth=0) when sequence length is short (seq_len < seq_len_threshold, default 128).
  - Bypasses recursion when token entropy is low (entropy < entropy_threshold, default 0.25).
  - Dynamically routes high-entropy tokens in long sequences to depth in [1, max_depth].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseRouter
from core.registry import register
from utils.validation import validate_shape


@register("adaptive_recursion_depth_router")
class AdaptiveRecursionDepthRouter(BaseRouter):
    """Adaptive Router for Mixture of Recursions (v0.92).

    Bypasses recursion (assigning depth=0) under short-context conditions or low entropy.
    """

    def __init__(
        self,
        config: IVERIConfig,
        seq_len_threshold: int = 128,
        entropy_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.max_depth = config.model.max_recursion_depth
        self.seq_len_threshold = seq_len_threshold
        self.entropy_threshold = entropy_threshold

    def route(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_shape(x, (-1, -1, self.hidden_dim), name="x")
        batch, seq_len, _ = x.shape

        # Default depth = 0 for short context
        if seq_len < self.seq_len_threshold or entropy is None:
            depths = torch.zeros((batch, seq_len, 1), dtype=torch.long, device=x.device)
            dispatch_weights = torch.ones_like(depths, dtype=torch.float32)
            return dispatch_weights, depths

        if entropy.ndim == 2:
            entropy = entropy.unsqueeze(-1)
        validate_shape(entropy, (batch, seq_len, 1), name="entropy")

        # Low-entropy tokens get depth 0; high-entropy tokens map to [1, max_depth]
        entropy_mask = (entropy >= self.entropy_threshold).long()
        mapped_depths = 1 + torch.floor(entropy * (self.max_depth - 1)).long()
        mapped_depths = torch.clamp(mapped_depths, min=1, max=self.max_depth)
        depths = mapped_depths * entropy_mask

        dispatch_weights = torch.ones_like(depths, dtype=torch.float32)
        return dispatch_weights, depths

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        entropy = kwargs.get("entropy")
        if entropy is not None and not isinstance(entropy, torch.Tensor):
            raise TypeError("entropy must be a torch.Tensor")

        _, depths = self.route(x, entropy=entropy)
        return depths
