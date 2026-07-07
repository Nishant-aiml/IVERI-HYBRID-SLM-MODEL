# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Memory Updater for Titans Neural Memory.

Implements the official momentum-based gradient updates with forgetting for the long-term
neural memory module parameters.
"""

from __future__ import annotations

import torch

from configs.base_config import IVERIConfig


class MemoryUpdater:
    """Updates neural memory parameters using momentum and adaptive decay.

    Enforces the frozen Titans learning rule:
    S_t = η * S_{t-1} - θ_t * ∇_W l
    W_t = (1 - α_t) * W_{t-1} + S_t

    This updater maintains full differentiability to allow autograd to track
    gradients through multiple sequential update cycles.
    """

    def __init__(self, config: IVERIConfig, momentum: float = 0.9) -> None:
        """Initialize the memory updater.

        Args:
            config: Configuration object containing model parameters.
            momentum: Momentum factor (eta) for surprise accumulation.
        """
        self.config = config
        self.momentum = momentum

    def update(
        self,
        weights: list[torch.Tensor],
        grads: list[torch.Tensor | None],
        surprise_state: list[torch.Tensor],
        lr: torch.Tensor,
        forget: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Perform a differentiable functional parameter update step.

        Args:
            weights: List of current parameter tensors, each of shape (B, ...).
            grads: List of gradient tensors for the parameters, each of shape (B, ...).
            surprise_state: List of surprise/momentum accumulator tensors of shape (B, ...).
            lr: Dynamic step size of shape (B, 1, 1).
            forget: Dynamic forgetting rate of shape (B, 1, 1).

        Returns:
            Tuple of (new_weights, new_surprise_state) lists of tensors.
        """
        new_weights = []
        new_surprise_state = []

        for w, g, s in zip(weights, grads, surprise_state, strict=False):
            # Ensure grads are not None and match weight shape
            if g is None:
                # Fallback in case gradient is None (e.g. no dependency on some weights)
                g = torch.zeros_like(w)

            # Clip gradients to prevent NaN/explosion
            g = torch.clamp(g, min=-10.0, max=10.0)

            # S_t = η * S_{t-1} - θ_t * g_t
            # lr is (B, 1, 1) and will broadcast over the batch dimension
            s_new = self.momentum * s - lr * g
            s_new = torch.clamp(s_new, min=-1.0, max=1.0)

            # W_t = (1 - α_t) * W_{t-1} + S_t
            w_new = (1.0 - forget) * w + s_new

            new_weights.append(w_new)
            new_surprise_state.append(s_new)

        return new_weights, new_surprise_state
