# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Learning Rate Generator for Titans Neural Memory.

Generates dynamic, bounded step-wise learning rates and forgetting rates
conditioned on the sequence representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from utils.validation import validate_shape


@register("titans_lr_gen")
class MemoryLearningRateGenerator(BaseModule):
    """Generates bounded learning and forgetting rates for memory updates.

    Projects sequence state vectors to sigmoid-bounded scalar rates to ensure
    numerical stability in the online test-time update loop.
    """

    def __init__(
        self,
        config: IVERIConfig,
        max_lr: float = 0.1,
        max_forget: float = 0.1,
    ) -> None:
        """Initialize the learning rate generator.

        Args:
            config: Configuration object containing model parameters.
            max_lr: Maximum bound for learning rate updates.
            max_forget: Maximum bound for the forgetting factor.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.max_lr = max_lr
        self.max_forget = max_forget

        # Projections to compute raw logits for lr and forget rate
        self.lr_proj = nn.Linear(self.hidden_dim, 1)
        self.forget_proj = nn.Linear(self.hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameter weights to stable small values."""
        # Initialize projections to yield small initial outputs
        nn.init.normal_(self.lr_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.lr_proj.bias, -2.0)  # sig(-2) * 0.1 ≈ 0.01 initial LR
        nn.init.normal_(self.forget_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.forget_proj.bias, -2.0)  # sig(-2) * 0.1 ≈ 0.01 initial forget rate

    def forward(self, x: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Generate bounded learning rate and forget rate.

        Args:
            x: Input sequence tensor of shape (B, S, D) or (B, 1, D).
            **kwargs: Extra arguments.

        Returns:
            Tuple of (learning_rate, forget_rate) tensors of shape (B, S, 1).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="input x")

        # Sigmoid projection and scaling to configure bounds
        lr = torch.sigmoid(self.lr_proj(x)) * self.max_lr
        forget = torch.sigmoid(self.forget_proj(x)) * self.max_forget

        return lr, forget
