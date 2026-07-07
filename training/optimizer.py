# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Optimizer and parameter grouping utilities for IVERI CORE.

Handles parameter decay grouping (excluding biases, normalizations, and scale terms from decay)
and configures the AdamW optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Create a parameter-grouped AdamW optimizer.

    Excludes 1D variables, biases, normalizations, and scaling parameters
    from weight decay.

    Args:
        model: Model whose parameters will be optimized.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        adam_beta1: Beta1 coefficient for Adam.
        adam_beta2: Beta2 coefficient for Adam.
        adam_eps: Epsilon coefficient for Adam.

    Returns:
        Configured torch.optim.AdamW optimizer.
    """
    # Fetch parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    # Filter into decayed and non-decayed groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude:
        # - Biases (ends with '.bias')
        # - Normalization scale (RMSNorm, LayerNorm weights, e.g. '.weight' with 1D shape)
        # - Scale parameters / embeddings with 1D shape
        # Rule of thumb: if it has shape <= 1 dimension, do not decay.
        if param.dim() <= 1 or name.endswith(".bias") or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Validate that all params are accounted for
    total_grouped = len(decay_params) + len(no_decay_params)
    assert total_grouped == len(params), (
        f"Grouping error: grouped={total_grouped} params, " f"expected={len(params)} params."
    )

    # Package into group dictionaries
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Initialize AdamW
    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
