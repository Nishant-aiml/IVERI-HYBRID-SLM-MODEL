# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Patcher for the Byte Latent Transformer (BLT).

Implements deterministic boundary map generation combining entropy threshold signals
and minimum/maximum patch length structural constraints.
"""

from __future__ import annotations

import torch

from configs.base_config import IVERIConfig
from utils.validation import validate_shape


class DynamicPatcher:
    """Computes deterministic boundary maps grouping raw byte sequences into patches."""

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the dynamic patcher.

        Args:
            config: Configuration object containing model parameters.
        """
        self.config = config
        self.patch_size_min = config.model.blt.patch_size_min
        self.patch_size_max = config.model.blt.patch_size_max
        self.entropy_threshold = config.model.blt.entropy_threshold

    def compute_boundaries(
        self,
        raw_bytes: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the boolean boundary map for the input bytes and entropy scores.

        Deterministic mapping. Boundary is True at index 0 and whenever:
        - The current patch length reaches patch_size_max.
        - The entropy value at that position exceeds entropy_threshold AND the current
          patch length is at least patch_size_min.

        Args:
            raw_bytes: Input byte sequence of shape (B, S).
            entropy: Normalized entropy scores of shape (B, S, 1) or (B, S).

        Returns:
            Boolean boundary map of shape (B, S) where True indicates patch starts.
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        batch_size, seq_len = raw_bytes.shape

        if seq_len == 0:
            return torch.zeros(batch_size, 0, device=raw_bytes.device, dtype=torch.bool)

        # Standardize entropy shape to (B, S)
        if entropy.ndim == 3:
            entropy = entropy.squeeze(-1)
        validate_shape(entropy, (batch_size, seq_len), name="entropy")

        # Initialize boundary map. Index 0 is always True (first patch starts)
        boundary_map = torch.zeros(batch_size, seq_len, device=raw_bytes.device, dtype=torch.bool)
        boundary_map[:, 0] = True

        # Keep track of current patch lengths per batch item
        current_len = torch.ones(batch_size, device=raw_bytes.device, dtype=torch.long)

        # Traverse the sequence sequentially to enforce length constraints deterministically
        for i in range(1, seq_len):
            # Check length constraint triggers
            max_len_triggered = current_len >= self.patch_size_max
            entropy_triggered = (entropy[:, i] >= self.entropy_threshold) & (
                current_len >= self.patch_size_min
            )

            boundary_today = max_len_triggered | entropy_triggered
            boundary_map[:, i] = boundary_today

            # Increment lengths where boundary is False, reset to 1 where boundary is True
            current_len = torch.where(boundary_today, torch.ones_like(current_len), current_len + 1)

        validate_shape(boundary_map, (batch_size, seq_len), name="boundary_map")
        return boundary_map
