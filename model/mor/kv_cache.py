# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mixture of Recursions (MoR) Selective Key-Value Cache.

Implements the key-value cache manager that selectively saves representation states
only for active computation recursion passes, reducing VRAM bloat.
"""

from __future__ import annotations

import torch


class SelectiveKVCache:
    """Manager for selective key-value caching in Mixture of Recursions.

    Only updates the cache state for positions/batches indicated as active by the
    boolean mask, ensuring that inactive steps do not write to or bloat the cache.
    """

    def __init__(self) -> None:
        """Initialize the selective key-value cache."""
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        """Clear all stored key and value caches."""
        self.k_cache = None
        self.v_cache = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache states for active elements.

        Assumes standard attention dimensions: (B, H, S, D_head) or (B, S, H, D_head).
        Also supports (B, S, D) general SSM state configurations.

        Args:
            new_k: New projected keys tensor.
            new_v: New projected values tensor.
            active_mask: Boolean mask of active elements, shape (B, S) or (B, S, 1).

        Returns:
            A tuple of the complete, updated (keys, values) cache tensors.
        """
        # Ensure active_mask matches key sequence length dimension (often index 1 or 2)
        # Determine sequence dimension: typically index -2 in 4D (B, H, S, D) or index 1 in 3D (B, S, D)
        seq_dim = 1 if new_k.ndim == 3 else 2

        if active_mask.ndim == 3:
            active_mask = active_mask.squeeze(-1)

        # Standardize mask shape to match new_k's shape for broadcast/where operations
        # Mask shape is (B, S). We unsqueeze dimensions to match new_k shape (e.g. B, 1, S, 1 or B, S, 1)
        mask_expanded = active_mask
        if new_k.ndim == 4:
            # (B, H, S, D) -> mask must be (B, 1, S, 1)
            mask_expanded = active_mask.unsqueeze(1).unsqueeze(-1)
        elif new_k.ndim == 3:
            # (B, S, D) -> mask must be (B, S, 1)
            mask_expanded = active_mask.unsqueeze(-1)

        # If cache is empty, initialize it
        if self.k_cache is None or self.v_cache is None:
            # Initialize cache only for the active items, padding/zeroing out inactive ones
            self.k_cache = torch.where(mask_expanded, new_k, torch.zeros_like(new_k))
            self.v_cache = torch.where(mask_expanded, new_v, torch.zeros_like(new_v))
        else:
            # Append along sequence dimension
            # For incremental decode (S=1), new_k has seq_len=1. We append to cache.
            # If active, append the actual new value; if inactive, we copy the last state or append zeros
            k_to_append = torch.where(mask_expanded, new_k, torch.zeros_like(new_k))
            v_to_append = torch.where(mask_expanded, new_v, torch.zeros_like(new_v))

            self.k_cache = torch.cat([self.k_cache, k_to_append], dim=seq_dim)
            self.v_cache = torch.cat([self.v_cache, v_to_append], dim=seq_dim)

        return self.k_cache, self.v_cache
