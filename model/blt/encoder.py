# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Local Encoder for the Byte Latent Transformer (BLT).

Implements the latent patch projection mapping raw bytes to patch-level
vector representations using local within-patch attention and mean pooling.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE
from core.interfaces import BaseEncoder
from core.registry import register
from utils.validation import validate_shape


@register("blt_encoder")
class BLTByteEncoder(BaseEncoder):
    """Encodes variable-length byte patches into fixed-size latent vectors.

    Uses within-patch self-attention over byte representations, averages them
    using mean pooling over patch boundaries, and projects to the latent dimension.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the BLT byte encoder.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim

        # Byte embedding and local attention layer
        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Output projection to latent space
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw bytes into a latent representation.

        Raises:
            NotImplementedError: Always, because BLT requires a boundary map.
        """
        raise NotImplementedError(
            "BLTByteEncoder requires a boundary_map. Use encode_with_boundaries or forward."
        )

    def encode_with_boundaries(
        self,
        raw_bytes: torch.Tensor,
        boundary_map: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw bytes into latent patch representations using boundaries.

        Args:
            raw_bytes: Raw byte indices tensor of shape (B, S).
            boundary_map: Boolean patch boundary map of shape (B, S).

        Returns:
            Latent patch representations tensor of shape (B, P_max, D).
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        validate_shape(boundary_map, raw_bytes.shape, name="boundary_map")
        batch_size, seq_len = raw_bytes.shape

        if seq_len == 0:
            return torch.zeros(
                batch_size, 0, self.hidden_dim, device=raw_bytes.device, dtype=torch.float32
            )

        # 1. Embed raw bytes
        embeddings = self.embed(raw_bytes)  # (B, S, D)

        # 2. Assign patch IDs to each byte using cumulative sum of boundary_map
        # Shape: (B, S) - patch_ids[b, i] represents the 0-indexed patch number of byte i
        patch_ids = torch.cumsum(boundary_map.long(), dim=-1) - 1

        # Determine maximum number of patches in this batch
        p_max = int(patch_ids.max().item() + 1)

        # 3. Within-patch causal self-attention
        # Mask True = disallowed. Block cross-patch keys and future byte positions (j > i).
        # Shape: (B, S, S)
        same_patch = patch_ids.unsqueeze(-1) == patch_ids.unsqueeze(-2)
        pos = torch.arange(seq_len, device=raw_bytes.device)
        future_key = pos.view(1, -1, 1) < pos.view(1, 1, -1)  # query i cannot attend to key j>i
        attn_mask = (~same_patch) | future_key

        # Replicate attn_mask to match multi-head attention requirements: (B * num_heads, S, S)
        num_heads = self.attention.num_heads
        # Expand and reshape to (B * num_heads, S, S)
        attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask_expanded = attn_mask_expanded.reshape(batch_size * num_heads, seq_len, seq_len)

        # Apply multi-head self-attention
        attended_embeddings, _ = self.attention(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=attn_mask_expanded,
        )  # (B, S, D)

        # 4. Vectorized Mean Pooling over patches
        # Create pooling matrix M of shape (B, p_max, S)
        # is_patch[b, p, s] is True if byte s belongs to patch p
        patch_indices = torch.arange(p_max, device=raw_bytes.device).view(1, -1, 1)
        is_patch = patch_ids.unsqueeze(1) == patch_indices  # (B, p_max, S)

        # Length of each patch
        patch_lengths = is_patch.sum(dim=-1, keepdim=True)  # (B, p_max, 1)
        # Clamp length to 1 to avoid division by zero on padded patches
        patch_lengths_clamped = torch.clamp(patch_lengths, min=1)

        # Pooling weights: 1/L_p for active positions, 0 otherwise
        M = is_patch.float() / patch_lengths_clamped.float()  # (B, p_max, S)

        # Aggregate: (B, p_max, S) x (B, S, D) -> (B, p_max, D)
        pooled = torch.bmm(M, attended_embeddings)

        # 5. Output projection
        latent_patches = self.proj(pooled)

        validate_shape(latent_patches, (batch_size, p_max, self.hidden_dim), name="latent_patches")
        return typing.cast(torch.Tensor, latent_patches)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward wrapper implementing abstract BaseEncoder.forward contract.

        Args:
            x: Raw input sequence byte indices of shape (B, S).
            **kwargs: Extra arguments including 'boundary_map'.

        Returns:
            Latent patches of shape (B, P_max, D).
        """
        boundary_map_val = kwargs.get("boundary_map")
        if boundary_map_val is None or not isinstance(boundary_map_val, torch.Tensor):
            raise ValueError("boundary_map must be provided as a torch.Tensor in kwargs")

        boundary_map: torch.Tensor = boundary_map_val
        return typing.cast(torch.Tensor, self.encode_with_boundaries(x, boundary_map))
