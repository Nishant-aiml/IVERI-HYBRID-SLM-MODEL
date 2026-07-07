# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Local Decoder for the Byte Latent Transformer (BLT).

Implements the latent-to-byte projection using cross-attention from local byte
queries to patch latent keys/values, producing logits for next-byte prediction.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE
from core.interfaces import BaseDecoder
from core.registry import register
from utils.validation import validate_shape


@register("blt_decoder")
class BLTByteDecoder(BaseDecoder):
    """Decodes latent patch representations back into byte-level predictions.

    Uses local byte embeddings as queries to cross-attend to patch latent representations,
    then projects the output to next-byte logits of shape (B, S, BYTE_VOCAB_SIZE).
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the BLT byte decoder.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim

        # Byte embedding and cross-attention layer
        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Output projection to byte classes
        self.logits_proj = nn.Linear(self.hidden_dim, BYTE_VOCAB_SIZE)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.logits_proj.weight, mean=0.0, std=0.02)
        if self.logits_proj.bias is not None:
            nn.init.zeros_(self.logits_proj.bias)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back to the output space.

        Raises:
            NotImplementedError: Always, because BLT requires boundary_map and raw_bytes context.
        """
        raise NotImplementedError(
            "BLTByteDecoder requires boundary_map and raw_bytes. Use decode_with_boundaries or forward."
        )

    def decode_with_boundaries(
        self,
        latent_patches: torch.Tensor,
        boundary_map: torch.Tensor,
        raw_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent patches back to byte predictions.

        Args:
            latent_patches: Latent patch representations of shape (B, P_max, D).
            boundary_map: Boolean patch boundary map of shape (B, S).
            raw_bytes: Raw input byte indices tensor of shape (B, S).

        Returns:
            Next-byte prediction logits tensor of shape (B, S, 256).
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        validate_shape(boundary_map, raw_bytes.shape, name="boundary_map")
        batch_size, seq_len = raw_bytes.shape
        p_max = latent_patches.shape[1]
        validate_shape(latent_patches, (batch_size, p_max, self.hidden_dim), name="latent_patches")

        if seq_len == 0:
            return torch.zeros(batch_size, 0, BYTE_VOCAB_SIZE, device=raw_bytes.device, dtype=torch.float32)

        # 1. Embed raw bytes as queries
        byte_embeddings = self.embed(raw_bytes)  # (B, S, D)

        # 2. Build key padding mask for latent patches
        # Assign patch IDs to determine actual patch counts per batch item
        patch_ids = torch.cumsum(boundary_map.long(), dim=-1) - 1
        n_patches = patch_ids[:, -1] + 1  # (B,)

        # key_padding_mask[b, p] is True if patch p is beyond n_patches[b] (i.e. is padding)
        patch_indices = torch.arange(p_max, device=raw_bytes.device).view(1, -1)
        key_padding_mask = patch_indices >= n_patches.unsqueeze(-1)  # (B, p_max)

        # Causal patch visibility: byte at position s may attend only to patches with end <= s.
        positions = torch.arange(seq_len, device=raw_bytes.device).view(1, -1).expand(batch_size, -1)
        is_patch = patch_ids.unsqueeze(1) == patch_indices.unsqueeze(-1)  # (B, p_max, S)
        patch_ends = (is_patch.float() * positions.unsqueeze(1)).max(dim=-1).values.long()  # (B, p_max)
        query_pos = torch.arange(seq_len, device=raw_bytes.device).view(1, seq_len, 1)
        patch_ends_exp = patch_ends.unsqueeze(1)  # (B, 1, p_max)
        cannot_attend = (patch_ends_exp > query_pos) | key_padding_mask.unsqueeze(1)

        num_heads = self.cross_attn.num_heads
        attn_mask_expanded = (
            cannot_attend.unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(batch_size * num_heads, seq_len, p_max)
        )

        # 3. Cross-attention: Query = byte_embeddings, Key/Value = latent_patches
        decoded_bytes, _ = self.cross_attn(
            query=byte_embeddings,
            key=latent_patches,
            value=latent_patches,
            attn_mask=attn_mask_expanded,
        )  # (B, S, D)

        # 4. Project to byte class logits
        logits = self.logits_proj(decoded_bytes)  # (B, S, V)

        validate_shape(logits, (batch_size, seq_len, BYTE_VOCAB_SIZE), name="logits")
        return typing.cast(torch.Tensor, logits)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward wrapper implementing abstract BaseDecoder.forward contract.

        Args:
            x: Latent patches tensor of shape (B, P_max, D).
            **kwargs: Extra arguments including 'boundary_map' and 'raw_bytes'.

        Returns:
            Next-byte prediction logits tensor of shape (B, S, 256).
        """
        boundary_map_val = kwargs.get("boundary_map")
        raw_bytes_val = kwargs.get("raw_bytes")
        if boundary_map_val is None or not isinstance(boundary_map_val, torch.Tensor):
            raise ValueError("boundary_map must be provided as a torch.Tensor in kwargs")
        if raw_bytes_val is None or not isinstance(raw_bytes_val, torch.Tensor):
            raise ValueError("raw_bytes must be provided as a torch.Tensor in kwargs")

        boundary_map: torch.Tensor = boundary_map_val
        raw_bytes: torch.Tensor = raw_bytes_val
        return typing.cast(torch.Tensor, self.decode_with_boundaries(x, boundary_map, raw_bytes))
