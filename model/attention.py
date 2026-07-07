# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention Wrapper implementation for IVERI CORE (Phase 1.4)."""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register

# Check for flash_attn library availability
_HAS_FLASH_ATTN = False
try:
    import flash_attn

    _HAS_FLASH_ATTN = True
except ImportError:
    pass


@register("attention")
class FlashAttentionWrapper(BaseModule):
    """Unified Attention Wrapper.

    Dynamically dispatches attention computation between PyTorch Scaled Dot-Product Attention (SDPA)
    and FlashAttention-2 (if CUDA is active and library is installed), ensuring backend independence.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the attention wrapper.

        Args:
            config: General architecture configurations.
        """
        super().__init__()
        self.config = config
        self.d_model = config.model.hidden_dim
        self.num_heads = config.model.num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        self.d_head = self.d_model // self.num_heads
        self.dropout_p = config.model.dropout

        # QKV Projection mapping: hidden state (D) -> (3 * D)
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)

        # Output projection mapping: (D) -> (D)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters using Kaiming Uniform initialization."""
        nn.init.kaiming_uniform_(self.qkv_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

    def _dispatch_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Execute attention via PyTorch Scaled Dot-Product Attention (SDPA).

        Args:
            q: Queries of shape (B, H, S_q, d_head)
            k: Keys of shape (B, H, S_k, d_head)
            v: Values of shape (B, H, S_k, d_head)
            is_causal: Apply causal attention masking.

        Returns:
            torch.Tensor: Computed attention tensor of shape (B, H, S_q, d_head).
        """
        # When generating incrementally, the sequence lengths of Q and K differ.
        # We must disable PyTorch's default is_causal=True for different length tensors to avoid exceptions.
        # Prefill/Full sequence: S_q == S_k -> is_causal acts causally.
        # Incremental decode: S_q == 1, S_k > 1 -> past tokens are already fully causal, no mask needed.
        actual_causal = is_causal and (q.shape[2] == k.shape[2])

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=actual_causal,
        )
        return out

    def _dispatch_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Execute attention via FlashAttention-2.

        Args:
            q: Queries of shape (B, H, S_q, d_head)
            k: Keys of shape (B, H, S_k, d_head)
            v: Values of shape (B, H, S_k, d_head)
            is_causal: Apply causal masking.

        Returns:
            torch.Tensor: Computed attention tensor of shape (B, H, S_q, d_head).
        """
        # Flash Attention expects: (B, S, H, D)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Call official flash_attn function
        out_t = flash_attn.flash_attn_func(
            q_t,
            k_t,
            v_t,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=is_causal,
        )

        # Transpose back to (B, H, S, D)
        return typing.cast(torch.Tensor, out_t.transpose(1, 2))

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass of the FlashAttentionWrapper.

        Args:
            x: Input sequence tensor of shape (B, S, D)
            **kwargs: Configuration flags:
                kv_cache (dict[str, torch.Tensor] | None): Optional KV cache dictionary.
                is_causal (bool): Apply causal attention masking. Defaults to True.

        Returns:
            torch.Tensor: Output attention projection of shape (B, S, D).
        """
        b, s, d = x.shape

        # Retrieve kwargs flags
        kv_cache = kwargs.get("kv_cache")
        is_causal = bool(kwargs.get("is_causal", True))

        # Linear projection to QKV: (B, S, 3 * D)
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to multi-head layout: (B, H, S, d_head)
        q = q.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.d_head).transpose(1, 2)

        # Update Key-Value caching
        if isinstance(kv_cache, dict):
            if "key" in kv_cache and "value" in kv_cache:
                k = torch.cat([kv_cache["key"], k], dim=2)
                v = torch.cat([kv_cache["value"], v], dim=2)
            kv_cache["key"] = k
            kv_cache["value"] = v

        # Backend Dispatching selection
        if _HAS_FLASH_ATTN and q.is_cuda:
            # Dispatch to native CUDA FlashAttention-2
            out = self._dispatch_flash_attn(q, k, v, is_causal=is_causal)
        else:
            # Dispatch to optimized PyTorch SDPA (CPU / default CUDA)
            out = self._dispatch_sdpa(q, k, v, is_causal=is_causal)

        # Transpose and project output back to hidden dimension D
        out_t = out.transpose(1, 2).contiguous().view(b, s, d)
        out_proj = self.out_proj(out_t)

        return typing.cast(torch.Tensor, out_proj)

    def extra_repr(self) -> str:
        """Represent extra configuration features."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, " f"dropout_p={self.dropout_p}"
