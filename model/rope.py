# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Rotary Positional Embedding (RoPE) for the IVERI CORE architecture.

Implements the rotation-based position encoding from:

    *RoFormer: Enhanced Transformer with Rotary Position Embedding*
    Su, Lu, Pan, Murtadha, Wen, & Liu (2021).  arXiv:2104.09864

Mathematical Definition
-----------------------

Pre-compute frequency bands (inverse frequencies):

    θ_i = base^(−2(i−1) / d),    for  i = 1, 2, …, d/2

where ``base`` defaults to 10 000.

For position *m* and feature pair (x_{2i}, x_{2i+1}) the rotation is:

    ┌ x'_{2i}   ┐   ┌ cos(m·θ_i)  −sin(m·θ_i) ┐ ┌ x_{2i}   ┐
    │            │ = │                           │ │           │
    └ x'_{2i+1} ┘   └ sin(m·θ_i)   cos(m·θ_i)  ┘ └ x_{2i+1} ┘

Efficient implementation via the **rotate-half** trick avoids the
explicit 2×2 matrix multiply:

    RoPE(x) = x ⊙ cos(m·Θ)  +  rotate_half(x) ⊙ sin(m·Θ)

where ``rotate_half(x) = [−x[d/2:], x[:d/2]]`` (negate-and-swap).

Reference
---------
HuggingFace ``LlamaRotaryEmbedding`` serves as the canonical reference
implementation for correctness validation.
"""

from __future__ import annotations

import torch
from torch import nn

from core.registry import register

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting in half and negating.

    Given ``x`` of shape ``(..., D)``, returns a tensor where the first
    ``D/2`` elements are the negated second half, and the last ``D/2``
    elements are the original first half:

        rotate_half(x) = cat(−x[..., D/2:], x[..., :D/2])

    Parameters
    ----------
    x:
        Input tensor whose last dimension ``D`` must be even.

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same shape as *x*.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor *x*.

    Uses the rotate-half trick for efficiency:

        RoPE(x) = x ⊙ cos  +  rotate_half(x) ⊙ sin

    The ``cos`` and ``sin`` tensors are expected to be broadcastable to
    the shape of *x*.  Typical input shapes:

    - ``x``:   ``(B, H, S, D_head)``  or  ``(B, S, H, D_head)``
    - ``cos``: ``(S, D_head)``  or  ``(1, 1, S, D_head)``
    - ``sin``: same as ``cos``

    Parameters
    ----------
    x:
        Query or key tensor to rotate.
    cos:
        Pre-computed cosine cache, sliced to the required sequence length.
    sin:
        Pre-computed sine cache, sliced to the required sequence length.

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same shape and dtype as *x*.
    """
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# RoPE Module
# ---------------------------------------------------------------------------


@register("rope")
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding with pre-computed cos/sin caches.

    This module is a **buffer-only utility** — it contains no learnable
    parameters and therefore inherits from :class:`nn.Module` directly
    rather than from :class:`~core.interfaces.BaseModule`.

    Buffers
    -------
    inv_freq : ``(dim // 2,)``
        Inverse frequency bands  θ_i = base^(−2i / dim).
    cos_cached : ``(max_seq_len, dim)``
        Pre-computed ``cos(m · Θ)`` for positions ``m ∈ [0, max_seq_len)``.
    sin_cached : ``(max_seq_len, dim)``
        Pre-computed ``sin(m · Θ)`` for the same range.

    Parameters
    ----------
    dim:
        Rotary embedding dimension (must be even; typically ``head_dim``).
    max_seq_len:
        Maximum sequence length for the initial cache.  The cache is
        automatically extended on demand if a longer sequence is encountered.
    base:
        Base value for computing inverse frequencies (default: 10 000).

    Example
    -------
    >>> rope = RotaryEmbedding(dim=64, max_seq_len=2048)
    >>> cos, sin = rope(x, seq_len=128)          # sliced to 128
    >>> q_rot = apply_rotary_emb(q, cos, sin)    # apply to queries
    >>> k_rot = apply_rotary_emb(k, cos, sin)    # apply to keys
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # θ_i = base^(−2i / dim)  for i = 0, 1, …, dim/2 − 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build the initial cos/sin cache
        self._build_cache(max_seq_len, inv_freq.device)

    # -- cache management ---------------------------------------------------

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """(Re-)build the cos/sin cache for positions ``[0, seq_len)``.

        This method is called automatically during ``__init__`` and
        whenever :meth:`forward` is invoked with a ``seq_len`` that
        exceeds the current cache size.

        Parameters
        ----------
        seq_len:
            Number of positions to pre-compute.
        device:
            Target device for the cached tensors.
        """
        self.max_seq_len = seq_len

        # Position indices: [0, 1, …, seq_len − 1]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        from typing import cast

        inv_freq = cast(torch.Tensor, self.inv_freq)
        freqs = torch.outer(t, inv_freq)

        # Duplicate to full dim → (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Register cos/sin as non-persistent buffers
        cos_cached = emb.cos()
        sin_cached = emb.sin()

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tensors sliced to the requested length.

        If *seq_len* exceeds the current cache, the cache is transparently
        extended to accommodate the new length.

        Parameters
        ----------
        x:
            Input tensor — used only to infer device and dtype.
        seq_len:
            Number of positions to return.  Defaults to ``x.shape[-2]``
            (the penultimate dimension, typically the sequence axis).

        Returns
        -------
        cos : torch.Tensor
            Cosine cache of shape ``(seq_len, dim)``.
        sin : torch.Tensor
            Sine cache of shape ``(seq_len, dim)``.
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Extend cache if necessary
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, x.device)

        from typing import cast

        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        cos = cos_cached[:seq_len].to(dtype=x.dtype)
        sin = sin_cached[:seq_len].to(dtype=x.dtype)
        return cos, sin

    # -- repr ---------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, " f"base={self.base}"
