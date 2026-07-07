# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU activation and feed-forward network.

Implements the SwiGLU gated activation from:

    Noam Shazeer, "GLU Variants Improve Transformer", 2020.
    https://arxiv.org/abs/2002.05202

**SwiGLU activation**::

    SwiGLU(gate, value) = Swish(gate) ⊙ value

where ``Swish(x) = x · σ(x)`` (also known as SiLU).

**SwiGLU FFN block** (as used in LLaMA)::

    FFN(x) = W_o · (Swish(x · W_g) ⊙ (x · W_v))

The intermediate dimension ``d_ff`` defaults to the LLaMA convention:
``round(2/3 · 4d)`` rounded up to the nearest multiple of 256 for
hardware-friendly memory alignment.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from core.interfaces import BaseModule
from core.registry import register

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_ff_dim(dim: int) -> int:
    """Compute LLaMA-style intermediate FFN dimension.

    Parameters
    ----------
    dim:
        Model hidden dimension.

    Returns
    -------
    int
        ``ceil(2/3 · 4 · dim)`` rounded up to the nearest multiple of 256.
    """
    ff = int(2 * dim * 4 / 3)
    # Round up to nearest multiple of 256 for hardware alignment
    return ((ff + 255) // 256) * 256


# ---------------------------------------------------------------------------
# SwiGLU activation (pure, parameter-free)
# ---------------------------------------------------------------------------


@register("swiglu")
class SwiGLU(BaseModule):
    """Parameter-free SwiGLU gated activation.

    Computes ``Swish(gate) ⊙ value`` element-wise, where
    ``Swish(x) = x · σ(x)`` (``F.silu``).

    This class owns no learnable parameters; it is a pure activation
    function intended to be composed inside larger modules.
    """

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply the SwiGLU activation.

        Parameters
        ----------
        x:
            Gate tensor, or concatenated (gate, value) tensor if *value* is not provided in kwargs.
        **kwargs:
            Optional keyword argument ``value`` of type :class:`torch.Tensor`.
            If not provided, *x* is split in half along the last dimension.

        Returns
        -------
        torch.Tensor
            SwiGLU activation output.
        """
        value_opt = kwargs.get("value")
        if value_opt is None:
            gate, value = x.chunk(2, dim=-1)
        else:
            if not isinstance(value_opt, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for value, got {type(value_opt)}")
            gate = x
            value = value_opt
        return F.silu(gate) * value


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


@register("swiglu_ffn")
class SwiGLUFFN(BaseModule):
    """Complete SwiGLU feed-forward block (LLaMA-style MLP).

    Three linear projections — *gate*, *value*, and *output* — combined
    with the SwiGLU activation::

        FFN(x) = W_o · (Swish(x · W_g) ⊙ (x · W_v))

    Parameters
    ----------
    dim:
        Input / output hidden dimension.
    hidden_dim:
        Intermediate (expanded) dimension.  When *None*, automatically
        computed via :func:`_compute_ff_dim` following LLaMA conventions.
    bias:
        Whether the linear layers include a bias term.  Defaults to
        ``False`` (modern practice, LLaMA).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else _compute_ff_dim(dim)
        self.bias = bias

        # Gate projection  (x → gate activations)
        self.w_gate = nn.Linear(self.dim, self.hidden_dim, bias=self.bias)
        # Value projection (x → value activations)
        self.w_value = nn.Linear(self.dim, self.hidden_dim, bias=self.bias)
        # Output projection (hidden → output)
        self.w_out = nn.Linear(self.hidden_dim, self.dim, bias=self.bias)

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Compute the SwiGLU FFN.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., dim)``.
        **kwargs:
            Ignored; accepted for interface compatibility.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., dim)``.
        """
        from typing import cast

        return cast(torch.Tensor, self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))

    # -- initialisation ----------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise all linear layers (PyTorch default: Kaiming uniform)."""
        for layer in (self.w_gate, self.w_value, self.w_out):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                # Match nn.Linear default: uniform fan-in bound
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(layer.bias, -bound, bound)

    # -- repr --------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, bias={self.bias}"
