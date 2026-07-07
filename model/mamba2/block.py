# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 Block layer implementation for IVERI CORE (Wave 3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from model.mamba2.scan import selective_ssd_scan


@register("mamba2")
class Mamba2Block(BaseModule):
    """Mamba2 (Structured State Space Duality) layer block.

    Assembles input projections, causal 1D convolutions, selective SSD scan,
    multiplicative gating, and output projection into a unified linear-time block.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize Mamba2Block.

        Args:
            config: Full project configuration dataclass.
        """
        super().__init__()
        self.config = config

        self.d_model = config.model.hidden_dim
        self.num_heads = config.model.num_heads

        # Mamba expansion factor defaults to 2
        self.expand = 2
        self.d_inner = self.d_model * self.expand

        if self.d_inner % self.num_heads != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by num_heads ({self.num_heads})"
            )
        self.d_head = self.d_inner // self.num_heads

        # State space dimension (default 16)
        self.d_state = 16

        # Projection size: x (d_inner), gate (d_inner), delta (d_inner), B (d_state), C (d_state)
        # Total: 3 * d_inner + 2 * d_state
        self.proj_dim = 3 * self.d_inner + 2 * self.d_state

        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.proj_dim, bias=False)

        # Causal 1D Convolution mapping for [x, delta, B, C] paths
        self.conv_channels = 2 * self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=4,
            groups=self.conv_channels,
            bias=True,
            padding=0,  # Manual causal padding handled in forward
        )

        # Transition matrix parameter A
        # Parameterized as log(-A) to ensure stability (always negative values)
        # Shape: (num_heads, d_head)
        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_head))

        # Discretization bias dt_bias
        self.dt_bias = nn.Parameter(torch.empty(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize block parameters."""
        # A matrix: initialize log(-A) uniformly in [1, 5]
        # This keeps A in [-e^5, -e^1] = [-148.4, -2.7] (stable decay)
        nn.init.uniform_(self.A_log, 1.0, 5.0)

        # dt_bias: log scale initialization
        nn.init.uniform_(self.dt_bias, -3.0, -1.0)

        # Projections initialization using standard Kaiming
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

        # Conv1d initialization
        if self.conv1d.bias is not None:
            nn.init.constant_(self.conv1d.bias, 0.0)
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))

    @property
    def A(self) -> torch.Tensor:
        """Continuous state transition parameter A (negative values)."""
        return -torch.exp(self.A_log)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward execution pass of the Mamba2 block.

        Args:
            x: Input tensor of shape (B, S, D)

        Returns:
            torch.Tensor: Projected outputs of shape (B, S, D)
        """
        # Input projection
        # Projected tensor shape: (B, S, proj_dim)
        projected = self.in_proj(x)

        # Split into paths
        # x_path: (B, S, d_inner)
        # gate: (B, S, d_inner)
        # delta: (B, S, d_inner)
        # B: (B, S, d_state)
        # C: (B, S, d_state)
        x_path, gate, delta, B, C = torch.split(
            projected,
            [self.d_inner, self.d_inner, self.d_inner, self.d_state, self.d_state],
            dim=-1,
        )

        # Causal Conv1D path: Concatenate x, delta, B, and C along channel dimension
        conv_in = torch.cat([x_path, delta, B, C], dim=-1)  # (B, S, conv_channels)
        conv_in = conv_in.transpose(1, 2)  # (B, conv_channels, S)

        # Causal padding of size (kernel_size - 1) on the left sequence dimension
        # Left pad by 3 for kernel_size=4
        padded = F.pad(conv_in, (3, 0))
        conv_out = self.conv1d(padded)
        conv_out = conv_out.transpose(1, 2)  # (B, S, conv_channels)

        # Split back after convolution
        x_conv, delta_conv, B_conv, C_conv = torch.split(
            conv_out, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=-1
        )

        # Discretization bias calculation
        # delta_param = softplus(delta_conv + dt_bias)
        delta_param = F.softplus(delta_conv + self.dt_bias)

        # Reshape / Transpose to Head Layout
        # Inputs shape: (B, S, D_inner) -> (B, S, H, D_head) -> (B, H, S, D_head)
        x_heads = x_conv.view(-1, x_conv.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        delta_heads = delta_param.view(
            -1, delta_param.shape[1], self.num_heads, self.d_head
        ).transpose(1, 2)

        # Expand B and C projections to be shared across all Heads
        # Shape: (B, H, S, D_state)
        B_heads = B_conv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        C_heads = C_conv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Run selective SSD scan
        y_scan, _ = selective_ssd_scan(x_heads, delta_heads, self.A, B_heads, C_heads)

        # Transpose back to sequence layout
        # (B, H, S, D_head) -> (B, S, H, D_head) -> (B, S, D_inner)
        y_out = y_scan.transpose(1, 2).contiguous().view(-1, y_scan.shape[2], self.d_inner)

        # Multiplicative scaling with gate path (SiLU activated)
        y_gated = y_out * F.silu(gate)

        # Output projection
        import typing

        out = self.out_proj(y_gated)
        return typing.cast(torch.Tensor, out)

    def extra_repr(self) -> str:
        """Extra representation helper."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"expand={self.expand}, d_state={self.d_state}"
        )
