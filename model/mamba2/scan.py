# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Selective Scan recurrence algorithms for Mamba2 Structured State Space Duality (Wave 2)."""

from __future__ import annotations

import torch
import torch.utils.checkpoint


def selective_ssd_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform sequential selective Structured State Space Duality (SSD) scan recurrence.

    Mathematical recurrence:
        h_t = A_bar_t * h_{t-1} + (x_t * delta_t) otimes B_t
        y_t = h_t . C_t

    Args:
        x: Input tensor of shape (B, H, S, D_head)
        delta: Step size tensor of shape (B, H, S, D_head)
        A: Transition parameters of shape (H, D_head) or (H,)
        B: Input projection tensor of shape (B, H, S, D_state)
        C: State projection tensor of shape (B, H, S, D_state)
        initial_state: Optional initial state of shape (B, H, D_head, D_state)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            y: Output tensor of shape (B, H, S, D_head)
            final_state: Final state tensor of shape (B, H, D_head, D_state)
    """
    b, h, s, d_head = x.shape
    d_state = B.shape[-1]
    device = x.device
    dtype = x.dtype

    # Expand A to shape (H, D_head) if it is 1D (H,)
    A_exp = A.unsqueeze(-1) if A.dim() == 1 else A

    # 1. Vectorized precomputation of discretization factors (A_bar)
    # delta shape: (B, H, S, D_head)
    # A_exp shape: (H, D_head) -> unsqueeze to (1, H, 1, D_head)
    A_bar = torch.exp(delta * A_exp.unsqueeze(0).unsqueeze(2))  # (B, H, S, D_head)
    
    # Precompute elementwise x_delta to avoid doing it inside the loop
    x_delta = x * delta  # (B, H, S, D_head)

    # Initialize state
    if initial_state is not None:
        state = initial_state.clone()
    else:
        state = torch.zeros(b, h, d_head, d_state, device=device, dtype=dtype)

    # Pre-unsqueeze static shapes to avoid repeated unsqueeze calls in the sequential loop
    # Note: B.unsqueeze(-2) and C.unsqueeze(-2) are extremely lightweight (dimension size 1)
    A_bar_unsqueezed = A_bar.unsqueeze(-1)  # (B, H, S, D_head, 1)
    B_unsqueezed = B.unsqueeze(-2)  # (B, H, S, 1, D_state)
    C_unsqueezed = C.unsqueeze(-2)  # (B, H, S, 1, D_state)

    # 2. Sequential recurrence loop (optimized out-of-place to preserve autograd efficiency)
    outputs = []
    for t in range(s):
        # Compute input outer product dynamically to avoid large (B, H, S, D_head, D_state) allocations
        input_term = x_delta[:, :, t].unsqueeze(-1) * B_unsqueezed[:, :, t]
        state = A_bar_unsqueezed[:, :, t] * state + input_term
        y_t = (state * C_unsqueezed[:, :, t]).sum(dim=-1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=2)  # (B, H, S, D_head)
    return y, state
