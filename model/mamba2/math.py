# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SSD (Structured State Space Duality) Mathematical Primitives (Wave 1).

Implements parameter discretization, state parameter formulations, and
helper matrix multiplication utilities for structured state space models.
"""

from __future__ import annotations

import torch


def discretize_parameters(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
    method: str = "zoh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize continuous-time state space parameters to discrete time.

    Args:
        A: Continuous state transition tensor of shape (..., D_state). Must be negative for stability.
        B: Continuous input-to-state projection tensor of shape (..., D_inner, D_state).
        delta: Discretization step size tensor of shape (..., D_inner) or (..., D_state).
        method: Discretization method, either 'euler' or 'zoh' (default).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Discretized state transition (A_bar) and input-to-state (B_bar).
            A_bar has shape matching A.
            B_bar has shape matching B.

    Raises:
        ValueError: If discretization method is unsupported.
    """
    delta_expanded = delta.unsqueeze(-1) if delta.dim() < B.dim() else delta
    if A.dim() == 2 and delta.dim() == 4:
        A_unsqueezed = A.unsqueeze(0).unsqueeze(2)
    else:
        A_unsqueezed = A.unsqueeze(0) if A.dim() < B.dim() else A

    if method == "euler":
        # Euler discretization: A_bar = 1 + delta * A, B_bar = delta * B
        A_bar = 1.0 + delta_expanded * A_unsqueezed
        B_bar = delta_expanded * B
        return A_bar, B_bar

    elif method == "zoh":
        # Zero-Order Hold (ZOH) discretization
        # A_bar = exp(delta * A)
        # B_bar = (exp(delta * A) - I) * A^-1 * B
        delta_a = delta_expanded * A_unsqueezed
        A_bar = torch.exp(delta_a)

        # Stable calculation of (exp(x) - 1) / x to avoid division by zero
        # When x is small, we use Taylor expansion: (exp(x) - 1)/x approx 1 + x/2 + x^2/6
        x = delta_a
        mask = torch.abs(x) < 1e-5

        coef = torch.empty_like(x)
        coef[mask] = 1.0 + 0.5 * x[mask] + (1.0 / 6.0) * x[mask].pow(2)
        coef[~mask] = (torch.exp(x[~mask]) - 1.0) / x[~mask]

        # B_bar = coef * delta * B
        B_bar = coef * delta_expanded * B
        return A_bar, B_bar
    else:
        raise ValueError(f"Unsupported discretization method: {method}")


def compute_ssd_matrix(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Compute the 1D semi-separable matrix kernel representing recurrent operations.

    For sequence length S, computes M of shape (B, H, S, S) such that:
        M[b, h, i, j] = C[b, h, i] * (prod_{k=j+1}^i A_bar[b, h, k]) * B_bar[b, h, j]   for i > j
        M[b, h, i, j] = C[b, h, i] * B_bar[b, h, j]                                   for i == j
        M[b, h, i, j] = 0                                                           for i < j

    This is the dense equivalent kernel representing Mamba2's SSD formulation.

    Args:
        A_bar: Discretized transition parameter of shape (B, H, S, D_state) or (B, H, D_state).
               If shape is (B, H, D_state), transition is time-invariant.
        B_bar: Discretized input projection of shape (B, H, S, D_state).
        C: Output state projection of shape (B, H, S, D_state).

    Returns:
        torch.Tensor: The kernel matrix of shape (B, H, S, S).
    """
    b, h, s, d = B_bar.shape
    device = B_bar.device
    dtype = B_bar.dtype

    # Expand A_bar to be time-varying (B, H, S, D_state) if it is time-invariant
    A_bar_seq = A_bar.unsqueeze(2).expand(b, h, s, d) if A_bar.dim() == 3 else A_bar

    # Compute sequence cumulative products for efficient semi-separable scaling
    # log_cumprod_A[t] = sum_{k=0}^t log(A_bar_seq[k])
    # Then prod_{k=j+1}^i A_bar_seq[k] = exp(log_cumprod_A[i] - log_cumprod_A[j])
    # Avoid numerical instability with small negative or zero values by clipping
    A_bar_seq_clamped = torch.clamp(A_bar_seq, min=1e-12)
    log_cumprod_A = torch.cumsum(torch.log(A_bar_seq_clamped), dim=2)  # (B, H, S, D_state)

    # Compute M[i, j] using tensor contractions
    # C[i] has shape (B, H, S, D_state)
    # B_bar[j] has shape (B, H, S, D_state)
    # We want to contract over D_state.
    # M[i, j] = sum_d C[i, d] * exp(log_cumprod_A[i, d] - log_cumprod_A[j, d]) * B_bar[j, d]

    # Expand dims to compute all pairs (i, j)
    log_cumprod_A_i = log_cumprod_A.unsqueeze(3)  # (B, H, S, 1, D_state)
    log_cumprod_A_j = log_cumprod_A.unsqueeze(2)  # (B, H, 1, S, D_state)

    # Cumulative transition factor: exp(log_cumprod_A_i - log_cumprod_A_j)
    # Shape is (B, H, S, S, D_state)
    transition_factor = torch.exp(log_cumprod_A_i - log_cumprod_A_j)

    # C_i shape: (B, H, S, 1, D_state)
    # B_j shape: (B, H, 1, S, D_state)
    C_i = C.unsqueeze(3)
    B_j = B_bar.unsqueeze(2)

    # Elements calculation: (B, H, S, S, D_state)
    M = C_i * transition_factor * B_j

    # Sum over state dimension D_state -> (B, H, S, S)
    M_sum = M.sum(dim=-1)

    # Mask out the upper triangular elements (where i < j) to enforce causality
    causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=dtype))
    return M_sum * causal_mask
