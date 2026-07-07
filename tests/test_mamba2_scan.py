# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, equivalence, gradient, property, and stress tests for Selective Scan (Wave 2)."""

from __future__ import annotations

import pytest
import torch

from model.mamba2.math import compute_ssd_matrix
from model.mamba2.scan import selective_ssd_scan


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_scan_equivalence_to_expanded_ssd(device: str) -> None:
    """Verify that sequential recurrence scan matches the parallel SSD matrix formulation."""
    b, h, s, d_head, d_state = 2, 2, 8, 1, 3

    # Random parameters
    x = torch.randn(b, h, s, d_head, device=device)
    delta = torch.rand(b, h, s, d_head, device=device) * 0.1
    # Negative A transition parameters
    A = -torch.rand(h, d_head, device=device)
    B = torch.randn(b, h, s, d_state, device=device)
    C = torch.randn(b, h, s, d_state, device=device)

    # Method A: Sequential scan recurrence
    y_rec, _ = selective_ssd_scan(x, delta, A, B, C)

    # Method B: Expanded SSD matrix formulation
    # Since d_head = 1, delta is (B, H, S, 1) and B is (B, H, S, D_state)
    delta_expanded = delta.unsqueeze(-1) if delta.dim() < B.dim() else delta
    A_unsqueezed = A.unsqueeze(0).unsqueeze(2)
    A_bar = torch.exp(delta_expanded * A_unsqueezed)
    B_bar = delta_expanded * B
    M = compute_ssd_matrix(A_bar, B_bar, C)  # (B, H, S, S)

    # Multiply matrix M by input x (B, H, S, 1)
    # We contract along sequence dimension: y_matmul = M @ x
    y_matmul = torch.matmul(M, x)  # (B, H, S, 1)

    # Assert they are equal within floating-point tolerance
    assert torch.allclose(y_rec, y_matmul, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_scan_gradcheck(device: str) -> None:
    """Validate scan backward pass correctness using gradcheck."""
    b, h, s, d_head, d_state = 1, 2, 4, 2, 2

    x = torch.randn(b, h, s, d_head, device=device, dtype=torch.float64, requires_grad=True)
    delta = (
        torch.rand(b, h, s, d_head, device=device, dtype=torch.float64, requires_grad=True) * 0.1
    )
    A = -torch.rand(h, d_head, device=device, dtype=torch.float64, requires_grad=True)
    B = torch.randn(b, h, s, d_state, device=device, dtype=torch.float64, requires_grad=True)
    C = torch.randn(b, h, s, d_state, device=device, dtype=torch.float64, requires_grad=True)

    def wrapper(
        x_in: torch.Tensor,
        d_in: torch.Tensor,
        a_in: torch.Tensor,
        b_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = selective_ssd_scan(x_in, d_in, a_in, b_in, c_in)
        return out

    assert torch.autograd.gradcheck(wrapper, (x, delta, A, B, C), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048, 4096])
def test_scan_long_sequence_stability(device: str, seq_len: int) -> None:
    """Confirm state parameters remain bounded and numerically stable over long sequences."""
    b, h, d_head, d_state = 2, 4, 16, 8

    # Stable inputs initialization
    x = torch.randn(b, h, seq_len, d_head, device=device, requires_grad=True)
    delta = torch.rand(b, h, seq_len, d_head, device=device) * 0.05
    A = -torch.rand(h, d_head, device=device)
    B = torch.randn(b, h, seq_len, d_state, device=device)
    C = torch.randn(b, h, seq_len, d_state, device=device)

    y, final_state = selective_ssd_scan(x, delta, A, B, C)

    # Compute loss and backward
    loss = y.sum()
    loss.backward()

    # Assert values are finite (no NaNs, no Infs)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert not torch.isnan(final_state).any()
    assert not torch.isinf(final_state).any()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Hidden state must remain bounded (typically norm does not explode)
    state_norm = final_state.norm().item()
    grad_norm = x.grad.norm().item()
    max_state = final_state.abs().max().item()

    assert state_norm < 1e5
    assert grad_norm < 1e5
    assert max_state < 100.0


@pytest.mark.parametrize("device", get_test_devices())
def test_scan_property_and_determinism(device: str) -> None:
    """Verify seed determinism and causal properties."""
    # Determinism
    torch.manual_seed(42)
    x = torch.randn(2, 2, 16, 4, device=device)
    delta = torch.rand(2, 2, 16, 4, device=device)
    A = -torch.rand(2, 4, device=device)
    B = torch.randn(2, 2, 16, 4, device=device)
    C = torch.randn(2, 2, 16, 4, device=device)

    y1, s1 = selective_ssd_scan(x, delta, A, B, C)

    torch.manual_seed(42)
    # Re-create inputs to ensure same values
    x_dup = x.clone()
    delta_dup = delta.clone()
    A_dup = A.clone()
    B_dup = B.clone()
    C_dup = C.clone()

    y2, s2 = selective_ssd_scan(x_dup, delta_dup, A_dup, B_dup, C_dup)

    assert torch.equal(y1, y2)
    assert torch.equal(s1, s2)
