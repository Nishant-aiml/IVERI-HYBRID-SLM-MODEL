# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, shape, gradient correctness, property, and stress tests for SSD Math (Wave 1)."""

from __future__ import annotations

import pytest
import torch

from model.mamba2.math import compute_ssd_matrix, discretize_parameters


def get_test_devices() -> list[str]:
    """Get available devices for testing."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("method", ["euler", "zoh"])
def test_discretize_parameters_shapes(device: str, method: str) -> None:
    """Verify shape retention for discretization."""
    b, inner, d_state = 2, 8, 4

    # Negative parameter for transition matrix to guarantee ZOH stability
    A = -torch.rand(inner, d_state, device=device)
    B = torch.randn(b, inner, d_state, device=device)
    delta = torch.rand(b, inner, device=device)

    A_bar, B_bar = discretize_parameters(A, B, delta, method=method)

    assert A_bar.shape == (b, inner, d_state)
    assert B_bar.shape == (b, inner, d_state)
    assert A_bar.dtype == torch.float32
    assert B_bar.dtype == torch.float32


@pytest.mark.parametrize("device", get_test_devices())
def test_discretize_zoh_stability_near_zero(device: str) -> None:
    """Ensure ZOH discretization remains stable and NaN-free near zero."""
    b, inner, d_state = 1, 4, 2

    # A values set extremely close to zero to trigger Taylor expansion mask
    A = torch.zeros(inner, d_state, device=device) + 1e-8
    B = torch.ones(b, inner, d_state, device=device)
    delta = torch.ones(b, inner, device=device)

    A_bar, B_bar = discretize_parameters(A, B, delta, method="zoh")

    assert not torch.isnan(A_bar).any()
    assert not torch.isnan(B_bar).any()
    # For A approx 0, delta=1, A_bar = exp(delta*A) approx 1.0
    assert torch.allclose(A_bar, torch.ones_like(A_bar), atol=1e-4)
    # B_bar = (exp(x)-1)/x * B approx 1.0 * B
    assert torch.allclose(B_bar, B, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_discretize_gradcheck(device: str) -> None:
    """Validate discretization analytical vs. numerical gradients via gradcheck."""
    A = -torch.rand(4, 2, device=device, dtype=torch.float64, requires_grad=True)
    B = torch.randn(2, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
    delta = torch.rand(2, 4, device=device, dtype=torch.float64, requires_grad=True)

    def wrapper(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        a_bar, b_bar = discretize_parameters(a, b, d, method="zoh")
        # Return sum of outputs to check joint gradients
        return a_bar + b_bar

    assert torch.autograd.gradcheck(wrapper, (A, B, delta), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_ssd_matrix_computation(device: str) -> None:
    """Verify shape, causal structures and mathematical values of computed SSD matrix."""
    b, h, s, d_state = 2, 3, 4, 2
    A_bar = torch.ones(b, h, s, d_state, device=device) * 0.9
    B_bar = torch.ones(b, h, s, d_state, device=device) * 0.5
    C = torch.ones(b, h, s, d_state, device=device) * 0.8

    M = compute_ssd_matrix(A_bar, B_bar, C)

    # Assert shape is (B, H, S, S)
    assert M.shape == (b, h, s, s)

    # Verify causality (strictly causal upper triangular is 0)
    for i in range(s):
        for j in range(i + 1, s):
            assert torch.all(M[:, :, i, j] == 0.0)

    # Check diagonal elements: M[i, i] = sum_d C[i, d] * B_bar[i, d] = 2 * 0.8 * 0.5 = 0.8
    diagonal = torch.diagonal(M, dim1=-2, dim2=-1)
    assert torch.allclose(diagonal, torch.ones_like(diagonal) * 0.8)


@pytest.mark.parametrize("device", get_test_devices())
def test_ssd_matrix_gradcheck(device: str) -> None:
    """Validate SSD matrix computation gradients via gradcheck."""
    b, h, s, d_state = 1, 2, 3, 2
    A_bar = torch.rand(b, h, s, d_state, device=device, dtype=torch.float64, requires_grad=True)
    B_bar = torch.rand(b, h, s, d_state, device=device, dtype=torch.float64, requires_grad=True)
    C = torch.rand(b, h, s, d_state, device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(compute_ssd_matrix, (A_bar, B_bar, C), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_math_property_and_stress(device: str) -> None:
    """Stress test mathematical functions with boundary values and properties."""
    # Property: A=0, delta=1 yields A_bar = 1
    A = torch.zeros(2, 2, device=device)
    B = torch.ones(1, 2, 2, device=device)
    delta = torch.ones(1, 2, device=device)
    A_bar, B_bar = discretize_parameters(A, B, delta, method="zoh")
    assert torch.allclose(A_bar, torch.ones_like(A_bar))

    # Stress: extremely large scale delta
    delta_large = torch.ones(1, 2, device=device) * 1e5
    A_neg = -torch.ones(2, 2, device=device)
    A_bar_large, _ = discretize_parameters(A_neg, B, delta_large, method="zoh")
    # exp(-1e5) must be extremely close to 0
    assert torch.allclose(A_bar_large, torch.zeros_like(A_bar_large), atol=1e-8)
