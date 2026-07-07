# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, shape, gradient, property, and stress tests for MoEExperts (Wave 2)."""

from __future__ import annotations

import typing

import pytest
import torch

from configs.base_config import get_base_config
from model.moe.experts import MoEExperts


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_experts_forward_shapes(device: str) -> None:
    """Verify experts forward execution shapes and dtypes."""
    config = get_base_config()
    experts = MoEExperts(config).to(device=device)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device)

    # Mock routing outputs: send all tokens to expert 0 and 1
    weights = torch.ones(b, s, 2, device=device) * 0.5
    indices = torch.zeros(b, s, 2, device=device, dtype=torch.long)
    indices[..., 1] = 1

    out, metrics = experts(x, weights, indices)

    assert out.shape == (b, s, d)
    assert out.dtype == torch.float32
    assert "capacity" in metrics
    assert "dropped_tokens" in metrics
    assert "overflow_pct" in metrics
    assert metrics["active_experts"] == 2.0


@pytest.mark.parametrize("device", get_test_devices())
def test_experts_capacity_dropping(device: str) -> None:
    """Verify that tokens exceeding capacity limits are dropped and output is zeroed."""
    config = get_base_config()
    # Force tiny capacity factor to trigger drop behavior
    config.model.num_experts = 4
    config.model.num_active_experts = 1
    experts = MoEExperts(config).to(device=device)
    experts.capacity_factor = 0.5  # Capacity = ceil(N * 1 / 4 * 0.5)

    n, d = 8, 256
    x = torch.randn(n, d, device=device)

    # Route all 8 tokens to expert 2
    weights = torch.ones(n, 1, device=device)
    indices = torch.full((n, 1), 2, device=device, dtype=torch.long)

    # Capacity = ceil((8 * 1 / 4) * 0.5) = 1
    # 7 tokens should be dropped
    out, metrics = experts(x, weights, indices)

    assert metrics["dropped_tokens"] == 7.0
    assert metrics["overflow_pct"] == 87.5  # 7 / 8

    # Verify that only the first token got computed, others are zero (bypass output)
    # The output at index 0 should be non-zero (or equal to FFN output)
    # Indices 1 to 7 should be zero
    assert torch.any(out[0] != 0.0)
    torch.testing.assert_close(out[1:], torch.zeros(7, d, device=device))


@pytest.mark.parametrize("device", get_test_devices())
def test_experts_gradcheck(device: str) -> None:
    """Validate experts analytical vs. numerical gradients via autograd gradcheck."""
    config = get_base_config()
    config.model.hidden_dim = 8
    config.model.num_experts = 2
    config.model.num_active_experts = 1
    experts = MoEExperts(config).to(device=device, dtype=torch.float64)

    # requires_grad inputs
    x = torch.randn(2, 4, 8, device=device, dtype=torch.float64, requires_grad=True)

    weights = torch.ones(2, 4, 1, device=device, dtype=torch.float64) * 0.8
    indices = torch.zeros(2, 4, 1, device=device, dtype=torch.long)

    def forward_wrapper(tensor: torch.Tensor) -> torch.Tensor:
        out, _ = experts(tensor, weights, indices)
        return typing.cast(torch.Tensor, out)

    assert torch.autograd.gradcheck(forward_wrapper, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_experts_reset_parameters(device: str) -> None:
    """Ensure reset_parameters updates expert network weights."""
    config = get_base_config()
    experts = MoEExperts(config).to(device=device)

    # Save original weights
    from model.swiglu import SwiGLUFFN

    expert0 = typing.cast(SwiGLUFFN, experts.experts[0])
    orig_weight = expert0.w_gate.weight.clone()

    # Reset
    experts.reset_parameters()

    # Weights must be different after reinitialization
    assert not torch.equal(orig_weight, expert0.w_gate.weight)
