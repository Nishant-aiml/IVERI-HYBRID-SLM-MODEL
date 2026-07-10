# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, shape, gradient, property, and stress tests for SparseMoERouter (Wave 1)."""

from __future__ import annotations

import typing

import pytest
import torch

from configs.base_config import get_base_config
from model.moe.router import SparseMoERouter


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_router_shapes_and_outputs(device: str) -> None:
    """Verify router forward shapes and returned dtypes."""
    config = get_base_config()
    # Explicitly set num_active_experts=2 to test K=2 routing output shape.
    # default num_active_experts may be 1 for dev iteration; this tests K=2 routing.
    config.model.num_active_experts = 2
    config.model.num_experts = 4
    router = SparseMoERouter(config).to(device=device)

    # 3D input shape (B, S, D)
    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device)
    weights, indices, aux_loss = router(x)

    assert weights.shape == (b, s, 2)
    assert indices.shape == (b, s, 2)
    assert weights.dtype == torch.float32
    assert indices.dtype == torch.int64
    assert aux_loss.shape == ()  # Scalar
    assert aux_loss.dtype == torch.float32


@pytest.mark.parametrize("device", get_test_devices())
def test_router_weights_sum_to_one(device: str) -> None:
    """Property test: Sum of routing weights per token must equal 1.0."""
    config = get_base_config()
    router = SparseMoERouter(config).to(device=device)

    # Run multiple times with random tensors
    g = torch.Generator(device=device)
    g.manual_seed(42)

    for _ in range(5):
        x = torch.randn(4, 32, 256, device=device, generator=g)
        weights, _, _ = router(x)

        # Sum of routing weights per token
        weights_sum = weights.sum(dim=-1)
        expected = torch.ones_like(weights_sum)
        torch.testing.assert_close(weights_sum, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_router_gradcheck(device: str) -> None:
    """Validate router analytical vs. numerical gradients via autograd gradcheck."""
    config = get_base_config()
    # Use smaller dims to speed up gradcheck
    config.model.hidden_dim = 16
    config.model.num_experts = 4
    config.model.num_active_experts = 2

    # Set dtype to double for numerical stability
    router = SparseMoERouter(config).to(device=device, dtype=torch.float64)

    # Double precision input with requires_grad
    x = torch.randn(2, 4, 16, device=device, dtype=torch.float64, requires_grad=True)

    # Wrap forward output to compute gradients only for weights
    def forward_weights(tensor: torch.Tensor) -> torch.Tensor:
        w, _, _ = router(tensor)
        return typing.cast(torch.Tensor, w)

    assert torch.autograd.gradcheck(forward_weights, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_router_determinism(device: str) -> None:
    """Ensure seeded router runs produce identical outputs."""
    config = get_base_config()
    torch.manual_seed(42)
    router1 = SparseMoERouter(config).to(device=device)
    router1.noise_enabled = False
    x = torch.randn(2, 8, 256, device=device)

    weights1, indices1, loss1 = router1(x)

    torch.manual_seed(42)
    router2 = SparseMoERouter(config).to(device=device)
    router2.noise_enabled = False
    weights2, indices2, loss2 = router2(x)

    torch.testing.assert_close(weights1, weights2)
    torch.testing.assert_close(indices1, indices2)
    torch.testing.assert_close(loss1, loss2)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256),  # Tiny inputs
        (8, 128, 256),  # Standard
        (16, 512, 256),  # Large sequence
    ],
)
def test_router_stress_shapes(device: str, shape: tuple[int, int, int]) -> None:
    """Stress test routing across variable scale dimensions."""
    config = get_base_config()
    # Explicitly set K=2 for shape checks below.
    config.model.num_active_experts = 2
    config.model.num_experts = 4
    router = SparseMoERouter(config).to(device=device)

    x = torch.randn(shape, device=device)
    weights, indices, loss = router(x)

    assert weights.shape == (shape[0], shape[1], 2)
    assert not torch.isnan(weights).any()
    assert not torch.isnan(loss).any()
