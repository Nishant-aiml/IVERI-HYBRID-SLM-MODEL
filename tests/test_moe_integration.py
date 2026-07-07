# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests combining SparseMoERouter and MoEExperts (Wave 3)."""

from __future__ import annotations

import typing

import pytest
import torch

from configs.base_config import get_base_config
from model.moe import MoEExperts, SparseMoERouter


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_moe_end_to_end_pipeline(device: str) -> None:
    """Verify combined gating router and experts container forward execution."""
    config = get_base_config()
    config.model.moe_capacity_factor = 10.0  # Prevent probabilistic token drop failures
    router = SparseMoERouter(config).to(device=device)
    experts = MoEExperts(config).to(device=device)

    # 10M Nano configuration dimensions
    b, s, d = 2, 32, 256
    x = torch.randn(b, s, d, device=device, requires_grad=True)

    # Forward routing
    weights, indices, aux_loss = router(x)

    # Forward experts execution
    out, metrics = experts(x, weights, indices)

    assert out.shape == (b, s, d)
    assert out.dtype == torch.float32
    assert metrics["active_experts"] <= 4.0
    assert metrics["dropped_tokens"] == 0.0

    # Backward pass verification
    loss = out.sum() + aux_loss
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize("device", get_test_devices())
def test_moe_end_to_end_gradcheck(device: str) -> None:
    """Validate autograd gradient correctness end-to-end using gradcheck."""
    config = get_base_config()
    config.model.hidden_dim = 8
    config.model.num_experts = 4
    config.model.num_active_experts = 2

    router = SparseMoERouter(config).to(device=device, dtype=torch.float64)
    experts = MoEExperts(config).to(device=device, dtype=torch.float64)

    # Disable router exploration noise for clean double-precision gradcheck
    router.noise_enabled = False

    x = torch.randn(2, 4, 8, device=device, dtype=torch.float64, requires_grad=True)

    def full_pipeline(tensor: torch.Tensor) -> torch.Tensor:
        w, idx, _ = router(tensor)
        out, _ = experts(tensor, w, idx)
        return typing.cast(torch.Tensor, out)

    assert torch.autograd.gradcheck(full_pipeline, (x,), eps=1e-6, atol=1e-4)


def test_moe_entropy_conditions_routing_with_fixed_hidden(device: str) -> None:
    """Entropy perturbation must change expert indices when hidden states are fixed."""
    config = get_base_config()
    config.model.hidden_dim = 32
    config.model.num_experts = 4
    config.model.num_active_experts = 2
    router = SparseMoERouter(config).to(device=device)
    router.eval()
    router.noise_enabled = False

    b, s, d = 2, 8, 32
    x = torch.randn(b, s, d, device=device)
    ent_low = torch.zeros(b, s, 1, device=device)
    ent_high = torch.ones(b, s, 1, device=device)

    with torch.no_grad():
        logits_low, _, _, _, _ = router._gating_logits(x, entropy=ent_low)
        logits_high, _, _, _, _ = router._gating_logits(x, entropy=ent_high)
        _, idx_low, _ = router(x, entropy=ent_low)
        _, idx_high, _ = router(x, entropy=ent_high)

    assert not torch.allclose(logits_low, logits_high)
    assert not torch.equal(idx_low, idx_high)


@pytest.mark.parametrize("device", get_test_devices())
def test_moe_flop_savings_invariant(device: str) -> None:
    """Verify that routing 2 active experts out of 4 yields exactly 50% FLOP savings."""
    config = get_base_config()
    config.model.num_experts = 4
    config.model.num_active_experts = 2
    experts = MoEExperts(config).to(device=device)

    # Inputs
    b, s, d = 4, 32, 256
    x = torch.randn(b, s, d, device=device)

    # Mock uniform routing inputs to ensure no capacity drops occur
    weights = torch.ones(b, s, 2, device=device) * 0.5
    indices = torch.zeros(b, s, 2, device=device, dtype=torch.long)
    indices_flat = indices.view(-1, 2)
    # 128 total tokens. Route first half to [0, 1] and second half to [2, 3]
    indices_flat[:64, 0] = 0
    indices_flat[:64, 1] = 1
    indices_flat[64:, 0] = 2
    indices_flat[64:, 1] = 3

    _, metrics = experts(x, weights, indices)

    # Assert exactly 50% sparse execution savings (no drops)
    assert metrics["flop_savings_pct"] == 50.0
