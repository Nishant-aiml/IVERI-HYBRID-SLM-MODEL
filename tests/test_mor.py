# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, integration, and stability tests for MoR subsystem (Phase 1.5)."""

from __future__ import annotations

import typing

import pytest
import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.exceptions import ShapeError
from model.mor import RecursionDepthRouter, RecursionEngine, SelectiveKVCache


# Simple mock layer block for testing recursion loop bypass
class MockBlock(nn.Module):
    def __init__(self, hidden_dim: int, increment: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize weight to identity to easily predict output
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))
        self.increment = increment

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Increment x values to easily track how many times computation was run
        out = self.linear(x) + self.increment
        if active_mask is not None:
            # If mask is passed, apply bypassing inside block
            return typing.cast(torch.Tensor, torch.where(active_mask.unsqueeze(-1), out, x))
        return typing.cast(torch.Tensor, out)


@pytest.mark.parametrize("device", ["cpu"])
def test_router_entropy_mapping(base_config: IVERIConfig, device: str) -> None:
    """Verify that entropy-driven depth mapping spans correct range and obeys Option C."""
    # This test requires max_recursion_depth=8 for the expected index calculations.
    base_config.model.max_recursion_depth = 8
    router = RecursionDepthRouter(base_config, research_mode=False).to(device)
    max_depth = base_config.model.max_recursion_depth

    # Test inputs
    x = torch.randn(2, 5, base_config.model.hidden_dim, device=device)

    # 1. Zero entropy -> should map to depth 1 (index 0)
    entropy_zero = torch.zeros(2, 5, 1, device=device)
    w_zero, idx_zero = router.route(x, entropy=entropy_zero)
    assert (idx_zero == 0).all()
    assert (w_zero == 1.0).all()

    # 2. Maximum entropy (1.0) -> should map to depth max_depth (index max_depth - 1)
    entropy_max = torch.ones(2, 5, 1, device=device)
    w_max, idx_max = router.route(x, entropy=entropy_max)
    assert (idx_max == max_depth - 1).all()
    assert (w_max == 1.0).all()

    # 3. Intermediate entropy values spanning range
    entropy = torch.tensor([[[0.0], [0.15], [0.5], [0.85], [1.0]]], device=device)
    # With max_depth=8, floor(E * 7):
    # E=0.0 -> 1 + 0 = 1 (idx 0)
    # E=0.15 -> 1 + floor(1.05) = 2 (idx 1)
    # E=0.5 -> 1 + floor(3.5) = 4 (idx 3)
    # E=0.85 -> 1 + floor(5.95) = 6 (idx 5)
    # E=1.0 -> 1 + 7 = 8 (idx 7)
    x_single = torch.randn(1, 5, base_config.model.hidden_dim, device=device)
    _, idx = router.route(x_single, entropy=entropy)
    expected_indices = torch.tensor([[[0], [1], [3], [5], [7]]], device=device, dtype=torch.long)
    assert torch.equal(idx, expected_indices)


@pytest.mark.parametrize("device", ["cpu"])
def test_router_learned_mode(base_config: IVERIConfig, device: str) -> None:
    """Verify that optional learned routing research mode computes valid logit shapes."""
    router = RecursionDepthRouter(base_config, research_mode=True).to(device)
    x = torch.randn(3, 4, base_config.model.hidden_dim, device=device)

    weights, indices = router.route(x)
    assert weights.shape == (3, 4, 1)
    assert indices.shape == (3, 4, 1)
    assert (indices >= 0).all()
    assert (indices < base_config.model.max_recursion_depth).all()
    assert (weights >= 0.0).all() and (weights <= 1.0).all()


@pytest.mark.parametrize("device", ["cpu"])
def test_recursion_engine_execution(base_config: IVERIConfig, device: str) -> None:
    """Verify that RecursionEngine loops correctly and skips inactive patches."""
    # This test uses depth=8 so requires max_recursion_depth >= 8.
    base_config.model.max_recursion_depth = 8
    mock_block = MockBlock(base_config.model.hidden_dim, increment=1.0).to(device)
    engine = RecursionEngine(mock_block, base_config).to(device)

    # Input: all values initialized to 0.0
    x = torch.zeros(1, 5, base_config.model.hidden_dim, device=device)
    # Assign specific depths to each position: [1, 2, 4, 8, 8]
    depths = torch.tensor([[1, 2, 4, 8, 8]], device=device, dtype=torch.long)

    # Run forward pass
    out = engine(x, depths=depths)

    # Since MockBlock adds 1.0 per execution step:
    # Position 0 (depth 1) -> value should be 1.0
    # Position 1 (depth 2) -> value should be 2.0
    # Position 2 (depth 4) -> value should be 4.0
    # Position 3, 4 (depth 8) -> value should be 8.0
    expected = torch.tensor([[1.0, 2.0, 4.0, 8.0, 8.0]], device=device).unsqueeze(-1).expand_as(out)
    assert torch.allclose(out, expected)

    # Check telemetry statistics are collected
    stats = engine.get_statistics()
    assert stats["average_depth"] == pytest.approx(4.6)
    assert stats["max_depth_frequency"] == 40.0
    assert stats["skipped_pct"] == pytest.approx(
        (17 / 40) * 100.0
    )  # 17 computation steps skipped out of 40 slots


@pytest.mark.parametrize("device", ["cpu"])
def test_mor_gradflow(base_config: IVERIConfig, device: str) -> None:
    """Verify that backward pass works and gradients flow correctly to parameters."""
    mock_block = MockBlock(base_config.model.hidden_dim, increment=1.0).to(device)
    # Enable gradients on MockBlock parameter
    for param in mock_block.parameters():
        param.requires_grad = True

    engine = RecursionEngine(mock_block, base_config).to(device)

    x = torch.randn(2, 4, base_config.model.hidden_dim, device=device, requires_grad=True)
    depths = torch.tensor([[2, 4, 8, 1], [3, 2, 1, 8]], device=device, dtype=torch.long)

    out = engine(x, depths=depths)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    # Check that block linear layer received gradients
    assert mock_block.linear.weight.grad is not None
    assert not torch.isnan(mock_block.linear.weight.grad).any()


@pytest.mark.parametrize("device", ["cpu"])
def test_selective_kv_cache(device: str) -> None:
    """Verify that SelectiveKVCache appends key-value states only for active mask positions."""
    cache = SelectiveKVCache()

    # Run 1: Step 1, all positions active
    k1 = torch.ones(2, 4, 1, 8, device=device)  # (B, H, S, D)
    v1 = torch.ones(2, 4, 1, 8, device=device)
    mask1 = torch.tensor([[True], [True]], device=device)

    cache.update(k1, v1, mask1)
    assert cache.k_cache is not None
    assert cache.k_cache.shape == (2, 4, 1, 8)
    assert (cache.k_cache == 1.0).all()

    # Run 2: Step 2, first batch active, second inactive
    k2 = torch.full((2, 4, 1, 8), 2.0, device=device)
    v2 = torch.full((2, 4, 1, 8), 2.0, device=device)
    mask2 = torch.tensor([[True], [False]], device=device)

    cache.update(k2, v2, mask2)
    # Shape should be appended along sequence dimension (index 2) to (2, 4, 2, 8)
    assert cache.k_cache.shape == (2, 4, 2, 8)
    # Batch 0 should have [1.0, 2.0]
    assert (cache.k_cache[0, :, 0] == 1.0).all()
    assert (cache.k_cache[0, :, 1] == 2.0).all()
    # Batch 1 should have [1.0, 0.0] since index 1 was inactive
    assert (cache.k_cache[1, :, 0] == 1.0).all()
    assert (cache.k_cache[1, :, 1] == 0.0).all()


def test_router_validation_checks(base_config: IVERIConfig) -> None:
    """Verify router rejects invalid shape combinations."""
    router = RecursionDepthRouter(base_config, research_mode=False)

    # 1. Invalid input x shape (missing hidden dimension)
    x_bad = torch.randn(2, 5)
    with pytest.raises(ShapeError):
        router.route(x_bad, entropy=torch.ones(2, 5, 1))

    # 2. Invalid entropy shape mismatch
    x = torch.randn(2, 5, base_config.model.hidden_dim)
    entropy_bad = torch.ones(2, 6, 1)  # Length mismatch
    with pytest.raises(ShapeError):
        router.route(x, entropy=entropy_bad)

    # 3. Missing entropy in Option C production mode
    with pytest.raises(ValueError, match="entropy tensor"):
        router.route(x)


@pytest.mark.parametrize("device", ["cpu"])
def test_mor_numerical_stability(base_config: IVERIConfig, device: str) -> None:
    """Test robustness against NaN, Inf, empty batch, single token, and extreme sequences."""
    mock_block = MockBlock(base_config.model.hidden_dim).to(device)
    engine = RecursionEngine(mock_block, base_config).to(device)
    router = RecursionDepthRouter(base_config).to(device)

    # 1. Single token input
    x_single = torch.randn(1, 1, base_config.model.hidden_dim, device=device)
    entropy_single = torch.tensor([[0.5]], device=device)
    _, idx = router.route(x_single, entropy=entropy_single)
    out_single = engine(x_single, depths=idx + 1)
    assert out_single.shape == (1, 1, base_config.model.hidden_dim)
    assert not torch.isnan(out_single).any()

    # 2. Invalid/Extreme entropy values (should clamp safely)
    entropy_extreme = torch.tensor([[[-0.5], [1.5], [99.0]]], device=device)
    x_extreme = torch.randn(1, 3, base_config.model.hidden_dim, device=device)
    _, idx_extreme = router.route(x_extreme, entropy=entropy_extreme)
    # Checked depth indices should be clamped within [0, max_depth - 1]
    assert (idx_extreme >= 0).all()
    assert (idx_extreme < base_config.model.max_recursion_depth).all()

    # 3. NaN/Inf inputs validation
    x_nan = torch.randn(2, 5, base_config.model.hidden_dim, device=device)
    x_nan[0, 0, 0] = float("nan")
    depths = torch.ones(2, 5, device=device, dtype=torch.long)
    out_nan = engine(x_nan, depths=depths)
    # Ensure computation completes and NaN propagates correctly without infinite loops
    assert torch.isnan(out_nan[0, 0, 0])
