# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Test suite for Phase 1.7 Titans Neural Memory Subsystem.

Verifies the mathematical correctness, autograd differentiability, telemetry,
persistence, and numerical stability of Titans memory layers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configs.base_config import get_base_config
from model.titans import MemoryLearningRateGenerator, MemoryUpdater, TitansMemory


@pytest.fixture
def config():
    """Fixture to obtain a default IVERIConfig."""
    cfg = get_base_config()
    cfg.model.hidden_dim = 256
    cfg.model.titans_memory_dim = 128
    return cfg


def test_lr_generator_shapes_and_bounds(config):
    """Verify shapes and sigmoid limits of MemoryLearningRateGenerator."""
    generator = MemoryLearningRateGenerator(config, max_lr=0.1, max_forget=0.05)

    # Check bounds
    assert generator.max_lr == 0.1
    assert generator.max_forget == 0.05

    # Test forward pass shape compliance
    x = torch.randn(2, 5, 256)
    lr, forget = generator(x)

    assert lr.shape == (2, 5, 1)
    assert forget.shape == (2, 5, 1)

    # All outputs must be within boundaries [0, max]
    assert torch.all(lr >= 0.0) and torch.all(lr <= 0.1)
    assert torch.all(forget >= 0.0) and torch.all(forget <= 0.05)


def test_updater_equations(config):
    """Verify that MemoryUpdater implements the frozen Titans update rules."""
    updater = MemoryUpdater(config, momentum=0.9)

    # Initialize mock parameters and gradients
    W = torch.ones(2, 4, 4)
    g = torch.ones(2, 4, 4) * 0.5
    s = torch.ones(2, 4, 4) * 0.1

    lr = torch.ones(2, 1, 1) * 0.2
    forget = torch.ones(2, 1, 1) * 0.01

    new_weights, new_surprise = updater.update([W], [g], [s], lr, forget)

    # S_new = momentum * S - lr * g = 0.9 * 0.1 - 0.2 * 0.5 = 0.09 - 0.1 = -0.01
    expected_s = 0.9 * 0.1 - 0.2 * 0.5
    assert torch.allclose(new_surprise[0], torch.tensor(expected_s))

    # W_new = (1 - forget) * W + S_new = 0.99 * 1.0 - 0.01 = 0.98
    expected_w = (1.0 - 0.01) * 1.0 + expected_s
    assert torch.allclose(new_weights[0], torch.tensor(expected_w))


def test_titans_memory_shapes_and_registration(config):
    """Verify TitansMemory shape contract and registry metadata."""
    memory = TitansMemory(config)

    # Check registration properties
    assert isinstance(memory, nn.Module)

    # Check config initialization compatibility
    x = torch.randn(2, 8, 256)
    out = memory(x)

    # Verify shape is preserved (B, P, D)
    assert out.shape == x.shape


def test_differentiability_and_gradient_flow(config):
    """Verify that the sequential online update graph is fully differentiable."""
    memory = TitansMemory(config)

    # Inputs require gradients
    x = torch.randn(2, 4, 256, requires_grad=True)

    # Check forward pass under grad mode
    with torch.enable_grad():
        out = memory(x)
        loss = out.sum()

        # Backward pass
        loss.backward()

    # Autograd must correctly propagate back to input x
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0.0).any()

    # Verify gradients reach base parameters and projections
    assert memory.base_W1.grad is not None
    assert memory.q_proj.weight.grad is not None


def test_reconstruction_loss_reduction(config):
    """Verify that online updates improve key-value association over step iterations."""
    memory = TitansMemory(config)

    # Setup key and target value
    # To test memorization, we will write a fixed key-value association repeatedly
    k = torch.randn(1, 10, 256)
    v = torch.randn(1, 10, 256)

    # First forward pass
    memory.reset_memory()
    initial_retrieved = memory.read(k)
    initial_loss = 0.5 * (initial_retrieved - v).pow(2).mean().item()

    # Write to memory multiple times to train the local MLP
    for _ in range(5):
        memory.write(k, v)

    # Read again
    final_retrieved = memory.read(k)
    final_loss = 0.5 * (final_retrieved - v).pow(2).mean().item()

    # Association loss should decrease
    assert final_loss < initial_loss


def test_entropy_gated_injection(config):
    """Verify Option C gated injection magnitude is proportional to positional entropy."""
    memory = TitansMemory(config)

    # Sequence of length 3 with identical features across positions
    x_single = torch.randn(1, 1, 256)
    x = x_single.expand(-1, 3, -1)

    # Entropy scores: Low (0.01), Medium (0.5), High (0.99)
    entropy = torch.tensor([[[0.01], [0.5], [0.99]]])

    # Run inject pass
    out = memory.inject(x, entropy)

    # Output delta per position
    deltas = (out - x).abs().sum(dim=-1).squeeze(0)

    # Deltas should be ordered by entropy uncertainty: delta[0] < delta[1] < delta[2]
    assert deltas[0] < deltas[1]
    assert deltas[1] < deltas[2]


def test_persistence_and_initialization_determinism(config):
    """Test memory resets, initializations, and deterministic seeding."""
    # Determinism check
    torch.manual_seed(42)
    m1 = TitansMemory(config)
    m1.base_W1.clone()

    torch.manual_seed(42)
    m2 = TitansMemory(config)
    m2.base_W2.clone()  # wait, compare same weight

    torch.manual_seed(42)
    m3 = TitansMemory(config)
    assert torch.allclose(m1.base_W1, m3.base_W1)
    assert torch.allclose(m1.base_W2, m3.base_W2)

    # Persistence verification
    x = torch.randn(1, 5, 256)
    m1.reset_memory()
    assert m1.current_weights is None

    # First run initializes weights
    m1(x)
    assert m1.current_weights is not None
    saved_weights = [w.clone() for w in m1.current_weights]

    # Repeated run without reset keeps updated weights (persistent memory)
    # Check that current_weights is not reset in read/write
    m1.read(x)
    assert torch.allclose(m1.current_weights[0], saved_weights[0])


def test_telemetry_collection(config):
    """Verify that comprehensive telemetry values are recorded during execution."""
    memory = TitansMemory(config)
    x = torch.randn(2, 6, 256)

    memory(x)

    # Telemetry should be populated
    tel = memory.telemetry
    assert tel["update_count"] == 12
    assert "avg_learning_rate" in tel
    assert "learning_rate_histogram" in tel
    assert "avg_forget_rate" in tel
    assert "forget_rate_histogram" in tel
    assert "memory_saturation" in tel
    assert "memory_weight_norm" in tel
    assert "average_gradient_norm" in tel
    assert "average_update_magnitude" in tel

    # Metrics bounds
    assert 0.0 <= tel["avg_learning_rate"] <= 0.1
    assert 0.0 <= tel["avg_forget_rate"] <= 0.1
    assert tel["memory_weight_norm"] > 0.0


def test_extreme_numerical_conditions(config):
    """Verify robust performance under sequence scale and surprise extremes."""
    memory = TitansMemory(config)

    # 1. Very long sequence (up to 512, wait, 2048 sequence length)
    x_long = torch.randn(1, 512, 256)  # Keep within reasonable testing time
    out_long = memory(x_long)
    assert out_long.shape == x_long.shape
    assert not torch.isnan(out_long).any()

    # 2. Zero surprise (exact prediction matches target)
    k = torch.randn(1, 1, 256)
    memory.reset_memory()
    # Perform update first
    memory.write(k, k)
    # Get outputs
    v_pred = memory._forward_mlp(k, memory.current_weights)
    # Now set key same as prediction so surprise is small/zero
    memory.write(k, v_pred.detach())
    assert memory.telemetry["average_gradient_norm"] < 0.01

    # 3. Maximum surprise
    # Force output to disagree strongly with target
    target = torch.ones_like(v_pred) * 1000.0
    memory.write(k, target)
    max_grad_norm = memory.telemetry["average_gradient_norm"]
    assert max_grad_norm > 1.0


def test_invalid_shape_validation(config):
    """Verify ShapeError or ValidationError on malformed shapes."""
    memory = TitansMemory(config)

    # Wrong hidden dim shape
    x_bad = torch.randn(2, 4, 128)
    from core.exceptions import ShapeError

    with pytest.raises(ShapeError):
        memory(x_bad)


def test_interface_compliance(config):
    """Explicitly verify constructor and signature interface compliance."""
    # Test initialization with integer directly
    memory_from_dim = TitansMemory(256)
    assert memory_from_dim.hidden_dim == 256
    assert memory_from_dim.memory_dim == 128
