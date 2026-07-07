# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, shape, gradient, parameter, mixed precision, and stress tests for Mamba2Block (Wave 3)."""

from __future__ import annotations

import pytest
import torch

from configs.base_config import get_base_config
from model.mamba2 import Mamba2Block


def get_test_devices() -> list[str]:
    """Get available devices for testing."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_mamba2_block_forward_shapes(device: str) -> None:
    """Verify that Mamba2Block forward pass matches expected input shapes."""
    config = get_base_config()
    block = Mamba2Block(config).to(device=device)

    # 10M Nano configuration defaults: B=2, S=32, D=256
    b, s, d = 2, 32, 256
    x = torch.randn(b, s, d, device=device)

    out = block(x)

    assert out.shape == (b, s, d)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("device", get_test_devices())
def test_mamba2_block_gradient_flow(device: str) -> None:
    """Verify end-to-end backpropagation and gradient flow to all parameters."""
    config = get_base_config()
    block = Mamba2Block(config).to(device=device)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device, requires_grad=True)

    out = block(x)
    loss = out.sum()
    loss.backward()

    # Input gradient checked
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()

    # Parameter gradients checked
    for name, p in block.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaNs"


@pytest.mark.parametrize("device", get_test_devices())
def test_mamba2_block_reset_parameters(device: str) -> None:
    """Ensure parameter initialization updates weights correctly."""
    config = get_base_config()
    block = Mamba2Block(config).to(device=device)

    # Record initial weights
    orig_in_proj = block.in_proj.weight.clone()
    orig_A_log = block.A_log.clone()

    block.reset_parameters()

    # Assert weights changed after reset
    assert not torch.equal(orig_in_proj, block.in_proj.weight)
    assert not torch.equal(orig_A_log, block.A_log)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mamba2_block_mixed_precision(device: str, dtype: torch.dtype) -> None:
    """Test block execution under various floating-point precisions."""
    # Skip half-precision on CPU (not natively supported for some ops like Conv1d)
    if device == "cpu" and dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Half precision not natively supported on CPU for Conv1d")

    config = get_base_config()
    block = Mamba2Block(config).to(device=device, dtype=dtype)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device, dtype=dtype)

    out = block(x)

    assert out.shape == (b, s, d)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize(
    "b, s, d_model",
    [
        (1, 128, 256),
        (4, 512, 256),
        (2, 1024, 256),
    ],
)
def test_mamba2_block_stress_shapes(device: str, b: int, s: int, d_model: int) -> None:
    """Stress test execution across various batch sizes and sequence dimensions."""
    config = get_base_config()
    config.model.hidden_dim = d_model
    block = Mamba2Block(config).to(device=device)

    x = torch.randn(b, s, d_model, device=device)
    out = block(x)

    assert out.shape == (b, s, d_model)
    assert not torch.isnan(out).any()
