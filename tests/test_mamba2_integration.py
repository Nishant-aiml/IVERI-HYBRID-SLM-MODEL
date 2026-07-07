# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration and regression tests for Mamba2 (Wave 4)."""

from __future__ import annotations

import pytest
import torch

from configs.base_config import get_base_config
from model.mamba2 import Mamba2Block
from model.norms import RMSNorm


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_mamba2_and_math_layers_coexistence(device: str) -> None:
    """Verify that Mamba2 block executes cleanly in a pipeline with Norms and RoPE."""
    config = get_base_config()
    mamba_block = Mamba2Block(config).to(device=device)
    norm = RMSNorm(dim=256).to(device=device)

    b, s, d = 2, 64, 256
    x = torch.randn(b, s, d, device=device)

    # Executing norm
    x_norm = norm(x)
    # Executing Mamba2 block
    x_mamba = mamba_block(x_norm)

    assert x_mamba.shape == (b, s, d)
    assert not torch.isnan(x_mamba).any()


@pytest.mark.parametrize("device", get_test_devices())
def test_mamba2_backward_with_norms(device: str) -> None:
    """Ensure joint backpropagation works correctly across Norm and Mamba2 layers."""
    config = get_base_config()
    mamba_block = Mamba2Block(config).to(device=device)
    norm = RMSNorm(dim=256).to(device=device)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device, requires_grad=True)

    out = mamba_block(norm(x))
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
