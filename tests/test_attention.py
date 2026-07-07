# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, shape, causal masking, KV cache, and mixed precision tests for FlashAttentionWrapper (Phase 1.4)."""

from __future__ import annotations

import pytest
import torch

from configs.base_config import get_base_config
from model.attention import FlashAttentionWrapper


def get_test_devices() -> list[str]:
    """Get available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", get_test_devices())
def test_attention_forward_shape(device: str) -> None:
    """Verify input-output shape consistency (B, S, D) -> (B, S, D)."""
    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device)

    b, s, d = 2, 64, 256
    x = torch.randn(b, s, d, device=device)

    out = wrapper(x)
    assert out.shape == (b, s, d)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("device", get_test_devices())
def test_attention_gradflow(device: str) -> None:
    """Verify that autograd gradients flow cleanly to all projection weights."""
    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device, requires_grad=True)

    out = wrapper(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()

    # Gradients on projections check
    assert wrapper.qkv_proj.weight.grad is not None
    assert not torch.isnan(wrapper.qkv_proj.weight.grad).any()
    assert wrapper.out_proj.weight.grad is not None
    assert not torch.isnan(wrapper.out_proj.weight.grad).any()


@pytest.mark.parametrize("device", get_test_devices())
def test_attention_causal_masking(device: str) -> None:
    """Ensure causal masking properties hold (future steps cannot affect past steps)."""
    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device)
    wrapper.eval()

    b, s, d = 1, 8, 256
    x1 = torch.randn(b, s, d, device=device)
    x2 = x1.clone()

    # Alter the last token of the second sequence
    x2[:, -1, :] = torch.randn(b, d, device=device)

    with torch.no_grad():
        out1 = wrapper(x1, is_causal=True)
        out2 = wrapper(x2, is_causal=True)

    # Outputs for tokens 0..S-2 should be identical
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_attention_kv_caching(device: str) -> None:
    """Verify that incremental decoding matches full sequence forward pass exactly."""
    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device)
    wrapper.eval()

    b, s, d = 1, 8, 256
    x = torch.randn(b, s, d, device=device)

    # 1. Full sequence forward pass (Pre-fill)
    with torch.no_grad():
        y_full = wrapper(x, is_causal=True)

    # 2. Incremental decode step-by-step
    kv_cache: dict[str, torch.Tensor] = {}
    y_steps = []

    with torch.no_grad():
        for t in range(s):
            x_t = x[:, t : t + 1, :]  # Shape: (B, 1, D)
            y_t = wrapper(x_t, kv_cache=kv_cache, is_causal=True)
            y_steps.append(y_t)

    y_incremental = torch.cat(y_steps, dim=1)

    # Assert outputs are mathematically equivalent
    assert torch.allclose(y_full, y_incremental, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_attention_reset_parameters(device: str) -> None:
    """Ensure parameters change successfully on reset."""
    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device)

    orig_weight = wrapper.qkv_proj.weight.clone()
    wrapper.reset_parameters()

    assert not torch.equal(orig_weight, wrapper.qkv_proj.weight)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_mixed_precision(device: str, dtype: torch.dtype) -> None:
    """Validate layer under different mixed-precision dtypes."""
    if device == "cpu" and dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Half precision not supported on CPU for SDPA")

    config = get_base_config()
    wrapper = FlashAttentionWrapper(config).to(device=device, dtype=dtype)

    b, s, d = 2, 16, 256
    x = torch.randn(b, s, d, device=device, dtype=dtype)

    out = wrapper(x)
    assert out.shape == (b, s, d)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
