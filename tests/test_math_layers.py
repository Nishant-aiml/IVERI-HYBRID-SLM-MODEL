# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive unit, shape, gradient, stability, stress, and compatibility tests

for Phase 1.1 Core Mathematical Layers (RMSNorm, RoPE, SwiGLU).
"""

from __future__ import annotations

import math

import pytest
import torch

from model.norms import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_emb
from model.swiglu import SwiGLU, SwiGLUFFN

# --- Helpers ---------------------------------------------------------------


def get_test_devices() -> list[str]:
    """Get list of available devices for testing."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


# --- RMSNorm Tests ---------------------------------------------------------


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_forward_shape_and_dtype(device: str, dtype: torch.dtype) -> None:
    """Verify that RMSNorm output shape and dtype match inputs."""
    dim = 256
    norm = RMSNorm(dim=dim, eps=1e-6).to(device=device, dtype=dtype)

    x = torch.randn(2, 16, dim, device=device, dtype=dtype)
    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize("device", get_test_devices())
def test_rmsnorm_mathematical_correctness(device: str) -> None:
    """Verify RMSNorm output matches the mathematical definition."""
    dim = 4
    norm = RMSNorm(dim=dim, eps=1e-6).to(device=device)

    # Set weight to a known custom value
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0], device=device))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)

    # Expected RMS = sqrt((1 + 4 + 9 + 16)/4 + 1e-6) = sqrt(7.5 + 1e-6)
    rms = math.sqrt(7.5 + 1e-6)
    expected = (x / rms) * torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

    out = norm(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_rmsnorm_gradient_flow(device: str) -> None:
    """Verify gradients flow correctly through RMSNorm."""
    dim = 64
    norm = RMSNorm(dim=dim).to(device=device)
    x = torch.randn(4, 16, dim, device=device, requires_grad=True)

    out = norm(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.parametrize("device", get_test_devices())
def test_rmsnorm_numerical_stability(device: str) -> None:
    """Verify RMSNorm is stable with extreme value bounds."""
    dim = 64
    norm = RMSNorm(dim=dim).to(device=device)

    # Large inputs
    x_large = torch.randn(2, 8, dim, device=device) * 1e4
    out_large = norm(x_large)
    assert not torch.isnan(out_large).any()
    assert not torch.isinf(out_large).any()

    # Small inputs
    x_small = torch.randn(2, 8, dim, device=device) * 1e-4
    out_small = norm(x_small)
    assert not torch.isnan(out_small).any()
    assert not torch.isinf(out_small).any()


# --- RoPE Tests ------------------------------------------------------------


@pytest.mark.parametrize("device", get_test_devices())
def test_rope_rotation_correctness(device: str) -> None:
    """Verify that applying RoPE performs the expected rotation angles."""
    dim = 2
    rope = RotaryEmbedding(dim=dim, max_seq_len=10).to(device=device)

    # Angle theta = 10000^0 = 1.0
    # For position m = 1, rotation angle = 1.0 radian
    # cos(1) = 0.5403, sin(1) = 0.8415
    # Input vector: [1.0, 0.0]
    # Rotated vector should be [cos(1), sin(1)] = [0.5403, 0.8415]

    x = torch.tensor([[[[1.0, 0.0]]]], device=device)  # (B=1, H=1, S=1, D_head=2)
    cos, sin = rope(x, seq_len=2)

    # Apply rotation at position index 1
    # cos, sin shape: (2, 2)
    c = cos[1:2].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 2)
    s = sin[1:2].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 2)

    rotated = apply_rotary_emb(x, c, s)
    expected = torch.tensor([[[[math.cos(1.0), math.sin(1.0)]]]], device=device)

    torch.testing.assert_close(rotated, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_rope_dynamic_extension(device: str) -> None:
    """Verify RoPE auto-extends frequency cache on demand."""
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=10).to(device=device)

    assert rope.max_seq_len == 10

    x = torch.randn(1, 1, 20, dim, device=device)
    cos, sin = rope(x, seq_len=20)

    assert rope.max_seq_len == 20
    assert cos.shape == (20, dim)
    assert sin.shape == (20, dim)


# --- SwiGLU Tests ----------------------------------------------------------


@pytest.mark.parametrize("device", get_test_devices())
def test_swiglu_activation_correctness(device: str) -> None:
    """Verify that SwiGLU class computes Silu(gate) * value."""
    swiglu = SwiGLU().to(device=device)

    gate = torch.tensor([1.0, 2.0], device=device)
    val = torch.tensor([3.0, 4.0], device=device)

    # Swish(gate) = gate * sigmoid(gate)
    expected = (gate * torch.sigmoid(gate)) * val
    out_with_value = swiglu(gate, value=val)
    torch.testing.assert_close(out_with_value, expected)

    # Test chunk split behavior
    out_split = swiglu(torch.cat([gate, val], dim=-1))
    torch.testing.assert_close(out_split, expected)


@pytest.mark.parametrize("device", get_test_devices())
def test_swiglu_ffn_dimension_rounding(device: str) -> None:
    """Verify hidden dimension computes correctly when None."""
    dim = 256
    # 2/3 * 4 * 256 = 682.66 -> Rounded up to nearest multiple of 256 is 768
    ffn = SwiGLUFFN(dim=dim).to(device=device)
    assert ffn.w_gate.out_features == 768
    assert ffn.w_value.out_features == 768


# --- Interface, Stress, and Robustness Tests --------------------------------


@pytest.mark.parametrize("device", get_test_devices())
def test_interfaces_and_reset_parameters(device: str) -> None:
    """Verify BaseModule abstract implementation compliance."""
    dim = 64
    rmsnorm = RMSNorm(dim=dim).to(device=device)
    swiglu_ffn = SwiGLUFFN(dim=dim).to(device=device)

    # Verify extra_repr returns non-empty string
    assert len(rmsnorm.extra_repr()) > 0
    assert len(swiglu_ffn.extra_repr()) > 0

    # Verify reset_parameters execution
    rmsnorm.reset_parameters()
    swiglu_ffn.reset_parameters()


def test_seed_determinism() -> None:
    """Verify seeded runs produce identical floating outputs."""
    torch.manual_seed(1337)
    norm1 = RMSNorm(dim=128)
    x = torch.randn(2, 10, 128)
    out1 = norm1(x)

    torch.manual_seed(1337)
    norm2 = RMSNorm(dim=128)
    out2 = norm2(x)

    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256),  # Minimal
        (1, 16, 256),  # Small batch
        (32, 512, 256),  # Standard
        (64, 512, 256),  # Large batch
        (4, 2048, 256),  # Long sequence
        (2, 128, 2048),  # Large hidden dimension
    ],
)
def test_stress_matrix_shapes(device: str, shape: tuple[int, int, int]) -> None:
    """Run stress testing across different dimensions."""
    b, s, d = shape
    rmsnorm = RMSNorm(dim=d).to(device=device)
    swiglu_ffn = SwiGLUFFN(dim=d).to(device=device)

    x = torch.randn(b, s, d, device=device)

    # 1. RMSNorm Shape and Stability
    out_norm = rmsnorm(x)
    assert out_norm.shape == shape
    assert not torch.isnan(out_norm).any()

    # 2. SwiGLU FFN Shape and Stability
    out_ffn = swiglu_ffn(x)
    assert out_ffn.shape == shape
    assert not torch.isnan(out_ffn).any()


@pytest.mark.parametrize("device", get_test_devices())
def test_rmsnorm_gradcheck(device: str) -> None:
    """Rigorous analytical vs numerical gradient checking for RMSNorm."""
    dim = 8
    # gradcheck requires double precision for numerical stability
    norm = RMSNorm(dim=dim).to(device=device, dtype=torch.float64)
    x = torch.randn(2, 4, dim, device=device, dtype=torch.float64, requires_grad=True)

    # Check gradients of output w.r.t input
    assert torch.autograd.gradcheck(norm, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_swiglu_ffn_gradcheck(device: str) -> None:
    """Rigorous analytical vs numerical gradient checking for SwiGLUFFN."""
    dim = 8
    # gradcheck requires double precision for numerical stability
    ffn = SwiGLUFFN(dim=dim, hidden_dim=16).to(device=device, dtype=torch.float64)
    x = torch.randn(2, 4, dim, device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(ffn, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("device", get_test_devices())
def test_rmsnorm_scale_invariance_property(device: str) -> None:
    """Property test: Verify RMSNorm output is invariant to positive scaling of input."""
    dim = 64
    norm = RMSNorm(dim=dim).to(device=device)

    # Fix seed locally for random generations
    g = torch.Generator(device=device)
    g.manual_seed(42)

    for _ in range(10):
        # Generate random inputs and a random positive scale factor
        x = torch.randn(2, 8, dim, device=device, generator=g)
        scale = torch.rand(1, device=device, generator=g).item() * 10.0 + 0.1

        out1 = norm(x)
        out2 = norm(x * scale)

        # Norm should be invariant to scale because scale factors cancel out:
        # RMSNorm(c * x) = (c * x) / (c * RMS(x)) * weight = RMSNorm(x)
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", get_test_devices())
def test_rope_norm_preservation_property(device: str) -> None:
    """Property test: Verify RoPE rotation preserves vector length (L2 norm)."""
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=100).to(device=device)

    g = torch.Generator(device=device)
    g.manual_seed(42)

    for _ in range(10):
        # Shape: (B, H, S, D_head)
        x = torch.randn(2, 4, 10, dim, device=device, generator=g)
        cos, sin = rope(x)

        # Apply RoPE
        # Broadcast cos/sin shape (S, D_head) to (1, 1, S, D_head)
        c = cos.unsqueeze(0).unsqueeze(1)
        s = sin.unsqueeze(0).unsqueeze(1)
        rotated = apply_rotary_emb(x, c, s)

        # Rotations preserve the L2 norm of the vectors
        norm_orig = torch.linalg.vector_norm(x, ord=2, dim=-1)
        norm_rot = torch.linalg.vector_norm(rotated, ord=2, dim=-1)

        torch.testing.assert_close(norm_orig, norm_rot, rtol=1e-5, atol=1e-5)
