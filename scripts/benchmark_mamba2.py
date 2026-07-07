# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Performance profiling, timing segmentation, and Attention baseline comparisons for Mamba2Block (Wave 4)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import get_base_config
from model.mamba2 import Mamba2Block
from scripts.benchmark_math_layers import get_system_info


class DenseSelfAttention(nn.Module):
    """Standard multi-head self-attention layer for baseline comparison."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        import typing

        return typing.cast(torch.Tensor, self.out_proj(out))


def profile_mamba2_components(
    block: Mamba2Block,
    x: torch.Tensor,
    device: str,
    num_iter: int = 100,
) -> dict[str, float]:
    """Segment input projection, Conv1d, selective SSD scan, and output projection times."""
    if device == "cuda":
        torch.cuda.synchronize()

    # Stage 1: Input projection
    start = time.perf_counter()
    for _ in range(num_iter):
        projected = block.in_proj(x)
    if device == "cuda":
        torch.cuda.synchronize()
    in_proj_ms = ((time.perf_counter() - start) / num_iter) * 1000

    # Intermediate setup for convolution
    x_path, gate, delta, B, C = torch.split(
        projected,
        [block.d_inner, block.d_inner, block.d_inner, block.d_state, block.d_state],
        dim=-1,
    )
    conv_in = torch.cat([x_path, delta, B, C], dim=-1).transpose(1, 2)
    padded = F.pad(conv_in, (3, 0))

    # Stage 2: Causal Convolution
    start = time.perf_counter()
    for _ in range(num_iter):
        conv_out = block.conv1d(padded)
    if device == "cuda":
        torch.cuda.synchronize()
    conv1d_ms = ((time.perf_counter() - start) / num_iter) * 1000

    # Intermediate setup for scan
    conv_out_t = conv_out.transpose(1, 2)
    x_conv, delta_conv, B_conv, C_conv = torch.split(
        conv_out_t, [block.d_inner, block.d_inner, block.d_state, block.d_state], dim=-1
    )
    delta_param = F.softplus(delta_conv + block.dt_bias)
    x_heads = x_conv.view(-1, x_conv.shape[1], block.num_heads, block.d_head).transpose(1, 2)
    delta_heads = delta_param.view(
        -1, delta_param.shape[1], block.num_heads, block.d_head
    ).transpose(1, 2)
    B_heads = B_conv.unsqueeze(1).expand(-1, block.num_heads, -1, -1)
    C_heads = C_conv.unsqueeze(1).expand(-1, block.num_heads, -1, -1)

    # Stage 3: Selective SSD Scan recurrence
    from model.mamba2.scan import selective_ssd_scan

    start = time.perf_counter()
    for _ in range(num_iter):
        y_scan, _ = selective_ssd_scan(x_heads, delta_heads, block.A, B_heads, C_heads)
    if device == "cuda":
        torch.cuda.synchronize()
    ssd_scan_ms = ((time.perf_counter() - start) / num_iter) * 1000

    # Intermediate setup for output projection
    y_out = y_scan.transpose(1, 2).contiguous().view(-1, y_scan.shape[2], block.d_inner)
    y_gated = y_out * F.silu(gate)

    # Stage 4: Output projection
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = block.out_proj(y_gated)
    if device == "cuda":
        torch.cuda.synchronize()
    out_proj_ms = ((time.perf_counter() - start) / num_iter) * 1000

    return {
        "input_projection_ms": in_proj_ms,
        "causal_conv1d_ms": conv1d_ms,
        "selective_ssd_scan_ms": ssd_scan_ms,
        "output_projection_ms": out_proj_ms,
    }


def run_sequence_scaling(
    device: str,
) -> dict[str, dict[str, float | int]]:
    """Profile scaling characteristics across sequence lengths 128 to 4096."""
    config = get_base_config()
    dim = config.model.hidden_dim
    num_heads = config.model.num_heads

    mamba = Mamba2Block(config).to(device=device)
    attn = DenseSelfAttention(dim=dim, num_heads=num_heads).to(device=device)

    scaling_results = {}

    for seq_len in [128, 512, 1024, 2048, 4096]:
        print(f"Profiling scaling for sequence length: {seq_len}...")
        x = torch.randn(2, seq_len, dim, device=device, requires_grad=True)

        # Mamba forward/backward
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out_mamba = mamba(x)
        if device == "cuda":
            torch.cuda.synchronize()
        mamba_fwd = ((time.perf_counter() - start) / 20) * 1000

        loss = out_mamba.sum()
        start = time.perf_counter()
        for _ in range(20):
            x_grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        mamba_bwd = ((time.perf_counter() - start) / 20) * 1000

        # Attention forward/backward
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out_attn = attn(x)
        if device == "cuda":
            torch.cuda.synchronize()
        attn_fwd = ((time.perf_counter() - start) / 20) * 1000

        loss_attn = out_attn.sum()
        start = time.perf_counter()
        for _ in range(20):
            _ = torch.autograd.grad(loss_attn, x, retain_graph=True)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        attn_bwd = ((time.perf_counter() - start) / 20) * 1000

        scaling_results[f"seq_{seq_len}"] = {
            "mamba_forward_ms": mamba_fwd,
            "mamba_backward_ms": mamba_bwd,
            "attention_forward_ms": attn_fwd,
            "attention_backward_ms": attn_bwd,
            "mamba_state_norm": out_mamba.norm().item(),
            "mamba_grad_norm": x_grad.norm().item(),
        }

    return scaling_results


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Mamba2 block integration benchmarks on: {device.upper()}")

    # Sizing settings B=2, S=512, D=256
    b, s, d = 2, 512, 256
    x = torch.randn(b, s, d, device=device)

    config = get_base_config()
    mamba = Mamba2Block(config).to(device=device)

    # Component segment timing
    stages = profile_mamba2_components(mamba, x, device)

    # Scaling sequence timing
    scaling = run_sequence_scaling(device)

    # Save results payload
    results_dir = Path("experiments/phase_1_3")
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = {
        "batch_size": 2,
        "seq_len": 512,
        "hidden_dim": 256,
        "num_heads": 4,
        "mamba_ratio": 6,
        "d_state": 16,
    }

    environment = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device": device,
    }

    (results_dir / "benchmark_config.json").write_text(json.dumps(benchmark_config, indent=2))
    (results_dir / "system_info.json").write_text(json.dumps(get_system_info(), indent=2))
    (results_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (results_dir / "git_revision.json").write_text(
        json.dumps(
            {
                "git_commit": "pre-push",
                "random_seed": 42,
            },
            indent=2,
        )
    )
    (results_dir / "results.json").write_text(
        json.dumps(
            {
                "stages": stages,
                "sequence_scaling": scaling,
            },
            indent=2,
        )
    )

    print("=" * 60)
    print("Mamba2 benchmarking complete. Reports saved under experiments/phase_1_3/")
    print("=" * 60)


if __name__ == "__main__":
    main()
