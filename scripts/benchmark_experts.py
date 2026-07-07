# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Performance and quality benchmarking suite for MoE Experts container (Wave 2).

Compares sparse MoE execution against a dense FFN baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from configs.base_config import get_base_config
from model.moe.experts import MoEExperts
from model.swiglu import SwiGLUFFN
from scripts.benchmark_math_layers import get_system_info


def benchmark_experts(
    experts: MoEExperts,
    dense_ffn: SwiGLUFFN,
    x: torch.Tensor,
    device: str,
    num_warmup: int = 50,
    num_iter: int = 200,
) -> dict[str, float]:
    """Measure forward/backward latency and FLOP statistics."""
    b, s, d = x.shape
    weights = torch.ones(b, s, 2, device=device) * 0.5
    indices = torch.randint(0, experts.num_experts, (b, s, 2), device=device)

    # 1. Warmup
    for _ in range(num_warmup):
        out_moe, metrics = experts(x, weights, indices)
        out_moe.sum().backward(retain_graph=True)
        experts.zero_grad()

        out_dense = dense_ffn(x)
        out_dense.sum().backward(retain_graph=True)
        dense_ffn.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    # 2. Sparse MoE Profiling
    start_fwd_moe = time.perf_counter()
    for _ in range(num_iter):
        out_moe, metrics = experts(x, weights, indices)
    if device == "cuda":
        torch.cuda.synchronize()
    end_fwd_moe = time.perf_counter()
    fwd_moe = ((end_fwd_moe - start_fwd_moe) / num_iter) * 1000  # ms

    start_bwd_moe = time.perf_counter()
    for _ in range(num_iter):
        out_moe.sum().backward(retain_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()
    end_bwd_moe = time.perf_counter()
    bwd_moe = ((end_bwd_moe - start_bwd_moe) / num_iter) * 1000  # ms
    experts.zero_grad()

    # 3. Dense FFN Profiling
    start_fwd_dense = time.perf_counter()
    for _ in range(num_iter):
        out_dense = dense_ffn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end_fwd_dense = time.perf_counter()
    fwd_dense = ((end_fwd_dense - start_fwd_dense) / num_iter) * 1000  # ms

    start_bwd_dense = time.perf_counter()
    for _ in range(num_iter):
        out_dense.sum().backward(retain_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()
    end_bwd_dense = time.perf_counter()
    bwd_dense = ((end_bwd_dense - start_bwd_dense) / num_iter) * 1000  # ms
    dense_ffn.zero_grad()

    # Compute parameters count
    moe_params = sum(p.numel() for p in experts.parameters())
    dense_params = sum(p.numel() for p in dense_ffn.parameters())

    return {
        "moe_fwd_ms": fwd_moe,
        "moe_bwd_ms": bwd_moe,
        "dense_fwd_ms": fwd_dense,
        "dense_bwd_ms": bwd_dense,
        "moe_params": float(moe_params),
        "dense_params": float(dense_params),
        "flop_savings_pct": metrics["flop_savings_pct"],
        "capacity_value": metrics["capacity"],
        "dropped_tokens": metrics["dropped_tokens"],
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking MoE Experts vs Dense FFN on: {device.upper()}")

    # Nano Default shapes
    x = torch.randn(32, 512, 256, device=device)

    config = get_base_config()
    experts = MoEExperts(config).to(device=device)

    # Dense Baseline FFN
    dense_ffn = SwiGLUFFN(dim=256, bias=False).to(device=device)

    metrics = benchmark_experts(experts, dense_ffn, x, device)

    print("=" * 60)
    print(f"MoE Forward Latency  : {metrics['moe_fwd_ms']:.4f} ms")
    print(f"Dense Forward Latency: {metrics['dense_fwd_ms']:.4f} ms")
    print(f"MoE Backward Latency : {metrics['moe_bwd_ms']:.4f} ms")
    print(f"Dense Backward Latency: {metrics['dense_bwd_ms']:.4f} ms")
    print(f"FLOP Savings         : {metrics['flop_savings_pct']:.2f}%")
    print(f"MoE Parameter Count  : {metrics['moe_params']:,}")
    print(f"Dense Parameter Count: {metrics['dense_params']:,}")
    print("=" * 60)

    # Save to temp experiments output
    out_dir = Path("experiments/phase_1_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    system_info = get_system_info()
    (out_dir / "system_info_experts.json").write_text(json.dumps(system_info, indent=2))
    (out_dir / "results_experts.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
