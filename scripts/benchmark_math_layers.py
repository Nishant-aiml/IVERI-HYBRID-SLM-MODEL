# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarking suite for core mathematical layers (Phase 1.1).

Measures forward/backward latency, memory allocation, parameter count,
element-wise throughput, and numerical stability metrics.
Saves results, config, and system info inside experiments/phase_1_1/.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from model.norms import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_emb
from model.swiglu import SwiGLUFFN


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash or return 'N/A'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "N/A"


def get_system_info() -> dict[str, str | int | float]:
    """Gather hardware and environment metadata."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else "N/A",
        "cpu_architecture": platform.machine(),
        "gpu_name": gpu_name,
        "git_commit_hash": get_git_commit_hash(),
    }


def benchmark_layer(
    name: str,
    module: torch.nn.Module,
    x: torch.Tensor,
    device: str,
    num_warmup: int = 100,
    num_iter: int = 1000,
) -> dict[str, float | list[float]]:
    """Run benchmarking and collect latency, memory, throughput, and error metrics."""
    param_count = sum(p.numel() for p in module.parameters())

    # Warmup
    for _ in range(num_warmup):
        out = module(x)
        loss = out[0].sum() if isinstance(out, tuple) else out.sum()
        loss.backward(retain_graph=True)
        module.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Forward Pass Latency
    start_fwd = time.perf_counter()
    for _ in range(num_iter):
        out = module(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end_fwd = time.perf_counter()

    fwd_latency = ((end_fwd - start_fwd) / num_iter) * 1000  # ms

    # Peak Memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    # Backward Pass Latency
    out = module(x)
    loss = out[0].sum() if isinstance(out, tuple) else out.sum()

    start_bwd = time.perf_counter()
    for _ in range(num_iter):
        loss.backward(retain_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()
    end_bwd = time.perf_counter()

    bwd_latency = ((end_bwd - start_bwd) / num_iter) * 1000  # ms

    # Retrieve gradients to run numeric checking
    grads = []
    nan_count = 0
    inf_count = 0
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad.norm().item())
            nan_count += int(torch.isnan(p.grad).sum().item())
            inf_count += int(torch.isinf(p.grad).sum().item())

    module.zero_grad()

    # Elements processed per second (Throughput)
    num_elements = x.numel()
    throughput = num_elements / (fwd_latency / 1000)  # elements/sec

    # Numerical validation of output stability
    out_flat = out[0] if isinstance(out, tuple) else out
    nan_out = int(torch.isnan(out_flat).sum().item())
    inf_out = int(torch.isinf(out_flat).sum().item())

    return {
        "forward_latency_ms": fwd_latency,
        "backward_latency_ms": bwd_latency,
        "peak_memory_mb": peak_memory,
        "parameter_count": float(param_count),
        "throughput_elements_per_sec": throughput,
        "gradients_norm_distribution": grads,
        "nan_count_gradients": float(nan_count),
        "inf_count_gradients": float(inf_count),
        "nan_count_outputs": float(nan_out),
        "inf_count_outputs": float(inf_out),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking Math Layers on: {device.upper()}")

    # 32 batch size, 512 sequence length, 256 hidden dim (Nano defaults)
    shape = (32, 512, 256)
    x = torch.randn(shape, device=device, requires_grad=True)

    # Configuration metadata
    benchmark_config = {
        "batch_size": shape[0],
        "seq_len": shape[1],
        "hidden_dim": shape[2],
        "num_warmup": 100,
        "num_iter": 1000,
        "device": device,
    }

    # Instantiate modules
    rmsnorm = RMSNorm(dim=256).to(device=device)
    swiglu = SwiGLUFFN(dim=256).to(device=device)
    rope = RotaryEmbedding(dim=64).to(device=device)

    class RoPEWrapper(torch.nn.Module):
        def __init__(self, rope_mod):
            super().__init__()
            self.rope_mod = rope_mod

        def forward(self, tensor):
            x_rope = tensor.view(32, 512, 4, 64).transpose(1, 2)  # (32, 4, 512, 64)
            cos, sin = self.rope_mod(x_rope)
            return apply_rotary_emb(
                x_rope, cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)
            )

    rope_wrapped = RoPEWrapper(rope)

    results = {}
    print("Running RMSNorm benchmarks...")
    results["rmsnorm"] = benchmark_layer("rmsnorm", rmsnorm, x, device)

    print("Running SwiGLU benchmarks...")
    results["swiglu"] = benchmark_layer("swiglu", swiglu, x, device)

    print("Running RoPE benchmarks...")
    results["rope"] = benchmark_layer("rope", rope_wrapped, x, device)

    # Gather metadata
    system_info = get_system_info()

    # Print comparison
    print("=" * 60)
    print(
        f"{'Layer':<15} | {'Params':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'VRAM (MB)':<10}"
    )
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<15} | {res['parameter_count']:<10.0f} | {res['forward_latency_ms']:<10.4f} | {res['backward_latency_ms']:<10.4f} | {res['peak_memory_mb']:<10.2f}"
        )
    print("=" * 60)

    # Save output folder structure
    out_dir = Path("experiments/phase_1_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "system_info.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")
    (out_dir / "benchmark_config.json").write_text(
        json.dumps(benchmark_config, indent=2), encoding="utf-8"
    )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Backward compatibility with single json file request
    compat_path = Path("experiments/phase_1_1_benchmarks.json")
    compat_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Benchmark results and environment metadata saved inside: {out_dir}")


if __name__ == "__main__":
    main()
