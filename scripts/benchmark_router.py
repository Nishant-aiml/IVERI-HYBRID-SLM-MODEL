# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Performance and quality benchmarking suite for MoE Gating Router (Wave 1).

Measures routing latency, throughput, gating entropy, and utilization variance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from configs.base_config import get_base_config
from model.moe.router import SparseMoERouter
from scripts.benchmark_math_layers import get_system_info


def benchmark_router(
    router: SparseMoERouter,
    x: torch.Tensor,
    device: str,
    num_warmup: int = 100,
    num_iter: int = 1000,
) -> dict[str, float | list[float]]:
    """Measure forward/backward latency and quality metrics for the router."""
    # Warmup
    for _ in range(num_warmup):
        weights, indices, loss = router(x)
        loss.backward(retain_graph=True)
        router.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    # Forward Latency
    start_fwd = time.perf_counter()
    for _ in range(num_iter):
        weights, indices, loss = router(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end_fwd = time.perf_counter()
    fwd_latency = ((end_fwd - start_fwd) / num_iter) * 1000  # ms

    # Backward Latency
    weights, indices, loss = router(x)
    start_bwd = time.perf_counter()
    for _ in range(num_iter):
        loss.backward(retain_graph=True)
    if device == "cuda":
        torch.cuda.synchronize()
    end_bwd = time.perf_counter()
    bwd_latency = ((end_bwd - start_bwd) / num_iter) * 1000  # ms
    router.zero_grad()

    # Retrieve quality metrics
    # Softmax probabilities over all experts
    # Let's extract router characteristics
    n, d = x.view(-1, x.shape[-1]).shape
    logits = router.wg(x.view(-1, d))
    probs = torch.softmax(logits, dim=-1)

    # 1. Routing Entropy: H(G) = -sum(p_i * log(p_i))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

    # 2. Expert Utilization Variance
    mask = torch.zeros_like(logits)
    _, topk_indices = torch.topk(logits, router.num_active_experts, dim=-1)
    mask.scatter_(dim=-1, index=topk_indices, value=1.0)
    utilization = mask.mean(dim=0)  # Shape (E,)
    variance = utilization.var().item()

    return {
        "forward_latency_ms": fwd_latency,
        "backward_latency_ms": bwd_latency,
        "mean_routing_entropy": entropy,
        "expert_utilization_variance": variance,
        "expert_utilization_histogram": utilization.tolist(),
        "aux_loss_value": loss.item(),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking MoE Router on: {device.upper()}")

    # Nano Default Shape
    shape = (32, 512, 256)
    x = torch.randn(shape, device=device, requires_grad=True)

    config = get_base_config()
    router = SparseMoERouter(config).to(device=device)

    metrics = benchmark_router(router, x, device)

    print("=" * 60)
    print(f"Forward Latency : {metrics['forward_latency_ms']:.4f} ms")
    print(f"Backward Latency: {metrics['backward_latency_ms']:.4f} ms")
    print(f"Mean Entropy    : {metrics['mean_routing_entropy']:.4f}")
    print(f"Util Variance   : {metrics['expert_utilization_variance']:.4f}")
    print(f"Util Histogram  : {metrics['expert_utilization_histogram']}")
    print("=" * 60)

    # Save to temp experiments output
    out_dir = Path("experiments/phase_1_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write configs
    system_info = get_system_info()
    (out_dir / "system_info_router.json").write_text(json.dumps(system_info, indent=2))
    (out_dir / "results_router.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
