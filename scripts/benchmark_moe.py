# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Performance profiling and baseline comparisons for MoE (Wave 3).

Benchmarks the full MoE layer, compares it with a Dense FFN, segments latencies,
and profiles scaling across 2, 4, and 8 experts. Saves all results to JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from configs.base_config import get_base_config
from model.moe import MoEExperts, SparseMoERouter
from model.swiglu import SwiGLUFFN
from scripts.benchmark_math_layers import get_system_info


def profile_multistage_latencies(
    router: SparseMoERouter,
    experts: MoEExperts,
    x: torch.Tensor,
    device: str,
    num_iter: int = 100,
) -> dict[str, float]:
    """Segment routing, dispatch, execution, and recombination latencies."""
    if device == "cuda":
        torch.cuda.synchronize()

    # Segment 1: Routing gate calculation
    start = time.perf_counter()
    for _ in range(num_iter):
        weights, indices, loss = router(x)
    if device == "cuda":
        torch.cuda.synchronize()
    routing_time = ((time.perf_counter() - start) / num_iter) * 1000  # ms

    # Pre-run router to profile experts stages
    weights, indices, loss = router(x)
    b, s, d = x.shape
    x_flat = x.view(-1, d)
    weights_flat = weights.view(-1, experts.num_active_experts)
    indices_flat = indices.view(-1, experts.num_active_experts)
    num_tokens = x_flat.shape[0]

    # Segment 2: Dispatch and gather index mapping overhead
    start = time.perf_counter()
    for _ in range(num_iter):
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(-1)
            .expand(-1, experts.num_active_experts)
        )
        rank_indices = (
            torch.arange(experts.num_active_experts, device=x.device)
            .unsqueeze(0)
            .expand(num_tokens, -1)
        )
        for e in range(experts.num_experts):
            mask = indices_flat == e
            tokens_e = token_indices[mask]
            ranks_e = rank_indices[mask]
    if device == "cuda":
        torch.cuda.synchronize()
    dispatch_time = ((time.perf_counter() - start) / num_iter) * 1000  # ms

    # Pre-dispatch for expert computation profiling
    token_indices = (
        torch.arange(num_tokens, device=x.device)
        .unsqueeze(-1)
        .expand(-1, experts.num_active_experts)
    )
    rank_indices = (
        torch.arange(experts.num_active_experts, device=x.device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
    )

    # Segment 3: FFN expert execution only
    start = time.perf_counter()
    for _ in range(num_iter):
        for e in range(experts.num_experts):
            mask = indices_flat == e
            tokens_e = token_indices[mask]
            if tokens_e.shape[0] > 0:
                input_e = x_flat[tokens_e]
                _ = experts.experts[e](input_e)
    if device == "cuda":
        torch.cuda.synchronize()
    expert_exec_time = ((time.perf_counter() - start) / num_iter) * 1000  # ms

    # Segment 4: Recombination (index accumulation)
    output_flat = torch.zeros_like(x_flat)
    outputs_e = []
    tokens_list = []
    for e in range(experts.num_experts):
        mask = indices_flat == e
        tokens_e = token_indices[mask]
        ranks_e = rank_indices[mask]
        if tokens_e.shape[0] > 0:
            input_e = x_flat[tokens_e]
            out_e = experts.experts[e](input_e)
            w_e = weights_flat[tokens_e, ranks_e].unsqueeze(-1)
            outputs_e.append(out_e * w_e)
            tokens_list.append(tokens_e)

    start = time.perf_counter()
    for _ in range(num_iter):
        for o_e, t_e in zip(outputs_e, tokens_list, strict=False):
            output_flat.index_add_(0, t_e, o_e)
    if device == "cuda":
        torch.cuda.synchronize()
    recombination_time = ((time.perf_counter() - start) / num_iter) * 1000  # ms

    return {
        "routing_gate_ms": routing_time,
        "dispatch_gather_ms": dispatch_time,
        "expert_ffn_exec_ms": expert_exec_time,
        "recombination_ms": recombination_time,
    }


def run_scaling_benchmark(
    config_base: object,
    x: torch.Tensor,
    device: str,
) -> dict[str, dict[str, float | list[float]]]:
    """Run MoE layer scaling benchmark over 2, 4, and 8 experts configurations."""
    scaling_results = {}

    for num_e in [2, 4, 8]:
        print(f"Profiling configuration: Experts = {num_e}...")
        config = get_base_config()
        config.model.num_experts = num_e
        # keep Top-K = 2 if E >= 2, else 1
        config.model.num_active_experts = min(2, num_e)

        router = SparseMoERouter(config).to(device=device)
        experts = MoEExperts(config).to(device=device)

        # Profile Forward
        weights, indices, loss = router(x)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            w, idx, loss_val = router(x)
            out, metrics = experts(x, w, idx)
        if device == "cuda":
            torch.cuda.synchronize()
        latency = ((time.perf_counter() - start) / 50) * 1000  # ms

        # Loss calculation
        total_loss = loss_val + out.sum()
        total_loss.backward()

        # Get gradient statistics per expert
        grad_norms = []
        for e in range(num_e):
            grad_sum = 0.0
            for p in experts.experts[e].parameters():
                if p.grad is not None:
                    grad_sum += p.grad.norm().item()
            grad_norms.append(grad_sum)

        scaling_results[f"{num_e}_experts"] = {
            "latency_ms": latency,
            "aux_loss": loss_val.item(),
            "grad_norms": grad_norms,
        }

    return scaling_results


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Wave 3 MoE Integration Benchmark on: {device.upper()}")

    # Default 10M configuration inputs
    shape = (32, 512, 256)
    x = torch.randn(shape, device=device, requires_grad=True)

    config = get_base_config()
    router = SparseMoERouter(config).to(device=device)
    experts = MoEExperts(config).to(device=device)

    # Dense baseline
    dense_ffn = SwiGLUFFN(dim=256, bias=False).to(device=device)

    # Multi-stage latencies
    stages = profile_multistage_latencies(router, experts, x, device)

    # Dense baseline comparison
    # MoE end-to-end forward run
    start_moe = time.perf_counter()
    for _ in range(100):
        w, idx, loss_val = router(x)
        out_moe, metrics = experts(x, w, idx)
    if device == "cuda":
        torch.cuda.synchronize()
    moe_fwd_ms = ((time.perf_counter() - start_moe) / 100) * 1000

    start_dense = time.perf_counter()
    for _ in range(100):
        _ = dense_ffn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dense_fwd_ms = ((time.perf_counter() - start_dense) / 100) * 1000

    # Gradient norms extraction
    loss = out_moe.sum() + loss_val
    loss.backward()

    grad_norms = {}
    for e in range(experts.num_experts):
        grad_sum = 0.0
        for _name, p in experts.experts[e].named_parameters():
            if p.grad is not None:
                grad_sum += p.grad.norm().item()
        grad_norms[f"expert_{e}"] = grad_sum

    # Router statistics
    raw_logits = router.wg(x.view(-1, 256))
    probs = torch.softmax(raw_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

    mask = torch.zeros_like(raw_logits)
    _, topk_indices = torch.topk(raw_logits, 2, dim=-1)
    mask.scatter_(dim=-1, index=topk_indices, value=1.0)
    utilization = mask.mean(dim=0).tolist()

    # Scaling results
    scaling_metrics = run_scaling_benchmark(config, x, device)

    # Compile result payloads
    results_dir = Path("experiments/phase_1_2")
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = {
        "batch_size": 32,
        "seq_len": 512,
        "hidden_dim": 256,
        "num_experts": 4,
        "num_active_experts": 2,
        "capacity_factor": 1.25,
    }

    # Permanent Artifacts creation
    (results_dir / "benchmark_config.json").write_text(json.dumps(benchmark_config, indent=2))
    (results_dir / "system_info.json").write_text(json.dumps(get_system_info(), indent=2))
    (results_dir / "results.json").write_text(
        json.dumps(
            {
                "moe_forward_ms": moe_fwd_ms,
                "dense_forward_ms": dense_fwd_ms,
                "stages": stages,
                "scaling_experts": scaling_metrics,
            },
            indent=2,
        )
    )
    (results_dir / "routing_histogram.json").write_text(
        json.dumps(
            {
                "utilization_histogram": utilization,
            },
            indent=2,
        )
    )
    (results_dir / "gradient_statistics.json").write_text(json.dumps(grad_norms, indent=2))
    (results_dir / "expert_utilization.json").write_text(
        json.dumps(
            {
                "imbalance_variance": torch.tensor(utilization).var().item(),
                "routing_entropy": entropy,
                "dropped_tokens": metrics["dropped_tokens"],
            },
            indent=2,
        )
    )

    print("=" * 60)
    print("Integration Benchmarking Complete. Saved JSON reports under experiments/phase_1_2")
    print("=" * 60)


if __name__ == "__main__":
    main()
