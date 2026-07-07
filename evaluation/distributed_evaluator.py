# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed evaluation wrapper for IVERI CORE.

:class:`DistributedEvaluator` wraps the frozen :class:`~evaluation.evaluator.Evaluator`
and extends it for multi-rank operation.  The frozen Evaluator is never
modified.

Responsibilities
----------------
- **Distributed perplexity**: each rank evaluates its shard of the
  validation set; NLL sums and token counts are all-reduced across ranks
  to compute the globally correct perplexity.
- **Distributed generation**: prompts are distributed across ranks; outputs
  are gathered on rank 0.
- **Distributed benchmark**: each rank benchmarks independently; latency
  and throughput statistics are reduced on rank 0.
- **Distributed telemetry reduction**: architecture telemetry dicts from
  all ranks are gathered and averaged on rank 0.

The existing :class:`~evaluation.evaluator.Evaluator` remains frozen;
:class:`DistributedEvaluator` adds only the rank-coordination layer.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from evaluation.evaluator import Evaluator
from training.distributed import DistributedManager


class DistributedEvaluator:
    """Distributed wrapper around the frozen :class:`~evaluation.evaluator.Evaluator`.

    Parameters
    ----------
    evaluator:
        A fully constructed :class:`~evaluation.evaluator.Evaluator`
        instance.  Its model should be the distributed-wrapped model.
    dist_manager:
        :class:`~training.distributed.DistributedManager` instance.
    config:
        Full project configuration.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        dist_manager: DistributedManager,
        config: IVERIConfig,
    ) -> None:
        self.evaluator = evaluator
        self.dist_manager = dist_manager
        self.config = config

    # ── Distributed perplexity ─────────────────────────────────────────

    @torch.no_grad()
    def evaluate_perplexity(self, dataloader: DataLoader[Any]) -> dict[str, float]:
        """Compute globally correct perplexity across all ranks.

        Each rank iterates its shard of *dataloader* (assumed to be backed
        by a :class:`~torch.utils.data.distributed.DistributedSampler`).
        NLL sum and token count are all-reduced; rank 0 computes the final
        perplexity.

        Parameters
        ----------
        dataloader:
            Validation dataloader (one shard per rank).

        Returns
        -------
        dict[str, float]
            Keys: ``loss``, ``perplexity``, ``num_tokens``, ``num_batches``.
        """
        model = self.evaluator.model
        model.eval()
        device = self.evaluator.device
        precision_handler = self.evaluator.precision_handler

        total_nll = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_tokens = torch.tensor(0, dtype=torch.int64, device=device)
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            elif isinstance(batch, dict):
                inputs = batch["input_ids"]
                targets = batch["labels"]
            else:
                inputs = batch
                targets = batch

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with precision_handler.autocast_context():
                outputs = model(inputs, return_dict=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            # Compute per-token NLL (sum, not mean, for correct global average)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            nll = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="sum")
            total_nll += nll.double()
            total_tokens += flat_targets.numel()
            num_batches += 1

        # All-reduce NLL and token count across all ranks
        self.dist_manager.all_reduce_mean(total_nll)
        total_nll *= self.dist_manager.world_size()  # sum → undo mean
        self.dist_manager.all_reduce_mean(total_tokens.float())
        total_tokens_float = total_tokens.float()
        total_tokens_float *= self.dist_manager.world_size()

        # Compute final metrics on all ranks (same values after all-reduce)
        avg_nll = (total_nll / total_tokens_float).item() if total_tokens_float.item() > 0 else 0.0

        try:
            perplexity = math.exp(avg_nll)
            if math.isnan(perplexity) or math.isinf(perplexity):
                perplexity = 0.0
        except OverflowError:
            perplexity = 0.0

        return {
            "loss": float(avg_nll),
            "perplexity": float(perplexity),
            "num_tokens": int(total_tokens_float.item()),
            "num_batches": num_batches,
        }

    # ── Distributed generation ─────────────────────────────────────────

    @torch.no_grad()
    def evaluate_generation(
        self,
        prompts: list[bytes],
        max_new_bytes: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[dict[str, Any]]:
        """Distribute generation across ranks; gather results on rank 0.

        Prompts are partitioned across ranks.  Each rank generates for its
        slice.  All outputs are gathered on rank 0 via
        :meth:`~training.distributed.DistributedManager.all_gather_object`.

        Parameters
        ----------
        prompts:
            List of byte-string prompts.  Length need not be divisible by
            ``world_size`` — remaining prompts are handled by rank 0.
        max_new_bytes, temperature, top_k, top_p:
            Generation parameters (fall back to config defaults if ``None``).

        Returns
        -------
        list[dict[str, Any]]
            One result dict per prompt.  Available on all ranks but only
            complete on rank 0 (other ranks see their own slice).
        """
        eval_cfg = self.config.evaluation
        max_new_bytes = max_new_bytes or eval_cfg.generation_max_new_bytes
        temperature = temperature if temperature is not None else eval_cfg.generation_temperature
        top_k = top_k if top_k is not None else eval_cfg.generation_top_k
        top_p = top_p if top_p is not None else eval_cfg.generation_top_p

        # Partition prompts across ranks
        rank = self.dist_manager.rank()
        world_size = self.dist_manager.world_size()
        local_prompts = [p for i, p in enumerate(prompts) if i % world_size == rank]

        local_results: list[dict[str, Any]] = []
        for prompt in local_prompts:
            input_ids = torch.tensor(
                list(prompt), dtype=torch.long, device=self.evaluator.device
            ).unsqueeze(0)
            gen_res = self.evaluator.generation_evaluator.generate(
                model=self.evaluator.model,
                input_ids=input_ids,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.evaluator.device,
            )
            local_results.append(
                {
                    "latency_seconds": gen_res.latency_seconds,
                    "bytes_per_second": gen_res.bytes_per_second,
                    "avg_generated_length": gen_res.avg_generated_length,
                    "early_stopped_ratio": gen_res.early_stopped_ratio,
                }
            )

        # Gather across ranks
        all_results = self.dist_manager.all_gather_object(local_results)
        merged: list[dict[str, Any]] = []
        for rank_results in all_results:
            merged.extend(rank_results)
        return merged

    # ── Distributed benchmark ──────────────────────────────────────────

    def evaluate_benchmark(
        self,
        input_ids: torch.Tensor | None = None,
        iterations: int | None = None,
        warmup_iterations: int | None = None,
    ) -> dict[str, float]:
        """Run inference benchmark on each rank; reduce stats on rank 0.

        Each rank benchmarks with the same inputs independently.  The
        returned stats are the mean across ranks (same values when the
        hardware is homogeneous; the mean is the correct aggregate metric
        for heterogeneous setups).

        Parameters
        ----------
        input_ids:
            Optional input tensor.  Falls back to a random tensor.
        iterations, warmup_iterations:
            Override config defaults.

        Returns
        -------
        dict[str, float]
            Latency and throughput stats, all-reduced across ranks.
        """
        eval_cfg = self.config.evaluation
        if input_ids is None:
            seq_len = self.config.training.seq_len
            batch_size = eval_cfg.batch_size
            input_ids = torch.randint(
                0, 256, (batch_size, seq_len), dtype=torch.long, device=self.evaluator.device
            )
        iterations = iterations or eval_cfg.benchmark_iterations
        warmup_iterations = warmup_iterations or eval_cfg.warmup_iterations

        bench_res = self.evaluator.inference_benchmark.run(
            input_ids=input_ids,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            device=self.evaluator.device,
        )

        local_stats: dict[str, float] = {
            "warmup_latency_ms": bench_res.warmup_latency_ms,
            "latency_mean_ms": bench_res.latency_mean_ms,
            "latency_median_ms": bench_res.latency_median_ms,
            "latency_p50_ms": bench_res.latency_p50_ms,
            "latency_p90_ms": bench_res.latency_p90_ms,
            "latency_p95_ms": bench_res.latency_p95_ms,
            "latency_p99_ms": bench_res.latency_p99_ms,
            "latency_min_ms": bench_res.latency_min_ms,
            "latency_max_ms": bench_res.latency_max_ms,
            "samples_per_sec": bench_res.samples_per_sec,
            "tokens_per_sec": bench_res.tokens_per_sec,
            "bytes_per_sec": bench_res.bytes_per_sec,
            "estimated_flops": bench_res.estimated_flops,
            "parameter_count": float(bench_res.parameter_count),
        }

        return self.dist_manager.reduce_dict(local_stats)

    # ── Distributed telemetry ──────────────────────────────────────────

    def evaluate_telemetry(self, telemetry_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Gather and average architecture telemetry dicts across ranks.

        Each rank may have collected different telemetry batches during
        its forward passes.  This method gathers all telemetry from all
        ranks, concatenates, and returns the averaged result on rank 0.

        Parameters
        ----------
        telemetry_list:
            List of telemetry dicts from this rank's forward passes.

        Returns
        -------
        dict[str, Any]
            Averaged architecture telemetry, available on all ranks.
        """
        all_telemetry_lists = self.dist_manager.all_gather_object(telemetry_list)
        merged_telemetry: list[dict[str, Any]] = []
        for rank_list in all_telemetry_lists:
            merged_telemetry.extend(rank_list)

        if not merged_telemetry:
            return {}

        return self.evaluator.architecture_evaluator.evaluate(
            telemetry_list=merged_telemetry,
            model=self.evaluator.model,
        )

    # ── Full distributed evaluation pass ──────────────────────────────

    def run_all(
        self,
        dataloader: DataLoader[Any] | None = None,
    ) -> dict[str, Any]:
        """Run all distributed evaluation components and aggregate results.

        Parameters
        ----------
        dataloader:
            Optional distributed validation dataloader.

        Returns
        -------
        dict[str, Any]
            Aggregated results dict.  Perplexity, benchmark, and telemetry
            are globally correct across all ranks.
        """
        t_start = time.perf_counter()
        results: dict[str, Any] = {}

        if dataloader is not None:
            results["perplexity"] = self.evaluate_perplexity(dataloader)

        results["benchmark"] = self.evaluate_benchmark()
        results["evaluation_duration_seconds"] = time.perf_counter() - t_start
        results["rank"] = self.dist_manager.rank()
        results["world_size"] = self.dist_manager.world_size()

        return results
