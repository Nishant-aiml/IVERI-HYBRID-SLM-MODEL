# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Central Evaluator orchestrator for IVERI CORE (Phase 2.5).

Coordinates perplexity evaluation, text generation tests, performance benchmarks,
resource memory tracking, and architecture telemetry analysis into reproducible reports.
"""

from __future__ import annotations

import math
import pathlib
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from core.constants import ARCHITECTURE_VERSION
from evaluation.arch_eval import ArchitectureEvaluator
from evaluation.benchmark import InferenceBenchmark
from evaluation.checkpoint_compare import CheckpointComparator
from evaluation.generation import GenerationEvaluator
from evaluation.memory_tracker import MemoryTracker
from evaluation.perplexity import PerplexityEvaluator
from evaluation.report_generator import ReportGenerator
from training.logger import _git_info, _system_info
from training.mixed_precision import PrecisionHandler


class Evaluator:
    """Orchestrates the read-only evaluation pipeline and benchmark suite."""

    def __init__(
        self,
        model: nn.Module,
        config: IVERIConfig,
        val_dataloader: DataLoader | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            model: IVERI model instance.
            config: General configuration object.
            val_dataloader: Optional dataloader for perplexity evaluation.
            device: Accelerator device to place tensors on.
        """
        self.model = model
        self.config = config
        self.val_dataloader = val_dataloader

        # Determine device
        if device is None:
            self.device = torch.device(config.hardware.device)
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Initialize precision handler
        self.precision_handler = PrecisionHandler(
            precision=config.hardware.mixed_precision,
            device=str(self.device),
        )

        # Initialize sub-evaluators
        self.perplexity_evaluator = PerplexityEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.inference_benchmark = InferenceBenchmark(self.model)
        self.architecture_evaluator = ArchitectureEvaluator()
        self.report_generator = ReportGenerator(output_dir=config.evaluation.report_dir)
        self.comparator = CheckpointComparator()

    def evaluate_checkpoint(self, checkpoint_path: str | pathlib.Path) -> dict[str, Any]:
        """Load weights from a checkpoint and execute the evaluation pipeline.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Dictionary containing evaluation metrics and metadata.
        """
        from training.checkpointing import load_checkpoint

        _ = load_checkpoint(checkpoint_path, self.model)
        return self.evaluate()

    def evaluate(self) -> dict[str, Any]:
        """Run the full read-only evaluation pipeline.

        Returns:
            Aggregated dictionary of evaluation metrics and metadata.
        """
        self.model.eval()

        t_start = time.perf_counter()

        # 1. Fallback inputs for benchmarking & generation if dataloader is empty/absent
        batch_size = self.config.evaluation.batch_size
        seq_len = 512  # default sequence length

        bench_input = None
        if self.val_dataloader is not None:
            # Try to grab a batch from val_dataloader
            try:
                iterator = iter(self.val_dataloader)
                first_batch = next(iterator)
                if isinstance(first_batch, (list, tuple)):
                    bench_input, _ = first_batch
                elif isinstance(first_batch, dict):
                    bench_input = first_batch["input_ids"]
                else:
                    bench_input = first_batch
            except StopIteration:
                pass

        if bench_input is None:
            # Generate dummy raw bytes indices as fallback
            bench_input = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long, device=self.device)
        else:
            bench_input = bench_input.to(self.device)

        # ── 2. Run perplexity dataset evaluation and collect telemetry ────
        perplexity_stats = {}
        telemetry_list: list[dict[str, Any]] = []

        if self.config.evaluation.enabled and self.val_dataloader is not None:
            total_loss = 0.0
            total_tokens = 0
            num_batches = 0
            max_batches = self.config.evaluation.max_eval_batches

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break

                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    elif isinstance(batch, dict):
                        inputs = batch["input_ids"]
                        targets = batch["labels"]
                    else:
                        inputs = batch
                        targets = batch

                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    with self.precision_handler.autocast_context():
                        outputs = self.model(inputs, return_dict=True)
                        if isinstance(outputs, dict):
                            logits = outputs["logits"]
                            # Capture batch telemetry if present
                            tel = outputs.get("telemetry")
                            if tel:
                                telemetry_list.append(tel)
                        else:
                            logits = outputs

                        batch_metrics = self.perplexity_evaluator.evaluate_batch(logits, targets)

                    total_loss += batch_metrics["loss"]
                    total_tokens += batch_metrics["num_tokens"]
                    num_batches += 1

            if total_tokens > 0:
                final_loss = total_loss / total_tokens
                try:
                    perplexity = math.exp(final_loss)
                    if math.isnan(perplexity) or math.isinf(perplexity):
                        perplexity = 0.0
                except OverflowError:
                    perplexity = 0.0

                if math.isnan(final_loss) or math.isinf(final_loss):
                    final_loss = 0.0

                perplexity_stats = {
                    "loss": float(final_loss),
                    "perplexity": float(perplexity),
                    "num_tokens": int(total_tokens),
                    "num_batches": int(num_batches),
                }

        # ── 3. Run generation decoding benchmark ─────────────────────────
        generation_stats = {}
        if self.config.evaluation.generation_enabled:
            # Use prompt slice from the fallback input
            prompt = bench_input[: min(4, bench_input.size(0)), :8]
            gen_res = self.generation_evaluator.generate(
                model=self.model,
                input_ids=prompt,
                max_new_bytes=self.config.evaluation.generation_max_new_bytes,
                temperature=self.config.evaluation.generation_temperature,
                top_k=self.config.evaluation.generation_top_k,
                top_p=self.config.evaluation.generation_top_p,
                device=self.device,
            )
            generation_stats = {
                "latency_seconds": gen_res.latency_seconds,
                "bytes_per_second": gen_res.bytes_per_second,
                "avg_generated_length": gen_res.avg_generated_length,
                "early_stopped_ratio": gen_res.early_stopped_ratio,
            }

        # ── 4. Run throughput inference benchmark ───────────────────────
        benchmark_stats = {}
        if self.config.evaluation.throughput_tracking:
            bench_res = self.inference_benchmark.run(
                input_ids=bench_input,
                iterations=self.config.evaluation.benchmark_iterations,
                warmup_iterations=self.config.evaluation.warmup_iterations,
                device=self.device,
            )
            benchmark_stats = {
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
                "patches_per_sec": bench_res.patches_per_sec,
                "docs_per_sec": bench_res.docs_per_sec,
                "cpu_utilization_pct": bench_res.cpu_utilization_pct,
                "gpu_utilization_pct": bench_res.gpu_utilization_pct,
                "vram_used_mb": bench_res.vram_used_mb,
                "ram_used_mb": bench_res.ram_used_mb,
                "estimated_flops": bench_res.estimated_flops,
                "parameter_count": bench_res.parameter_count,
            }

        # ── 5. Run Memory tracking ───────────────────────────────────────
        memory_stats = {}
        if self.config.evaluation.memory_tracking:
            with MemoryTracker(self.model) as tracker:
                # Perform 3 fast forward dummy passes to record peak allocation
                for _ in range(3):
                    with self.precision_handler.autocast_context():
                        _ = self.model(bench_input, return_dict=True)
            mem_snapshot = tracker.get_snapshot()
            memory_stats = {
                "gpu_allocated_mb": mem_snapshot.gpu_allocated_mb,
                "gpu_reserved_mb": mem_snapshot.gpu_reserved_mb,
                "gpu_peak_mb": mem_snapshot.gpu_peak_mb,
                "cpu_ram_mb": mem_snapshot.cpu_ram_mb,
                "cpu_peak_ram_mb": mem_snapshot.cpu_peak_ram_mb,
                "parameter_mb": mem_snapshot.parameter_mb,
                "activation_mb": mem_snapshot.activation_mb,
                "checkpoint_mb": mem_snapshot.checkpoint_mb,
                "fragmentation_ratio": mem_snapshot.fragmentation_ratio,
                "growth_mb": mem_snapshot.growth_mb,
            }

        # ── 6. Aggregate architecture telemetry ─────────────────────────
        architecture_stats = {}
        if self.config.evaluation.architecture_tracking:
            # If no forward pass telemetry was gathered (e.g. dataloader was None),
            # trigger a dummy forward pass to extract single-batch telemetry
            if not telemetry_list:
                with torch.no_grad(), self.precision_handler.autocast_context():
                    outputs = self.model(bench_input, return_dict=True)
                    if isinstance(outputs, dict) and "telemetry" in outputs:
                        tel = outputs["telemetry"]
                        if tel:
                            telemetry_list.append(tel)

            architecture_stats = self.architecture_evaluator.evaluate(
                telemetry_list=telemetry_list,
                model=self.model,
            )

        # ── 7. Build metadata and aggregate report output ────────────────
        duration = time.perf_counter() - t_start
        git_info = _git_info()
        sys_info = _system_info()

        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_info.get("git_commit", ""),
            "git_branch": git_info.get("git_branch", ""),
            "architecture_version": ARCHITECTURE_VERSION,
            "random_seed": self.config.training.get("seed", 42) if hasattr(self.config.training, "get") else 42,
            "device": str(self.device),
            "dtype": str(self.precision_handler.dtype).replace("torch.", ""),
            "pytorch_version": sys_info.get("pytorch_version", ""),
            "cuda_version": sys_info.get("cuda_version", ""),
            "evaluation_duration_seconds": duration,
        }

        output = {
            "metadata": metadata,
            "perplexity": perplexity_stats,
            "generation": generation_stats,
            "benchmark": benchmark_stats,
            "memory": memory_stats,
            "architecture": architecture_stats,
            "config": self.config.to_dict(),
        }

        # Generate report outputs
        if self.config.evaluation.generate_reports:
            self.report_generator.generate_report(output)

        return output

    def compare_checkpoints(
        self,
        checkpoint_path_a: str | pathlib.Path,
        checkpoint_path_b: str | pathlib.Path,
    ) -> dict[str, Any]:
        """Compare config, weights, and metadata metrics between two checkpoints.

        Args:
            checkpoint_path_a: Path to checkpoint A.
            checkpoint_path_b: Path to checkpoint B.

        Returns:
            Checkpoint comparison report dict.
        """
        return self.comparator.compare(checkpoint_path_a, checkpoint_path_b)
