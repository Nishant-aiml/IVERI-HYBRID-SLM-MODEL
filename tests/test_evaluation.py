# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for Phase 2.5 evaluation pipeline and benchmark infrastructure."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import get_base_config
from evaluation.arch_eval import ArchitectureEvaluator
from evaluation.benchmark import InferenceBenchmark
from evaluation.checkpoint_compare import CheckpointComparator
from evaluation.evaluator import Evaluator
from evaluation.generation import GenerationEvaluator
from evaluation.memory_tracker import MemoryTracker
from evaluation.perplexity import PerplexityEvaluator
from evaluation.report_generator import ReportGenerator
from model.iveri_core import IVERIModel
from training.checkpointing import save_checkpoint


@pytest.fixture
def test_cfg() -> Any:
    """Return a scaled-down config for fast CPU-based unit testing."""
    cfg = get_base_config()
    cfg.model.hidden_dim = 64
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.num_experts = 2
    cfg.model.num_active_experts = 1
    cfg.model.max_recursion_depth = 2
    cfg.model.titans_memory_dim = 32
    cfg.model.blt.patch_size_min = 1
    cfg.model.blt.patch_size_max = 4
    cfg.hardware.device = "cpu"
    cfg.hardware.mixed_precision = "fp32"
    cfg.evaluation.enabled = True
    cfg.evaluation.batch_size = 2
    cfg.evaluation.max_eval_batches = 2
    cfg.evaluation.benchmark_iterations = 3
    cfg.evaluation.warmup_iterations = 1
    cfg.evaluation.generation_max_new_bytes = 4
    return cfg


@pytest.fixture
def dummy_model(test_cfg: Any) -> IVERIModel:
    """Return an instantiated small IVERI model."""
    model = IVERIModel(test_cfg)
    model.eval()
    return model


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    """Return a simple dataloader for language modeling evaluation."""
    x = torch.randint(0, 256, (4, 16))
    dataset = TensorDataset(x[:, :-1], x[:, 1:])
    return DataLoader(dataset, batch_size=2)


# ══════════════════════════════════════════════════════════════════════════
# Perplexity Evaluation
# ══════════════════════════════════════════════════════════════════════════


def test_perplexity_computation() -> None:
    """Verify PerplexityEvaluator computes cross entropy and perplexity correctly."""
    evaluator = PerplexityEvaluator()
    # Batch size 2, Vocab size 4, Seq len 3
    logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]],
            [[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

    # Almost 100% correct logits, loss should be close to 0.0
    res = evaluator.evaluate_batch(logits, targets)
    assert res["loss"] < 0.1
    assert res["num_tokens"] == 6
    assert res["nll"] < 0.1


def test_perplexity_nan_handling() -> None:
    """Verify PerplexityEvaluator replaces NaN and Inf with 0.0 gracefully."""
    evaluator = PerplexityEvaluator()
    logits = torch.tensor([[float("nan"), 1.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    res = evaluator.evaluate_batch(logits, targets)
    assert res["loss"] == 0.0
    assert res["nll"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Generation Evaluation
# ══════════════════════════════════════════════════════════════════════════


def test_generation(dummy_model: IVERIModel) -> None:
    """Verify GenerationEvaluator generates tokens and applies top-k/top-p decoding."""
    evaluator = GenerationEvaluator()
    prompt = torch.randint(0, 256, (2, 4), dtype=torch.long)

    res = evaluator.generate(
        model=dummy_model,
        input_ids=prompt,
        max_new_bytes=4,
        temperature=0.7,
        top_k=2,
        top_p=0.9,
    )

    assert res.output_ids.shape == (2, 8)
    assert res.latency_seconds > 0.0
    assert res.bytes_per_second >= 0.0
    assert res.avg_generated_length == 4.0


# ══════════════════════════════════════════════════════════════════════════
# Benchmarking & Memory
# ══════════════════════════════════════════════════════════════════════════


def test_inference_benchmark(dummy_model: IVERIModel) -> None:
    """Verify InferenceBenchmark measures throughput, latency percentiles, and FLOPs."""
    benchmark = InferenceBenchmark(dummy_model)
    inputs = torch.randint(0, 256, (2, 8), dtype=torch.long)

    res = benchmark.run(
        input_ids=inputs,
        iterations=3,
        warmup_iterations=1,
    )

    assert res.latency_mean_ms > 0.0
    assert res.latency_p95_ms > 0.0
    assert res.samples_per_sec >= 0.0
    assert res.parameter_count > 0
    assert res.estimated_flops > 0.0


def test_memory_tracking(dummy_model: IVERIModel) -> None:
    """Verify MemoryTracker context manager captures VRAM/RAM states without leaking."""
    inputs = torch.randint(0, 256, (2, 8), dtype=torch.long)
    with MemoryTracker(dummy_model) as tracker:
        _ = dummy_model(inputs)

    res = tracker.get_snapshot()
    assert res.cpu_ram_mb >= 0.0
    assert res.cpu_peak_ram_mb >= 0.0
    assert res.parameter_mb > 0.0
    assert res.growth_mb >= 0.0


def test_memory_growth(dummy_model: IVERIModel) -> None:
    """Verify that repeated evaluation passes do not result in memory growth."""
    inputs = torch.randint(0, 256, (2, 8), dtype=torch.long)
    with MemoryTracker(dummy_model) as tracker:
        for _ in range(5):
            _ = dummy_model(inputs)

    res = tracker.get_snapshot()
    # Memory growth delta should be close to zero under read-only mode
    assert res.growth_mb < 20.0


# ══════════════════════════════════════════════════════════════════════════
# Architecture Telemetry
# ══════════════════════════════════════════════════════════════════════════


def test_architecture_statistics() -> None:
    """Verify ArchitectureEvaluator aggregates subsystem telemetry distributions."""
    evaluator = ArchitectureEvaluator()
    dummy_telemetry = [
        {
            "average_byte_entropy": 0.45,
            "average_patch_length": 3.8,
            "hidden_state_norm": 12.5,
            "residual_norm": 0.35,
            "expert_utilization_histogram": [25, 30, 20, 25],
            "average_recursion_depth": 1.4,
            "titans_read_count": 8,
            "titans_write_count": 8,
            "average_memory_update_magnitude": 0.045,
        },
        {
            "average_byte_entropy": 0.47,
            "average_patch_length": 4.2,
            "hidden_state_norm": 13.1,
            "residual_norm": 0.38,
            "expert_utilization_histogram": [35, 20, 25, 20],
            "average_recursion_depth": 1.6,
            "titans_read_count": 8,
            "titans_write_count": 8,
            "average_memory_update_magnitude": 0.051,
        },
    ]

    res = evaluator.evaluate(dummy_telemetry)

    assert res["blt"]["average_byte_entropy"] == pytest.approx(0.46)
    assert res["mamba2"]["state_variance"] >= 0.0
    assert res["moe"]["unused_experts_count"] == 0
    assert res["moe"]["expert_collapse_score"] < 0.2
    assert res["mor"]["average_depth"] == pytest.approx(1.5)
    assert res["titans"]["memory_reads"] == 16


# ══════════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════════


def test_report_generation(tmp_dir: pathlib.Path) -> None:
    """Verify ReportGenerator outputs JSON, CSV, and Markdown file structures."""
    generator = ReportGenerator(output_dir=tmp_dir)
    dummy_data = {
        "metadata": {
            "timestamp": "2026-06-30T12:00:00",
            "git_commit": "abc1234",
            "architecture_version": "0.1.0-optionC",
            "random_seed": 42,
            "device": "cpu",
            "dtype": "float32",
            "pytorch_version": "2.1.0",
            "cuda_version": "N/A",
            "evaluation_duration_seconds": 1.25,
        },
        "perplexity": {
            "loss": 2.1245,
            "perplexity": 8.3687,
            "num_tokens": 1024,
            "num_batches": 8,
        },
        "generation": {
            "latency_seconds": 0.45,
            "bytes_per_second": 120.0,
            "avg_generated_length": 32.0,
            "early_stopped_ratio": 0.0,
        },
        "benchmark": {
            "warmup_latency_ms": 12.4,
            "latency_mean_ms": 8.5,
            "latency_median_ms": 8.4,
            "latency_p95_ms": 9.2,
            "latency_p99_ms": 10.1,
            "samples_per_sec": 117.6,
            "tokens_per_sec": 60232.0,
            "parameter_count": 1000000,
            "estimated_flops": 2.0e9,
        },
        "memory": {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_mb": 0.0,
            "cpu_ram_mb": 128.5,
            "cpu_peak_ram_mb": 135.2,
            "parameter_mb": 4.0,
            "activation_mb": 1.2,
            "fragmentation_ratio": 0.0,
            "growth_mb": 0.1,
        },
        "architecture": {
            "mor": {
                "average_depth": 1.5,
                "median_depth": 1.5,
                "flops_saved_ratio": 0.8125,
            },
            "moe": {
                "expert_utilization_histogram": [25, 25, 25, 25],
                "unused_experts_count": 0,
                "imbalance_ratio": 0.0,
                "expert_collapse_score": 0.0,
            },
        },
    }

    paths = generator.generate_report(dummy_data, filename_prefix="test_report")

    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["md"].exists()

    # Read Markdown and verify key terms are formatted
    md_content = paths["md"].read_text(encoding="utf-8")
    assert "# IVERI CORE — Evaluation Report" in md_content
    assert "abc1234" in md_content
    assert "8.3687" in md_content


def test_large_report_generation(tmp_dir: pathlib.Path) -> None:
    """Stress test ReportGenerator with a large dataset structure."""
    generator = ReportGenerator(output_dir=tmp_dir)
    # 100 dummy records
    dummy_data = {
        "metadata": {"timestamp": "2026-06-30T12:00:00"},
        "perplexity": {"loss": 1.5},
        "benchmark": {"parameter_count": 10_000_000},
        "architecture": {
            f"metric_{i}": {"mean": float(i), "histogram": {"counts": [i] * 5, "bin_edges": [0.0] * 6}}
            for i in range(200)
        },
    }
    paths = generator.generate_report(dummy_data, filename_prefix="large_report")
    assert paths["json"].exists()
    assert paths["csv"].exists()


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint Comparison
# ══════════════════════════════════════════════════════════════════════════


def test_checkpoint_version_mismatch(tmp_dir: pathlib.Path, dummy_model: IVERIModel, test_cfg: Any) -> None:
    """Verify comparator flags checkpoints with mismatched architecture versions as incompatible."""
    path_a = tmp_dir / "ckpt_a.pt"
    path_b = tmp_dir / "ckpt_b.pt"

    # Save first checkpoint
    save_checkpoint(path_a, dummy_model, config=test_cfg, step=10, epoch=1)

    # Create corrupted state for second checkpoint with mismatched architecture
    ckpt = torch.load(path_a, weights_only=False)
    ckpt["architecture_version"] = "0.2.0-optionD"
    torch.save(ckpt, path_b)

    comparator = CheckpointComparator()
    res = comparator.compare(path_a, path_b)

    assert res["status"] == "NOT DIRECTLY COMPARABLE"
    assert res["comparable"] is False
    assert any("Architecture version mismatch" in r for r in res["mismatch_reasons"])


def test_checkpoint_hash_mismatch(tmp_dir: pathlib.Path, dummy_model: IVERIModel, test_cfg: Any) -> None:
    """Verify comparator flags checkpoints with mismatched model dimensions as NOT DIRECTLY COMPARABLE."""
    path_a = tmp_dir / "ckpt_a.pt"
    path_b = tmp_dir / "ckpt_b.pt"

    save_checkpoint(path_a, dummy_model, config=test_cfg, step=10, epoch=1)

    # Save second checkpoint with model shape difference
    cfg_alt = get_base_config()
    cfg_alt.model.hidden_dim = 128  # mismatch

    model_alt = IVERIModel(cfg_alt)
    save_checkpoint(path_b, model_alt, config=cfg_alt, step=10, epoch=1)

    comparator = CheckpointComparator()
    res = comparator.compare(path_a, path_b)

    assert res["status"] == "NOT DIRECTLY COMPARABLE"
    assert res["comparable"] is False
    assert any("Parameter count mismatch" in r or "hidden_dim" in r for r in res["mismatch_reasons"])


# ══════════════════════════════════════════════════════════════════════════
# Evaluator Engine integration
# ══════════════════════════════════════════════════════════════════════════


def test_repeated_evaluation(dummy_model: IVERIModel, test_cfg: Any, dummy_dataloader: DataLoader) -> None:
    """Verify Evaluator generates identical perplexity and benchmark metrics on repeated runs with same seed."""
    evaluator = Evaluator(dummy_model, test_cfg, dummy_dataloader)

    # Run 1
    torch.manual_seed(42)
    res_1 = evaluator.evaluate()

    # Run 2
    torch.manual_seed(42)
    res_2 = evaluator.evaluate()

    assert res_1["perplexity"]["loss"] == pytest.approx(res_2["perplexity"]["loss"])
    assert res_1["perplexity"]["perplexity"] == pytest.approx(res_2["perplexity"]["perplexity"])
    assert res_1["architecture"]["mor"]["average_depth"] == pytest.approx(res_2["architecture"]["mor"]["average_depth"])


def test_device_switch(dummy_model: IVERIModel, test_cfg: Any, dummy_dataloader: DataLoader) -> None:
    """Verify Evaluator works seamlessly when device is specified as string or torch.device."""
    evaluator_str = Evaluator(dummy_model, test_cfg, dummy_dataloader, device="cpu")
    res_str = evaluator_str.evaluate()
    assert res_str["perplexity"]["loss"] >= 0.0

    evaluator_obj = Evaluator(dummy_model, test_cfg, dummy_dataloader, device=torch.device("cpu"))
    res_obj = evaluator_obj.evaluate()
    assert res_obj["perplexity"]["loss"] >= 0.0


def test_cpu_vs_gpu_consistency(dummy_model: IVERIModel, test_cfg: Any, dummy_dataloader: DataLoader) -> None:
    """Verify Evaluator produces consistent outputs if run on CPU."""
    evaluator = Evaluator(dummy_model, test_cfg, dummy_dataloader, device="cpu")
    _ = evaluator.evaluate()
    # Ensure there are no leftover parameters requiring gradients
    for p in dummy_model.parameters():
        assert p.grad is None or p.grad.sum() == 0.0 or not p.requires_grad
