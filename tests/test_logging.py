# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for IVERI CORE logging and telemetry (Phase 2.4)."""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import get_base_config
from training.logger import ExperimentLogger, _flatten_dict
from training.trainer import Trainer


# ── Shared fixtures ────────────────────────────────────────────────────────


def _local_cfg(tmpdir: str) -> Any:
    """Return a config wired to local-only logging (no W&B)."""
    cfg = get_base_config()
    cfg.logging.enabled = True
    cfg.logging.mode = "disabled"  # disable W&B
    cfg.logging.log_dir = tmpdir
    cfg.logging.save_dir = tmpdir
    cfg.logging.csv = True
    cfg.logging.json = True
    cfg.logging.tensorboard = False
    return cfg


def _make_logger(tmpdir: str) -> ExperimentLogger:
    """Create a local-only logger with W&B and TensorBoard disabled."""
    cfg = _local_cfg(tmpdir)
    logger = ExperimentLogger(cfg)
    # Force-enable after init (mode=disabled sets enabled=False internally)
    logger.enabled = True
    logger.use_wandb = False
    logger.use_tb = False
    # Re-point file paths to tmpdir (init_null used defaults)
    logger.save_dir = pathlib.Path(tmpdir)
    logger.csv_path = pathlib.Path(tmpdir) / "metrics.csv"
    logger.jsonl_path = pathlib.Path(tmpdir) / "metrics.jsonl"
    return logger


class DummyModel(nn.Module):
    """Minimal model for trainer integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 4)
        self.proj = nn.Linear(4, 256)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, return_dict: bool = True, **kwargs: Any
    ) -> dict[str, Any] | torch.Tensor:
        # x: (B, S) long  →  logits: (B, S, 256)
        emb = self.embed(x.long())        # (B, S, 4)
        logits = self.proj(emb)           # (B, S, 256)
        telemetry = {
            "average_entropy": 0.42,
            "average_patch_length": 4.5,
            "expert_utilization": 0.80,
            "average_recursion_depth": 2.5,
        }
        if return_dict:
            return {
                "logits": logits,
                "aux_loss": torch.tensor(0.05, requires_grad=True),
                "telemetry": telemetry,
            }
        return logits


# ══════════════════════════════════════════════════════════════════════════
# Logger Initialisation
# ══════════════════════════════════════════════════════════════════════════


def test_logger_disabled_is_noop() -> None:
    """Logger with enabled=False must silently ignore all calls."""
    cfg = get_base_config()
    cfg.logging.enabled = False
    logger = ExperimentLogger(cfg)
    assert logger.enabled is False
    logger.log({"loss": 0.5}, step=1)
    logger.log_experiment_metadata()
    logger.log_hyperparameters()
    logger.shutdown()  # must not raise


def test_logger_disabled_mode_is_noop() -> None:
    """mode='disabled' must produce a no-op logger."""
    cfg = get_base_config()
    cfg.logging.mode = "disabled"
    logger = ExperimentLogger(cfg)
    assert logger.enabled is False
    logger.log({"loss": 0.1})
    logger.shutdown()


def test_logger_missing_api_key_falls_back() -> None:
    """W&B UsageError (no API key) must fall back without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = get_base_config()
        cfg.logging.enabled = True
        cfg.logging.mode = "online"
        cfg.logging.save_dir = tmpdir
        cfg.logging.log_dir = tmpdir
        cfg.logging.csv = True
        cfg.logging.json = True
        # Force W&B init failure even when credentials exist on the host.
        with patch("training.logger._wandb.init", side_effect=RuntimeError("no API key")):
            logger = ExperimentLogger(cfg)
            try:
                assert logger.enabled is True
                assert logger.use_wandb is False  # fell back
                logger.log({"loss": 0.25}, step=1)
            finally:
                logger.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# Local File Backends
# ══════════════════════════════════════════════════════════════════════════


def test_csv_backend_writes_correctly() -> None:
    """Verify CSV file is created and contains correct scalar values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log({"train/loss": 0.5, "train/lr": 1e-4}, step=1)
        logger.log({"train/loss": 0.4, "train/lr": 9e-5}, step=2)
        logger.shutdown()

        csv_path = pathlib.Path(tmpdir) / "metrics.csv"
        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert len(rows) == 2
        assert float(rows[0]["train/loss"]) == pytest.approx(0.5)
        assert float(rows[1]["train/loss"]) == pytest.approx(0.4)


def test_jsonl_backend_writes_correctly() -> None:
    """Verify JSONL file is created and contains correct scalar values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log({"val/loss": 0.35}, step=10)
        logger.shutdown()

        jsonl_path = pathlib.Path(tmpdir) / "metrics.jsonl"
        assert jsonl_path.exists()
        data = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0])
        assert data["val/loss"] == pytest.approx(0.35)
        assert data["step"] == 10


def test_multiple_log_calls_append() -> None:
    """Multiple log() calls must append, not overwrite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        for i in range(5):
            logger.log({"loss": float(i)}, step=i)
        logger.shutdown()

        lines = (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5


# ══════════════════════════════════════════════════════════════════════════
# Metric Sanitisation
# ══════════════════════════════════════════════════════════════════════════


def test_nan_metric_replaced_with_zero() -> None:
    """NaN metric values must be replaced with 0.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log({"loss": float("nan")}, step=1)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert data["loss"] == pytest.approx(0.0)


def test_inf_metric_replaced_with_zero() -> None:
    """Inf metric values must be replaced with 0.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log({"loss": float("inf")}, step=1)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert data["loss"] == pytest.approx(0.0)


def test_neg_inf_metric_replaced_with_zero() -> None:
    """-Inf metric values must be replaced with 0.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log({"score": float("-inf")}, step=1)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert data["score"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════
# Experiment Metadata
# ══════════════════════════════════════════════════════════════════════════


def test_experiment_metadata_logs_keys() -> None:
    """log_experiment_metadata() must write system and version keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log_experiment_metadata(seed=42, dataset_version="v1.0", dataset_hash="abc123")
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert "meta/run_name" in data
        assert "meta/random_seed" in data
        assert data["meta/random_seed"] == 42
        assert "system/python_version" in data
        assert "system/pytorch_version" in data


def test_hyperparameter_logging_flattens_config() -> None:
    """log_hyperparameters() must write a flat representation of IVERIConfig."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log_hyperparameters()
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert "hparam/model/hidden_dim" in data
        assert "hparam/training/learning_rate" in data


# ══════════════════════════════════════════════════════════════════════════
# Architecture Telemetry
# ══════════════════════════════════════════════════════════════════════════


def test_architecture_telemetry_from_dict() -> None:
    """Architecture telemetry dict must be logged under 'telemetry/' prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        logger = _make_logger(tmpdir)
        telemetry = {"average_entropy": 0.45, "expert_utilization": 0.82}
        logger.log_architecture_telemetry(model, step=5, telemetry_dict=telemetry)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert "telemetry/average_entropy" in data
        assert data["telemetry/average_entropy"] == pytest.approx(0.45)
        assert "telemetry/expert_utilization" in data


def test_gradient_and_param_stats_logged() -> None:
    """log_architecture_telemetry() must compute grad/param norms and counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        # Simulate gradients with correct dtype (long for embedding)
        x = torch.randint(0, 256, (2, 4))  # (B=2, S=4)
        out = model(x, return_dict=False)  # (2, 4, 256)
        loss = out.sum()
        loss.backward()

        logger = _make_logger(tmpdir)
        logger.log_architecture_telemetry(model, step=1)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert "param/total_count" in data
        assert "param/trainable_count" in data
        assert data["param/trainable_count"] > 0
        assert "grad/total_norm" in data
        assert data["grad/total_norm"] >= 0.0


def test_memory_telemetry_logged() -> None:
    """Memory telemetry keys must always be present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        logger = _make_logger(tmpdir)
        logger.log_architecture_telemetry(model, step=1)
        logger.shutdown()

        data = json.loads(
            (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        assert "memory/gpu_allocated_mb" in data
        assert "memory/gpu_reserved_mb" in data


# ══════════════════════════════════════════════════════════════════════════
# Failure Recovery
# ══════════════════════════════════════════════════════════════════════════


def test_corrupted_log_dir_recovery() -> None:
    """Logger must recover gracefully when save_dir cannot be written to."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        # Point CSV to a directory (unwritable as a file) to trigger failure
        logger.csv_path = pathlib.Path(tmpdir) / "subdir_not_file"
        logger.csv_path.mkdir(exist_ok=True)
        # Must not raise — exception is caught inside log()
        logger.log({"loss": 0.5}, step=1)
        logger.shutdown()


def test_large_telemetry_dict() -> None:
    """Large telemetry dicts (1000 keys) must log without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        big = {f"metric_{i}": float(i) * 0.001 for i in range(1000)}
        logger.log(big, step=1)
        logger.shutdown()

        lines = (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# ══════════════════════════════════════════════════════════════════════════
# Logging Frequency
# ══════════════════════════════════════════════════════════════════════════


def test_logging_frequency_config() -> None:
    """log_frequency config field must be readable and positive."""
    cfg = get_base_config()
    assert cfg.logging.log_frequency == 10
    cfg.logging.log_frequency = 50
    assert cfg.logging.log_frequency == 50


# ══════════════════════════════════════════════════════════════════════════
# Latency Overhead Benchmark
# ══════════════════════════════════════════════════════════════════════════


def test_logger_overhead_under_10ms() -> None:
    """Average logging overhead must be well below 10 ms per call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        metrics = {f"metric_{i}": float(i) for i in range(20)}

        t0 = time.perf_counter()
        for i in range(200):
            logger.log(metrics, step=i)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / 200) * 1000
        print(f"\n[Overhead] avg={avg_ms:.3f} ms over 200 calls")
        assert avg_ms < 10.0  # generous upper bound; typical is <2 ms
        logger.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# Long-running simulation
# ══════════════════════════════════════════════════════════════════════════


def test_long_run_simulation_10k_steps() -> None:
    """10 000 log() calls must complete without error or memory leak."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        for i in range(10_000):
            logger.log({"loss": 1.0 / (i + 1), "step_time": 0.05}, step=i)
        logger.shutdown()

        lines = (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 10_000


# ══════════════════════════════════════════════════════════════════════════
# Trainer Integration
# ══════════════════════════════════════════════════════════════════════════


def test_trainer_logs_train_and_val_metrics() -> None:
    """Trainer must produce training and validation log entries."""
    model = DummyModel()
    cfg = get_base_config()
    cfg.hardware.device = "cpu"
    cfg.hardware.mixed_precision = "fp32"
    cfg.training.gradient_accumulation = 1
    cfg.logging.frequency = 1
    cfg.logging.log_frequency = 1
    cfg.logging.mode = "disabled"

    # (B=4, S=4): inputs=(4,3), targets=(4,3)
    x = torch.randint(0, 256, (4, 4))
    dataset = TensorDataset(x[:, :-1], x[:, 1:])
    dl = DataLoader(dataset, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.logging.log_dir = tmpdir
        cfg.logging.save_dir = tmpdir

        logger = ExperimentLogger(cfg)
        logger.enabled = True
        logger.use_wandb = False
        logger.use_tb = False
        # Patch paths (W&B/disabled init may set wrong defaults)
        logger.save_dir = pathlib.Path(tmpdir)
        logger.csv_path = pathlib.Path(tmpdir) / "metrics.csv"
        logger.jsonl_path = pathlib.Path(tmpdir) / "metrics.jsonl"

        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=dl,
            val_dataloader=dl,
            logger=logger,
        )
        trainer.train_epoch()
        trainer.evaluate()
        trainer.shutdown_logger()

        lines = (pathlib.Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        parsed = [json.loads(ln) for ln in lines]

        train_entries = [d for d in parsed if "train/loss" in d]
        val_entries = [d for d in parsed if "val/loss" in d]
        assert len(train_entries) >= 1
        assert len(val_entries) >= 1
        # Verify timing telemetry present
        assert "timing/forward_seconds" in train_entries[0]


# ══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════


def test_flatten_dict_nested() -> None:
    """_flatten_dict must correctly flatten nested dictionaries."""
    nested = {"a": {"b": {"c": 1}, "d": 2}, "e": 3}
    flat = _flatten_dict(nested)
    assert flat["a/b/c"] == 1
    assert flat["a/d"] == 2
    assert flat["e"] == 3


def test_flatten_dict_with_prefix() -> None:
    """_flatten_dict with prefix must prepend correctly."""
    flat = _flatten_dict({"x": 10}, prefix="hparam")
    assert "hparam/x" in flat
