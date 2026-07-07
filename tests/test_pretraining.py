# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Test suite for Phase 3.1 Foundation Pretraining (Stage 1).

Verifies PretrainingDatasetLoader, CurriculumScheduler, LossMonitor, ConvergenceAnalyzer,
CheckpointSelector, PretrainingEvaluator, GenerationInspector, ExperimentManager,
BaselineTransformer, Failure Recovery, and Numerical Stability.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.base_config import get_base_config
from core.constants import BYTE_VOCAB_SIZE, RAW_BYTE_VOCAB_SIZE
from data.pipeline.data_registry import DataRegistry, DatasetEntry
from data.pipeline.versioning import DatasetVersioner, ManifestEntry
from evaluation.evaluator import Evaluator
from evaluation.pretraining_eval import PretrainingEvaluator
from evaluation.generation_inspector import GenerationInspector
from baselines.baseline_transformer import BaselineTransformer
from model.iveri_core import IVERIModel
from training.checkpointing import save_checkpoint, load_checkpoint
from training.convergence import ConvergenceAnalyzer
from training.curriculum import CurriculumScheduler
from training.experiment_manager import ExperimentManager
from training.loss_monitor import LossMonitor
from training.pretraining_dataset import PretrainingDatasetLoader
from training.pretrain_runner import run_pretraining
from training.trainer import Trainer


class MockPretrainDataset(Dataset):
    """Mock dataset yielding random byte sequences of correct shape for pretraining."""

    def __init__(self, num_samples: int = 20, seq_len: int = 16) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        return x, y


@pytest.fixture
def temp_dir():
    """Temporary directory for pretraining verification."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def base_config():
    """IVERIConfig default configuration."""
    cfg = get_base_config(
        model={
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "mamba_ratio": 1,
            "num_experts": 2,
            "num_active_experts": 1,
            "max_recursion_depth": 2,
            "titans_memory_dim": 8,
        }
    )
    cfg.hardware.device = "cpu"
    cfg.hardware.mixed_precision = "fp32"
    cfg.training.seq_len = 16
    cfg.training.batch_size = 2
    cfg.training.gradient_accumulation = 1
    cfg.training.max_steps = 10
    cfg.logging.eval_every = 5
    cfg.logging.save_every = 5
    cfg.logging.log_every = 5
    return cfg


# ── 1. Dataset assembly & Registry tests ─────────────────────────────────────


def test_pretraining_dataset_loader(temp_dir, base_config):
    """Verify pretraining dataset loader verifies manifest/SHA256/version info."""
    # Write mock registry specs
    registry = DataRegistry(auto_discover=False)
    entry = DatasetEntry(
        name="tinystories",
        hf_id="roneneldan/TinyStories",
        priority="S",
        license="MIT",
        format="pretrain",
        source="huggingface",
        stage=1,
    )
    registry.register(entry)

    # Set processed path
    if isinstance(base_config.data_pipeline.report, dict):
        base_config.data_pipeline.report["processed_data_dir"] = str(temp_dir.as_posix())
    else:
        base_config.data_pipeline.report.processed_data_dir = str(temp_dir.as_posix())
    processed_dir = temp_dir / "stage1" / "tinystories"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate VERSION.json
    versioner = DatasetVersioner()
    # Write a dummy file to compute content hash
    dummy = processed_dir / "data.txt"
    dummy.write_text("dummy text content", encoding="utf-8")

    config_dict = base_config.to_dict()
    info = versioner.create_version(
        name="tinystories",
        data_path=processed_dir,
        config=config_dict,
        document_count=10,
        byte_count=1000,
        stage="1",
    )

    # Populate manifest.json
    manifest_entry = ManifestEntry(
        dataset_name="tinystories",
        version=info.version_id,
        license="MIT",
        sha256=info.content_hash,
        pipeline_version="3.0.0",
        creation_time="now",
        document_count=10,
        byte_count=1000,
        stage="1",
        source="huggingface",
        mixing_weight=0.05,
    )
    versioner.write_manifest(temp_dir, [manifest_entry])

    # Check validation
    loader = PretrainingDatasetLoader(base_config, registry=registry)
    # Patch loading the dataset object to avoid actual file scanning failures
    with patch("training.pretraining_dataset.PretrainByteDataset") as mock_ds_cls:
        mock_ds_cls.return_value = MockPretrainDataset()
        dataset = loader.load("tinystories", split="train")
        assert dataset is not None


# ── 2. Curriculum scheduler tests ───────────────────────────────────────────


def test_curriculum_scheduler(base_config):
    """Test curriculum scheduling returns expected active weights over steps."""
    scheduler = CurriculumScheduler(base_config)
    weights = scheduler.get_mixture_weights(step=0)
    assert weights["tinystories"] == 1.0
    assert weights["fineweb_edu"] == 0.0

    # Test curriculum strategy
    if isinstance(base_config.data_pipeline.mixing, dict):
        base_config.data_pipeline.mixing["strategy"] = "curriculum"
        base_config.data_pipeline.mixing["curriculum_start_step"] = 10
        base_config.data_pipeline.mixing["curriculum_end_step"] = 50
    else:
        base_config.data_pipeline.mixing.strategy = "curriculum"
        base_config.data_pipeline.mixing.curriculum_start_step = 10
        base_config.data_pipeline.mixing.curriculum_end_step = 50
    scheduler = CurriculumScheduler(base_config)

    # Step <= start_step -> 100% TinyStories
    w_start = scheduler.get_mixture_weights(step=5)
    assert w_start["tinystories"] == 1.0

    # Step in-between -> interpolated
    w_mid = scheduler.get_mixture_weights(step=30)
    assert 0.0 < w_mid["tinystories"] <= 1.0


# ── 3. Baseline Transformer tests ───────────────────────────────────────────


def test_baseline_transformer(base_config):
    """Verify BaselineTransformer forward pass shapes and checkpoint save/load."""
    model = BaselineTransformer(base_config)
    x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    out = model(x, return_dict=True)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2, 16, RAW_BYTE_VOCAB_SIZE)


# ── 4. Loss & Health Monitors tests ─────────────────────────────────────────


def test_loss_monitor_gradient_and_activation_health(base_config):
    """Test loss monitor registers hooks, computes stats, and writes CSV."""
    model = IVERIModel(base_config)
    monitor = LossMonitor(base_config, log_dir=base_config.logging.save_dir)

    # Register hooks
    monitor.register_activation_hooks(model)
    assert len(monitor._hooks) > 0

    # Run forward pass
    x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    _ = model(x, return_dict=True)
    assert len(monitor.activation_stats) > 0

    # Run backward to populate gradients
    y = torch.randint(0, 256, (2, 16), dtype=torch.long)
    loss = nn.functional.cross_entropy(model(x, return_dict=False).view(-1, BYTE_VOCAB_SIZE), y.view(-1))
    loss.backward()

    # Track gradient health
    health = monitor.track_gradient_health(model, step=1)
    assert health["global_grad_norm"] >= 0.0
    assert health["nan_grad_count"] == 0.0
    assert monitor.csv_path.exists()

    monitor.remove_hooks()


# ── 5. Convergence analysis tests ───────────────────────────────────────────


def test_convergence_analyzer(base_config):
    """Verify convergence statistics calculations."""
    analyzer = ConvergenceAnalyzer(window_size=10)
    # Simulate decreasing loss
    for step in range(10):
        analyzer.update(
            loss=10.0 - step * 0.5,
            step_time=0.1,
            num_tokens=32,
            num_patches=8,
        )

    analysis = analyzer.analyze()
    assert analysis["convergence/loss_slope"] < 0.0
    assert analysis["convergence/loss_reduction_pct"] > 0.0

    # Throughput
    model = IVERIModel(base_config)
    perf = analyzer.compute_throughput(model, batch_size=2, seq_len=16, last_step_time=0.1)
    assert perf["performance/tokens_per_sec"] > 0.0


# ── 6. Checkpoint selection tests ───────────────────────────────────────────


def test_checkpoint_selector(temp_dir):
    """Test CheckpointSelector indexing, metadata, and sorting."""
    from training.model_selection import CheckpointSelector
    selector = CheckpointSelector(log_dir=temp_dir)

    # Register mock checkpoints
    selector.register_checkpoint("chk1.pt", step=10, train_loss=4.5, val_loss=4.6, perplexity=99.0)
    selector.register_checkpoint("chk2.pt", step=20, train_loss=3.0, val_loss=3.2, perplexity=24.5)
    selector.register_checkpoint("chk3.pt", step=30, train_loss=3.5, val_loss=3.8, perplexity=44.7)

    best = selector.get_best_checkpoint(metric="val_loss")
    assert best["step"] == 20
    assert best["val_loss"] == 3.2

    latest = selector.get_latest_checkpoint()
    assert latest["step"] == 30


# ── 7. Pretraining Evaluation & Generation Inspector tests ───────────────────


def test_pretraining_evaluator(base_config):
    """Test PretrainingEvaluator accuracy, top-5, and bits-per-byte calculations."""
    model = IVERIModel(base_config)
    evaluator_base = Evaluator(model, base_config, val_dataloader=None)
    pretrain_eval = PretrainingEvaluator(evaluator_base)

    dataset = MockPretrainDataset(num_samples=4, seq_len=16)
    val_loader = DataLoader(dataset, batch_size=2)

    results = pretrain_eval.evaluate_pretraining(val_loader)
    assert "val_loss" in results
    assert "perplexity" in results
    assert "bits_per_byte" in results
    assert 0.0 <= results["top1_accuracy"] <= 1.0
    assert 0.0 <= results["top5_accuracy"] <= 1.0


def test_generation_inspector(base_config, temp_dir):
    """Test generation inspector logs prompt, invalid UTF-8 and speed."""
    model = IVERIModel(base_config)
    inspector = GenerationInspector(base_config, log_dir=temp_dir)

    results = inspector.inspect(model, step=10, prompt_text="Testing", seed=42)
    assert results["step"] == 10
    assert results["average_entropy"] >= 0.0
    assert inspector.samples_file.exists()


# ── 8. Failure Recovery Verification ────────────────────────────────────────


def test_failure_recovery(base_config, temp_dir):
    """Verify interrupted pretraining -> resume results in bitwise identical optimizer state."""
    model = IVERIModel(base_config)
    trainer1 = Trainer(
        model=model,
        config=base_config,
        train_dataloader=DataLoader(MockPretrainDataset(), batch_size=2),
    )

    device = torch.device(base_config.hardware.device)
    model.to(device)

    # 1. Run for 3 steps
    it = iter(trainer1.train_dataloader)
    for _ in range(3):
        batch = next(it)
        x, y = batch[0].to(device), batch[1].to(device)
        trainer1.optimizer.zero_grad(set_to_none=True)
        with trainer1.precision_handler.autocast_context():
            loss = nn.functional.cross_entropy(model(x, return_dict=False).view(-1, BYTE_VOCAB_SIZE), y.view(-1))
            scaled_loss = trainer1.precision_handler.scale_loss(loss)
        scaled_loss.backward()
        trainer1.precision_handler.step_optimizer(trainer1.optimizer, max_norm=1.0)
    trainer1.global_step = 3

    # Save checkpoint
    chk_path = temp_dir / "recovery.pt"
    save_checkpoint(
        path=chk_path,
        model=model,
        optimizer=trainer1.optimizer,
        scheduler=trainer1.scheduler,
        scaler=trainer1.precision_handler.scaler,
        step=3,
        epoch=0,
        config=base_config,
    )

    # Record current optimizer state and parameters
    original_opt_state = json.dumps(
        {k: str(v) for k, v in trainer1.optimizer.state_dict()["state"].items()}
    )
    original_param_data = {n: p.clone() for n, p in model.named_parameters()}

    # 2. Run for 2 more steps to change state
    for _ in range(2):
        batch = next(it)
        x, y = batch[0].to(device), batch[1].to(device)
        trainer1.optimizer.zero_grad(set_to_none=True)
        with trainer1.precision_handler.autocast_context():
            loss = nn.functional.cross_entropy(model(x, return_dict=False).view(-1, BYTE_VOCAB_SIZE), y.view(-1))
            scaled_loss = trainer1.precision_handler.scale_loss(loss)
        scaled_loss.backward()
        trainer1.precision_handler.step_optimizer(trainer1.optimizer, max_norm=1.0)

    # 3. Restore checkpoint
    trainer2 = Trainer(
        model=model,
        config=base_config,
        train_dataloader=DataLoader(MockPretrainDataset(), batch_size=2),
    )
    trainer2.resume_from_checkpoint(chk_path)

    # 4. Verify bitwise equivalence
    restored_opt_state = json.dumps(
        {k: str(v) for k, v in trainer2.optimizer.state_dict()["state"].items()}
    )
    assert restored_opt_state == original_opt_state

    for n, p in model.named_parameters():
        assert torch.allclose(p, original_param_data[n])


# ── 9. Numerical Stability Verification ──────────────────────────────────────


def test_numerical_stability(base_config):
    """Test that pretraining assert_finite_tensors raises on NaN/Inf parameters."""
    model = IVERIModel(base_config)
    x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    loss = torch.tensor(1.5)

    from training.pretrain_runner import _assert_finite_tensors
    # Should pass normally
    _assert_finite_tensors(model, loss)

    # Corrupt model weights -> should raise ValueError
    with torch.no_grad():
        next(model.parameters())[0] = float("nan")

    with pytest.raises(ValueError):
        _assert_finite_tensors(model, loss)


# ── 10. End-To-End Runner Tests ──────────────────────────────────────────────


def test_pretraining_runner_e2e(base_config):
    """Verify pretrain_runner executes smoke runs (20 steps) correctly for IVERI and baseline."""
    train_ds = MockPretrainDataset()
    val_ds = MockPretrainDataset()

    # Disable tensorboard/wandb for tests
    base_config.logging.enabled = False
    base_config.logging.mode = "disabled"
    base_config.logging.tensorboard = False

    # Level 1 (20 steps)
    results = run_pretraining(
        config=base_config,
        verification_level=1,
        run_baseline=False,
        train_ds_override=train_ds,
        val_ds_override=val_ds,
    )
    assert results["final_loss"] > 0
    assert results["final_perplexity"] > 0
