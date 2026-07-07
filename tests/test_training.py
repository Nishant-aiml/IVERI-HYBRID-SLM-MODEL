# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for IVERI CORE training engine (Phase 2.2)."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import get_base_config
from core.exceptions import CheckpointError
from training.checkpointing import load_checkpoint, save_checkpoint
from training.mixed_precision import PrecisionHandler
from training.optimizer import get_optimizer
from training.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple dummy model for quick training infrastructure tests."""

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.linear = nn.Linear(32, vocab_size, bias=True)
        self.norm = nn.LayerNorm(32)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        emb = self.embedding(x)
        norm_emb = self.norm(emb)
        logits: torch.Tensor = self.linear(norm_emb)
        if return_dict:
            res: dict[str, torch.Tensor] = {
                "logits": logits,
                "aux_loss": torch.tensor(0.1, device=x.device, requires_grad=True),
            }
            return res
        return logits


# --- PrecisionHandler Tests ---


def test_precision_handler_context() -> None:
    """Verify precision handler handles autocast context and scaler operations."""
    handler = PrecisionHandler(precision="fp16", device="cpu")
    # Autocast on CPU for FP16 is nullcontext
    with handler.autocast_context():
        pass
    assert handler.scaler is None

    # BF16 on CPU
    handler_bf16 = PrecisionHandler(precision="bf16", device="cpu")
    with handler_bf16.autocast_context():
        pass
    assert handler_bf16.scaler is None


# --- Optimizer Parameter Grouping Tests ---


def test_optimizer_parameter_decay_groups() -> None:
    """Verify parameters are correctly split into decayed and non-decayed groups."""
    model = SimpleModel()
    decay_coef = 0.1
    optimizer = get_optimizer(model, learning_rate=3e-4, weight_decay=decay_coef)

    groups = optimizer.param_groups
    assert len(groups) == 2

    # Group 0: decayed (Embedding weight, Linear weight)
    assert groups[0]["weight_decay"] == decay_coef
    # Group 1: not decayed (Biases, LayerNorm weight/bias)
    assert groups[1]["weight_decay"] == 0.0

    decay_names = {p.shape for p in groups[0]["params"]}
    # Embedding: (256, 32), Linear: (256, 32)
    assert torch.Size([256, 32]) in decay_names


# --- Checkpointing Tests ---


def test_checkpoint_save_and_load_roundtrip() -> None:
    """Verify saving and loading checkpoints preserves weights and metadata."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = None

    # Change weights so they are not default
    with torch.no_grad():
        model.linear.weight.fill_(0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "checkpoint.pt"
        cfg = get_base_config()

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=42,
            epoch=3,
            metrics={"val_loss": 0.25},
            config=cfg,
        )

        # Re-initialize second model (should be different from filled 0.5)
        model2 = SimpleModel()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        meta = load_checkpoint(
            path=path,
            model=model2,
            optimizer=optimizer2,
            scaler=scaler,
        )

        assert meta["step"] == 42
        assert meta["epoch"] == 3
        assert meta["metrics"]["val_loss"] == 0.25

        # Check weights match filled 0.5
        assert torch.allclose(model2.linear.weight, torch.full_like(model2.linear.weight, 0.5))


def test_checkpoint_compatibility_assertion() -> None:
    """Verify loading checkpoints checks architecture compatibility and version."""
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "corrupt_checkpoint.pt"
        # Save a dict without iveri headers
        torch.save({"dummy": "state"}, path)

        with pytest.raises(CheckpointError):
            load_checkpoint(path, model)


# --- Trainer Loop Verification Tests ---


def test_trainer_training_step() -> None:
    """Verify Trainer forward pass, loss, gradient propagation, and optimizer step."""
    model = SimpleModel()
    cfg = get_base_config()
    cfg.hardware.device = "cpu"
    cfg.training.batch_size = 2
    cfg.training.seq_len = 8
    cfg.training.gradient_accumulation = 2

    # Create dummy dataset of shape (4, 9) representing inputs + target tokens
    x_data = torch.randint(0, 256, (4, 9), dtype=torch.long)
    inputs = x_data[:, :-1]
    targets = x_data[:, 1:]

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(
        model=model,
        config=cfg,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
    )

    # Re-initialize linear weights to verify they change after optimization step
    with torch.no_grad():
        model.linear.weight.copy_(torch.randn_like(model.linear.weight) * 0.02)
    original_weight = model.linear.weight.clone()

    # Run 1 epoch (containing 2 batches, accumulation=2 -> 1 optimizer step)
    metrics = trainer.train_epoch()
    assert metrics["train_loss"] > 0
    assert metrics["train_aux_loss"] > 0
    assert trainer.global_step == 1

    # Verify parameters updated
    assert not torch.equal(original_weight, model.linear.weight)


def test_trainer_eval_and_checkpoint() -> None:
    """Verify Trainer evaluation saves best and latest checkpoints correctly."""
    model = SimpleModel()
    cfg = get_base_config()
    cfg.hardware.device = "cpu"
    cfg.training.batch_size = 2
    cfg.training.seq_len = 4

    x_data = torch.randint(0, 256, (2, 5), dtype=torch.long)
    dataset = TensorDataset(x_data[:, :-1], x_data[:, 1:])
    dataloader = DataLoader(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.logging.log_dir = tmpdir
        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=dataloader,
            val_dataloader=dataloader,
        )

        metrics = trainer.evaluate()
        assert "val_loss" in metrics

        best_path = pathlib.Path(tmpdir) / "best_model.pt"
        latest_path = pathlib.Path(tmpdir) / "latest_model.pt"

        assert best_path.is_file()
        assert latest_path.is_file()
