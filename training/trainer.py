# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Trainer class orchestration for IVERI CORE pretraining.

Orchestrates training epochs, validation iterations, gradient accumulation,
optimizer steps, mixed precision scaling, metric logging, and checkpointing.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from training.checkpointing import load_checkpoint, save_checkpoint
from training.mixed_precision import PrecisionHandler
from training.optimizer import get_optimizer
from training.logger import ExperimentLogger
from utils.validation import get_gpu_memory_usage


class Trainer:
    """Orchestrates model training and evaluation loops.

    Encapsulates all optimization pipeline controls, maintaining complete separation of concerns
    from the underlying model architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        config: IVERIConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        precision_handler: PrecisionHandler | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: Integrated IVERIModel model instance.
            config: Full project configuration object.
            train_dataloader: DataLoader for the training set.
            val_dataloader: Optional DataLoader for the validation set.
            optimizer: Optional PyTorch optimizer. If None, created from config.
            scheduler: Optional learning rate scheduler.
            precision_handler: Optional PrecisionHandler for mixed precision.
            logger: Optional ExperimentLogger instance.
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Get device
        self.device = torch.device(config.hardware.device)
        self.model.to(self.device)

        # Set optimizer
        if optimizer is None:
            self.optimizer = get_optimizer(
                model=model,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Setup precision handler
        if precision_handler is None:
            self.precision_handler = PrecisionHandler(
                precision=config.hardware.mixed_precision,
                device=config.hardware.device,
            )
        else:
            self.precision_handler = precision_handler

        # Setup experiment logger
        if logger is None:
            self.logger = ExperimentLogger(config)
        else:
            self.logger = logger

        # Training state trackers
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Directories
        self.log_dir = pathlib.Path(config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict[str, float]:
        """Train the model for a single epoch.

        Returns:
            Dictionary of epoch-average training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        step_count = 0
        start_time = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_dataloader):
            step_start = time.perf_counter()

            # Parse batch. Batch is expected to be either a tuple (inputs, targets) or dict
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

            dataloader_time = time.perf_counter() - step_start

            # Autocast mixed precision context
            forward_start = time.perf_counter()
            with self.precision_handler.autocast_context():
                # Forward pass: model returns dict
                outputs = self.model(inputs, return_dict=True)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                    aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))
                    telemetry = outputs.get("telemetry")
                else:
                    logits = outputs
                    aux_loss = torch.tensor(0.0, device=self.device)
                    telemetry = None

                # Autoregressive cross-entropy loss:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                ce_loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)

                # Composite loss: CE + load-balancing aux loss
                loss = ce_loss + 0.01 * aux_loss

                # Scale loss for gradient accumulation
                loss = loss / self.config.training.gradient_accumulation

            forward_time = time.perf_counter() - forward_start

            # Backward pass
            backward_start = time.perf_counter()
            scaled_loss = self.precision_handler.scale_loss(loss)
            scaled_loss.backward()
            backward_time = time.perf_counter() - backward_start

            total_loss += ce_loss.item()
            total_aux_loss += aux_loss.item()

            # Step optimizer after gradient accumulation steps
            if (batch_idx + 1) % self.config.training.gradient_accumulation == 0:
                opt_start = time.perf_counter()
                self.precision_handler.step_optimizer(
                    optimizer=self.optimizer,
                    max_norm=self.config.training.grad_clip,
                )
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_time = time.perf_counter() - opt_start

                sched_start = time.perf_counter()
                if self.scheduler is not None:
                    self.scheduler.step()
                scheduler_time = time.perf_counter() - sched_start

                self.global_step += 1
                step_count += 1

                step_total_time = time.perf_counter() - step_start

                # Metrics calculations
                lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler is not None
                    else self.config.training.learning_rate
                )

                # Print console output
                if self.global_step % self.config.logging.log_every == 0:
                    mem = get_gpu_memory_usage()
                    print(
                        f"[Step {self.global_step}] loss={ce_loss.item():.4f} "
                        f"aux={aux_loss.item():.4f} lr={lr:.2e} "
                        f"VRAM={mem.get('allocated', 0.0):.1f}MB"
                    )

                # Standard structured logging
                if self.global_step % self.config.logging.frequency == 0:
                    metrics = {
                        "train/loss": ce_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/learning_rate": lr,
                        "timing/dataloader_seconds": dataloader_time,
                        "timing/forward_seconds": forward_time,
                        "timing/backward_seconds": backward_time,
                        "timing/optimizer_seconds": optimizer_time,
                        "timing/scheduler_seconds": scheduler_time,
                        "timing/step_total_seconds": step_total_time,
                        "performance/samples_per_sec": float(inputs.size(0) / step_total_time) if step_total_time > 0 else 0.0,
                        "performance/tokens_per_sec": float(inputs.numel() / step_total_time) if step_total_time > 0 else 0.0,
                    }
                    self.logger.log(metrics, step=self.global_step)
                    # Log architecture telemetry
                    self.logger.log_architecture_telemetry(
                        model=self.model,
                        step=self.global_step,
                        telemetry_dict=telemetry,
                    )

                # Save periodic checkpoint
                if self.global_step % self.config.logging.save_every == 0:
                    save_checkpoint(
                        path=self.log_dir / f"checkpoint_{self.global_step}.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.precision_handler.scaler,
                        step=self.global_step,
                        epoch=self.epoch,
                        metrics={"loss": ce_loss.item()},
                        config=self.config,
                    )

        epoch_time = time.perf_counter() - start_time
        avg_loss = total_loss / max(1, len(self.train_dataloader))
        avg_aux = total_aux_loss / max(1, len(self.train_dataloader))

        return {
            "train_loss": avg_loss,
            "train_aux_loss": avg_aux,
            "epoch_time": epoch_time,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the validation set.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        count = 0

        for batch in self.val_dataloader:
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
                    aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))
                else:
                    logits = outputs
                    aux_loss = torch.tensor(0.0, device=self.device)

                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                ce_loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)

            total_loss += ce_loss.item()
            total_aux_loss += aux_loss.item()
            count += 1

        avg_loss = total_loss / max(1, count)
        avg_aux = total_aux_loss / max(1, count)

        # Log evaluation metrics to logger
        val_metrics = {
            "val/loss": avg_loss,
            "val/aux_loss": avg_aux,
        }
        self.logger.log(val_metrics, step=self.global_step)

        # Check if this is the best loss, save best checkpoint
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            save_checkpoint(
                path=self.log_dir / "best_model.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.precision_handler.scaler,
                step=self.global_step,
                epoch=self.epoch,
                metrics={"val_loss": avg_loss},
                config=self.config,
            )

        # Always save latest checkpoint
        save_checkpoint(
            path=self.log_dir / "latest_model.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision_handler.scaler,
            step=self.global_step,
            epoch=self.epoch,
            metrics={"val_loss": avg_loss},
            config=self.config,
        )

        return {
            "val_loss": avg_loss,
            "val_aux_loss": avg_aux,
        }

    def resume_from_checkpoint(self, checkpoint_path: str | pathlib.Path) -> None:
        """Resume training states from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        meta = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision_handler.scaler,
        )
        self.global_step = meta["step"]
        self.epoch = meta["epoch"]
        if "val_loss" in meta["metrics"]:
            self.best_val_loss = meta["metrics"]["val_loss"]
        elif "loss" in meta["metrics"]:
            self.best_val_loss = meta["metrics"]["loss"]
        print(f"Resumed training from checkpoint at step={self.global_step}, epoch={self.epoch}")

    def shutdown_logger(self) -> None:
        """Shutdown the active logging session."""
        self.logger.shutdown()
