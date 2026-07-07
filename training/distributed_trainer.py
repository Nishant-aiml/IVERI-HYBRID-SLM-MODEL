# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed training wrapper for IVERI CORE.

:class:`DistributedTrainer` wraps the frozen :class:`~training.trainer.Trainer`
and adds only what distributed execution requires:

1. **Metric all-reduce** — after every ``train_epoch()`` and ``evaluate()``
   call, *all* scalar metrics (loss, aux_loss, MoE auxiliary loss, Titans
   statistics, telemetry) are reduced via
   :meth:`~training.distributed.DistributedManager.reduce_dict` so that
   reported values are identical regardless of GPU count.

2. **Rank guard on logging** — delegates to
   :class:`~training.distributed_logger.DistributedLogger`.

3. **Rank guard on checkpointing** — delegates to
   :func:`~training.distributed_checkpointing.save_checkpoint_distributed`.

4. **Distributed evaluation** — calls ``evaluate()`` from all ranks,
   reduces metrics on rank 0.

No training logic is duplicated.  The frozen ``Trainer`` drives all
forward/backward passes, gradient accumulation, and optimizer steps.
"""

from __future__ import annotations

import pathlib
from typing import Any

from configs.base_config import IVERIConfig
from training.distributed import DistributedManager
from training.trainer import Trainer


class DistributedTrainer:
    """Thin wrapper around :class:`~training.trainer.Trainer` for multi-GPU training.

    Parameters
    ----------
    trainer:
        A fully constructed, frozen :class:`~training.trainer.Trainer`
        instance.  Its model should already be wrapped via
        :meth:`~training.distributed.DistributedManager.wrap_model` before
        being passed here.
    dist_manager:
        An initialized :class:`~training.distributed.DistributedManager`.
    config:
        Full project configuration.
    """

    def __init__(
        self,
        trainer: Trainer,
        dist_manager: DistributedManager,
        config: IVERIConfig,
    ) -> None:
        self.trainer = trainer
        self.dist_manager = dist_manager
        self.config = config

    # ── Training ───────────────────────────────────────────────────────

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch on this rank, then all-reduce all metrics.

        Every scalar value returned by the frozen ``Trainer.train_epoch()``
        — including loss, aux_loss, MoE auxiliary loss, Titans auxiliary
        statistics, and any telemetry values — is passed through
        :meth:`~training.distributed.DistributedManager.reduce_dict`.

        Returns
        -------
        dict[str, float]
            Globally reduced metrics dictionary.
        """
        metrics = self.trainer.train_epoch()
        return self._reduce_metrics(metrics)

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on this rank, then all-reduce val metrics.

        Returns
        -------
        dict[str, float]
            Globally reduced validation metrics dictionary.
        """
        metrics = self.trainer.evaluate()
        return self._reduce_metrics(metrics)

    # ── Checkpoint passthrough ─────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str | pathlib.Path,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint — rank-0 only (or FSDP distributed save).

        Delegates to :func:`~training.distributed_checkpointing.save_checkpoint_distributed`.
        """
        from training.distributed_checkpointing import save_checkpoint_distributed

        save_checkpoint_distributed(
            path=path,
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            scaler=self.trainer.precision_handler.scaler,
            step=self.trainer.global_step,
            epoch=self.trainer.epoch,
            metrics=metrics or {},
            config=self.config,
            dist_manager=self.dist_manager,
        )

    def resume_from_checkpoint(self, checkpoint_path: str | pathlib.Path) -> None:
        """Resume from checkpoint — delegates to the frozen Trainer."""
        self.trainer.resume_from_checkpoint(checkpoint_path)

    # ── Synchronisation ────────────────────────────────────────────────

    def barrier(self) -> None:
        """Synchronise all ranks."""
        self.dist_manager.barrier()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def global_step(self) -> int:
        """Current global optimiser step (from the inner Trainer)."""
        return self.trainer.global_step

    @property
    def epoch(self) -> int:
        """Current epoch (from the inner Trainer)."""
        return self.trainer.epoch

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen so far (from the inner Trainer)."""
        return self.trainer.best_val_loss

    def is_main_process(self) -> bool:
        """Return ``True`` iff this is rank 0."""
        return self.dist_manager.is_main_process()

    def shutdown_logger(self) -> None:
        """Shutdown the logging session (rank 0 only)."""
        if self.is_main_process():
            self.trainer.shutdown_logger()

    # ── Internal helpers ───────────────────────────────────────────────

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """All-reduce every scalar value in *metrics* across ranks.

        Uses :meth:`~training.distributed.DistributedManager.reduce_dict`
        which handles loss, aux_loss, MoE auxiliary loss, Titans auxiliary
        statistics, and all other float values uniformly.
        """
        return self.dist_manager.reduce_dict(metrics)
