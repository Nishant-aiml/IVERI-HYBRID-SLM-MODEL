# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed-aware checkpoint save/load for IVERI CORE.

Wraps the frozen :func:`~training.checkpointing.save_checkpoint` and
:func:`~training.checkpointing.load_checkpoint` functions with distributed
rank guards and FSDP state-dict consolidation.

Supports two checkpoint state-dict modes (configured via
:attr:`~configs.distributed_config.DistributedConfig.checkpoint_state_dict_type`):

``"full"`` (default)
    - **DDP**: rank 0 unwraps ``model.module``, saves its ``state_dict()``.
      Checkpoint format is identical to Phase 2.2 — fully backward compatible.
    - **FSDP**: uses ``FullStateDictConfig(offload_to_cpu=True, rank0_only=True)``
      to consolidate all shards onto rank 0 before saving.  The saved file
      is identical to a single-GPU checkpoint.

``"sharded"``
    - **FSDP only**: each rank saves its own shard as
      ``checkpoint_rank{rank}.pt``.  Required for very large future models
      where full consolidation would exhaust CPU RAM.
    - Load: each rank loads its own shard file.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from training.checkpointing import load_checkpoint, save_checkpoint
from training.distributed import DistributedManager

# ── Optional FSDP imports ──────────────────────────────────────────────────

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        ShardedStateDictConfig,
    )

    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, stripping DDP or FSDP wrappers if present."""
    # DDP wraps the model in .module
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        return model.module
    return model


def save_checkpoint_distributed(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    epoch: int,
    metrics: dict[str, Any] | None,
    config: IVERIConfig,
    dist_manager: DistributedManager,
) -> None:
    """Save a training checkpoint in distributed mode.

    Parameters
    ----------
    path:
        Destination checkpoint path.  For ``"sharded"`` mode the path is
        used as a stem: ``<path>.rank<N>`` files are created per rank.
    model:
        Possibly-wrapped (DDP/FSDP) model.
    optimizer, scheduler, scaler:
        Optimizer, scheduler, and GradScaler instances.
    step, epoch:
        Current training step and epoch indices.
    metrics:
        Metrics dictionary to embed in the checkpoint.
    config:
        Full project configuration.
    dist_manager:
        Initialized :class:`~training.distributed.DistributedManager`.
    """
    path = pathlib.Path(path)
    state_dict_type = config.distributed.checkpoint_state_dict_type
    strategy = config.distributed.strategy if config.distributed.enabled else "none"

    if state_dict_type == "sharded" and strategy == "fsdp":
        _save_sharded(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            epoch=epoch,
            metrics=metrics,
            config=config,
            dist_manager=dist_manager,
        )
        return

    # ── Full state dict (default) ──────────────────────────────────────
    if strategy == "fsdp" and _FSDP_AVAILABLE and isinstance(model, FSDP):
        _save_fsdp_full(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            epoch=epoch,
            metrics=metrics,
            config=config,
            dist_manager=dist_manager,
        )
    else:
        # DDP or single-GPU: rank-0 only save
        if dist_manager.is_main_process():
            save_checkpoint(
                path=path,
                model=_unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                epoch=epoch,
                metrics=metrics or {},
                config=config,
            )
        dist_manager.barrier()


def _save_fsdp_full(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    epoch: int,
    metrics: dict[str, Any] | None,
    config: IVERIConfig,
    dist_manager: DistributedManager,
) -> None:
    """FSDP full state dict consolidation — rank 0 saves the checkpoint."""
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available.  Cannot save FSDP full state dict.")
    full_state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_cfg):
        state_dict = model.state_dict()

    if dist_manager.is_main_process():
        # Temporarily replace model's state_dict for the frozen save_checkpoint
        # by wrapping in a thin module
        class _StateDictModule(nn.Module):
            def __init__(self, sd: dict[str, Any]) -> None:
                super().__init__()
                self._sd = sd

            def state_dict(self, **_kw: Any) -> dict[str, Any]:  # type: ignore[override]
                return self._sd

        proxy = _StateDictModule(state_dict)
        save_checkpoint(
            path=path,
            model=proxy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            epoch=epoch,
            metrics=metrics or {},
            config=config,
        )
    dist_manager.barrier()


def _save_sharded(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    epoch: int,
    metrics: dict[str, Any] | None,
    config: IVERIConfig,
    dist_manager: DistributedManager,
) -> None:
    """FSDP sharded state dict — each rank saves its own shard."""
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available.  Cannot save FSDP sharded state dict.")

    rank = dist_manager.rank()
    rank_path = path.with_suffix(f".rank{rank}{path.suffix}")

    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
        state_dict = model.state_dict()

    class _StateDictModule(nn.Module):
        def __init__(self, sd: dict[str, Any]) -> None:
            super().__init__()
            self._sd = sd

        def state_dict(self, **_kw: Any) -> dict[str, Any]:  # type: ignore[override]
            return self._sd

    proxy = _StateDictModule(state_dict)
    save_checkpoint(
        path=rank_path,
        model=proxy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        epoch=epoch,
        metrics=metrics or {},
        config=config,
    )
    dist_manager.barrier()


def load_checkpoint_distributed(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    dist_manager: DistributedManager,
    map_location: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint in distributed mode.

    Parameters
    ----------
    path:
        Checkpoint file path.  For ``"sharded"`` FSDP, each rank appends
        ``.rank<N>`` to find its own shard.
    model:
        Model to load weights into.
    optimizer, scheduler, scaler:
        Optional state containers to restore.
    dist_manager:
        Initialized :class:`~training.distributed.DistributedManager`.
    map_location:
        Optional override for ``torch.load`` map_location.  Defaults to
        the device for ``dist_manager.get_device()``.

    Returns
    -------
    dict[str, Any]
        Metadata dict: ``{step, epoch, metrics, config}``.
    """
    path = pathlib.Path(path)

    # Determine device for loading
    device = dist_manager.get_device() if map_location is None else None
    _map_loc = map_location or str(device)

    # For sharded FSDP, each rank loads its own shard
    if _FSDP_AVAILABLE and isinstance(model, FSDP) and not path.exists():
        rank = dist_manager.rank()
        rank_path = path.with_suffix(f".rank{rank}{path.suffix}")
        if rank_path.exists():
            path = rank_path

    # All ranks load — the frozen load_checkpoint handles device mapping
    meta = load_checkpoint(
        path=path,
        model=_unwrap_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    dist_manager.barrier()
    return meta
