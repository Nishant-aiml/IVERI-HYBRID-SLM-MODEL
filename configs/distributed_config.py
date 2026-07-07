# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed training configuration for IVERI CORE.

Kept as a separate module from :mod:`configs.base_config` so that
the configuration system remains modular as more phases are added.
Imported by :class:`~configs.base_config.IVERIConfig`.

Quick start
-----------
>>> from configs.distributed_config import DistributedConfig
>>> cfg = DistributedConfig()            # single-GPU defaults (disabled)
>>> cfg = DistributedConfig(enabled=True, strategy="ddp", world_size=4)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import ConfigError

# ── Allowed values ─────────────────────────────────────────────────────────

_VALID_STRATEGIES: frozenset[str] = frozenset({"none", "ddp", "fsdp"})
_VALID_BACKENDS: frozenset[str] = frozenset({"nccl", "gloo", "mpi"})
_VALID_FSDP_SHARDING: frozenset[str] = frozenset({"full_shard", "shard_grad_op", "no_shard"})
_VALID_STATE_DICT_TYPES: frozenset[str] = frozenset({"full", "sharded"})


@dataclass(frozen=False, slots=True)
class DistributedConfig:
    """Configuration for distributed training infrastructure.

    Attributes
    ----------
    enabled:
        Master switch.  When ``False`` the entire distributed stack is a
        strict no-op and the project behaves identically to pre-2.6.
    strategy:
        Distribution strategy: ``'none'``, ``'ddp'``, or ``'fsdp'``.
        Adding a new backend (e.g. DeepSpeed) requires only a new branch
        in :meth:`~training.distributed.DistributedManager.wrap_model`.
    backend:
        PyTorch process-group backend: ``'nccl'`` (GPU), ``'gloo'`` (CPU),
        or ``'mpi'``.
    world_size:
        Total number of processes across all nodes.  Must be ≥ 1.
    rank:
        Global rank of this process.  Must satisfy ``0 <= rank < world_size``.
    local_rank:
        Rank within the current node (used for device assignment).
    master_addr:
        Hostname or IP of the rank-0 process for rendezvous.
    master_port:
        Port for the rank-0 rendezvous server.

    find_unused_parameters:
        DDP flag — enable when some parameters are conditionally unused.
    gradient_as_bucket_view:
        DDP flag — reduces gradient memory copy overhead.
    broadcast_buffers:
        DDP flag — synchronise non-parameter buffers at step start.
    sync_gradients:
        Whether to synchronise gradients across ranks.

    fsdp_sharding_strategy:
        FSDP parameter sharding mode: ``'full_shard'``, ``'shard_grad_op'``,
        or ``'no_shard'``.
    fsdp_cpu_offload:
        Offload FSDP parameters and gradients to CPU.
    fsdp_mixed_precision:
        Enable FSDP mixed-precision policy.
    fsdp_activation_checkpointing:
        Apply activation checkpointing inside FSDP wrapping.

    checkpoint_state_dict_type:
        ``'full'`` — consolidate all shards to rank 0 before saving
        (default; identical format to Phase 2.2 checkpoints).
        ``'sharded'`` — each rank saves its own shard; required for very
        large models where full consolidation would OOM.

    dataloader_drop_last:
        Drop the last incomplete batch in the distributed sampler.
    dataloader_persistent_workers:
        Keep DataLoader worker processes alive between epochs.

    log_rank_zero_only:
        ``True`` — only rank 0 writes logs (default).
        ``False`` — debug mode: every rank writes to a rank-scoped file
        (``rank0.jsonl``, ``rank1.jsonl``, …) for gradient divergence
        diagnosis.
    eval_rank_zero_only:
        Run evaluation only on rank 0 (reduces eval overhead; may
        under-sample validation data in multi-rank scenarios).

    timeout_minutes:
        Collective operation timeout in minutes for fault-tolerance
        monitoring.
    graceful_shutdown:
        Attempt graceful cleanup (barrier + destroy process group) before
        exiting on failure.
    """

    # ── Master switch ───────────────────────────────────────────────────
    enabled: bool = False

    # ── Strategy & backend ──────────────────────────────────────────────
    strategy: str = "ddp"
    backend: str = "nccl"

    # ── Process topology ────────────────────────────────────────────────
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"

    # ── DDP flags ───────────────────────────────────────────────────────
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True
    sync_gradients: bool = True

    # ── FSDP flags ──────────────────────────────────────────────────────
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = False

    # ── Checkpoint type ─────────────────────────────────────────────────
    checkpoint_state_dict_type: str = "full"

    # ── DataLoader options ──────────────────────────────────────────────
    dataloader_drop_last: bool = True
    dataloader_persistent_workers: bool = True

    # ── Logging ─────────────────────────────────────────────────────────
    log_rank_zero_only: bool = True
    eval_rank_zero_only: bool = False

    # ── Fault tolerance ─────────────────────────────────────────────────
    timeout_minutes: int = 30
    graceful_shutdown: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ConfigError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}, " f"got '{self.strategy}'"
            )
        if self.backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"backend must be one of {sorted(_VALID_BACKENDS)}, " f"got '{self.backend}'"
            )
        if self.fsdp_sharding_strategy not in _VALID_FSDP_SHARDING:
            raise ConfigError(
                f"fsdp_sharding_strategy must be one of "
                f"{sorted(_VALID_FSDP_SHARDING)}, "
                f"got '{self.fsdp_sharding_strategy}'"
            )
        if self.checkpoint_state_dict_type not in _VALID_STATE_DICT_TYPES:
            raise ConfigError(
                f"checkpoint_state_dict_type must be one of "
                f"{sorted(_VALID_STATE_DICT_TYPES)}, "
                f"got '{self.checkpoint_state_dict_type}'"
            )
        if self.world_size < 1:
            raise ConfigError(f"world_size must be >= 1, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ConfigError(
                f"rank must satisfy 0 <= rank < world_size " f"({self.world_size}), got {self.rank}"
            )
        if not (0 <= self.local_rank < self.world_size):
            raise ConfigError(
                f"local_rank must satisfy 0 <= local_rank < world_size "
                f"({self.world_size}), got {self.local_rank}"
            )
        if self.timeout_minutes <= 0:
            raise ConfigError(f"timeout_minutes must be > 0, got {self.timeout_minutes}")
        if self.checkpoint_state_dict_type == "sharded" and self.strategy != "fsdp":
            raise ConfigError("checkpoint_state_dict_type='sharded' requires strategy='fsdp'")
