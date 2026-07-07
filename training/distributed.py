# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed training lifecycle manager for IVERI CORE.

:class:`DistributedManager` is an **instance-based** class (not a singleton)
so that unit tests can construct independent, isolated instances without
interfering with each other.

The manager wraps the underlying PyTorch distributed process group and
provides a clean API for model wrapping, collective operations, and
process topology queries.  All methods are safe no-ops when distributed
is not enabled or when ``world_size == 1``.

Usage
-----
>>> dm = DistributedManager(config)
>>> dm.setup()
>>> model = dm.wrap_model(model)
>>> ...
>>> dm.teardown()
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from configs.distributed_config import DistributedConfig

logger = logging.getLogger(__name__)

# ── Optional distributed imports (guard for CPU-only environments) ─────────

try:
    import torch.distributed as dist

    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

try:
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

    _DDP_AVAILABLE = True
except ImportError:
    _DDP_AVAILABLE = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        CPUOffload,
    )

    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False


def _sharding_strategy_from_str(s: str) -> Any:
    """Convert string sharding strategy name to FSDP ``ShardingStrategy`` enum."""
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this environment.")
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if s not in mapping:
        from core.exceptions import ConfigError

        raise ConfigError(
            f"Unknown fsdp_sharding_strategy '{s}'. " f"Valid: {sorted(mapping.keys())}"
        )
    return mapping[s]


class DistributedManager:
    """Instance-based distributed training lifecycle manager.

    One manager instance is created per training run.  Each unit test
    constructs its own instance; there is no class-level shared state.

    Parameters
    ----------
    config:
        :class:`~configs.distributed_config.DistributedConfig` controlling
        strategy, backend, topology, and all distributed flags.
    """

    def __init__(self, config: DistributedConfig) -> None:
        self._config = config
        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialize the PyTorch distributed process group.

        No-op when:
        - ``config.enabled`` is ``False``
        - ``config.strategy`` is ``'none'``
        - ``config.world_size`` is 1
        - Process group is already initialized

        The environment variables ``MASTER_ADDR`` and ``MASTER_PORT``
        are set from the config before ``init_process_group`` is called.
        """
        if (
            not self._config.enabled
            or self._config.strategy == "none"
            or self._config.world_size <= 1
        ):
            logger.debug(
                "DistributedManager.setup(): distributed disabled or "
                "world_size=1 — skipping init_process_group."
            )
            return

        if not _DIST_AVAILABLE:
            raise RuntimeError("torch.distributed is not available in this environment.")

        if self._initialized:
            logger.warning(
                "DistributedManager.setup() called on an already-initialized " "manager.  Skipping."
            )
            return

        os.environ.setdefault("MASTER_ADDR", self._config.master_addr)
        os.environ.setdefault("MASTER_PORT", self._config.master_port)

        import datetime

        timeout = datetime.timedelta(minutes=self._config.timeout_minutes)
        dist.init_process_group(
            backend=self._config.backend,
            rank=self._config.rank,
            world_size=self._config.world_size,
            timeout=timeout,
        )
        self._initialized = True
        logger.info(
            "DistributedManager initialized: rank=%d world_size=%d " "backend=%s strategy=%s",
            self._config.rank,
            self._config.world_size,
            self._config.backend,
            self._config.strategy,
        )

    def teardown(self) -> None:
        """Destroy the process group.  No-op if not initialized."""
        if self._initialized and _DIST_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
            logger.info("DistributedManager: process group destroyed.")

    # ── Topology queries ───────────────────────────────────────────────

    def rank(self) -> int:
        """Global rank of this process (0 when not distributed)."""
        if self._initialized and _DIST_AVAILABLE and dist.is_initialized():
            return dist.get_rank()
        return self._config.rank

    def world_size(self) -> int:
        """Total number of processes (1 when not distributed)."""
        if self._initialized and _DIST_AVAILABLE and dist.is_initialized():
            return dist.get_world_size()
        return self._config.world_size if self._config.enabled else 1

    def local_rank(self) -> int:
        """Local rank within the current node."""
        return self._config.local_rank

    def is_main_process(self) -> bool:
        """Return ``True`` iff this is the rank-0 (or only) process."""
        return self.rank() == 0

    def is_initialized(self) -> bool:
        """Return ``True`` iff ``setup()`` has been called successfully."""
        return self._initialized

    def get_device(self) -> torch.device:
        """Return the appropriate torch device for this rank.

        Uses ``local_rank`` for CUDA device selection so that each process
        on a multi-GPU node uses its own GPU.
        """
        if not self._config.enabled or self._config.strategy == "none":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self._config.local_rank}")
        return torch.device("cpu")

    # ── Collective operations ──────────────────────────────────────────

    def barrier(self) -> None:
        """Synchronise all processes.  No-op when not initialized."""
        if self._initialized and _DIST_AVAILABLE and dist.is_initialized():
            dist.barrier()

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce *tensor* and divide by world_size (mean reduction).

        Returns the input tensor unchanged when not in distributed mode.
        """
        if not (self._initialized and _DIST_AVAILABLE and dist.is_initialized()):
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / self.world_size()

    def all_gather_object(self, obj: Any) -> list[Any]:
        """Gather *obj* from all ranks into a list on every rank.

        Returns ``[obj]`` when not in distributed mode.
        """
        if not (self._initialized and _DIST_AVAILABLE and dist.is_initialized()):
            return [obj]
        gathered: list[Any] = [None] * self.world_size()
        dist.all_gather_object(gathered, obj)
        return gathered

    def reduce_dict(self, metrics: dict[str, float]) -> dict[str, float]:
        """All-reduce every float value in *metrics* via mean reduction.

        This is the primary method for synchronising training metrics
        (loss, aux_loss, MoE auxiliary loss, Titans statistics, telemetry
        dictionaries, architecture statistics) across ranks so that
        reported values are identical regardless of GPU count.

        Non-float values are passed through unchanged.

        Parameters
        ----------
        metrics:
            Dictionary of metric name → value.

        Returns
        -------
        dict[str, float]
            Reduced metrics dictionary.
        """
        if not (self._initialized and _DIST_AVAILABLE and dist.is_initialized()):
            return metrics

        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                t = torch.tensor(float(value), dtype=torch.float64)
                t = self.all_reduce_mean(t)
                reduced[key] = t.item()
            else:
                reduced[key] = value
        return reduced

    # ── Model wrapping ─────────────────────────────────────────────────

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap *model* according to ``config.strategy``.

        Strategy dispatch:
        - ``'none'``  — identity (no wrapping)
        - ``'ddp'``   — :class:`~torch.nn.parallel.DistributedDataParallel`
        - ``'fsdp'``  — :class:`~torch.distributed.fsdp.FullyShardedDataParallel`

        Adding a new backend (e.g. DeepSpeed) requires only a new branch
        here; the ``DistributedTrainer`` API remains unchanged.

        Parameters
        ----------
        model:
            Unwrapped model (any frozen IVERI architecture component).

        Returns
        -------
        nn.Module
            Wrapped model.  Access the original via ``.module`` for DDP
            or the FSDP context API for FSDP.
        """
        if not self._config.enabled or self._config.strategy == "none":
            return model

        if not self._initialized:
            raise RuntimeError("DistributedManager.wrap_model() called before setup().")

        strategy = self._config.strategy

        if strategy == "ddp":
            return self._wrap_ddp(model)
        if strategy == "fsdp":
            return self._wrap_fsdp(model)

        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Add a new branch to DistributedManager.wrap_model()."
        )

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        if not _DDP_AVAILABLE:
            raise RuntimeError("DDP is not available in this environment.")

        device = self.get_device()
        model = model.to(device)
        return DDP(
            model,
            device_ids=[self._config.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=self._config.find_unused_parameters,
            gradient_as_bucket_view=self._config.gradient_as_bucket_view,
            broadcast_buffers=self._config.broadcast_buffers,
        )

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap model with FullyShardedDataParallel."""
        if not _FSDP_AVAILABLE:
            raise RuntimeError("FSDP is not available.  Requires PyTorch >= 1.12.")

        sharding = _sharding_strategy_from_str(self._config.fsdp_sharding_strategy)

        cpu_offload = CPUOffload(offload_params=True) if self._config.fsdp_cpu_offload else None

        mixed_precision_policy: MixedPrecision | None = None
        if self._config.fsdp_mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        wrapped = FSDP(
            model,
            sharding_strategy=sharding,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision_policy,
        )

        if self._config.fsdp_activation_checkpointing:
            try:
                from torch.distributed.fsdp.wrap import (  # type: ignore[attr-defined]
                    apply_activation_checkpointing,
                )

                apply_activation_checkpointing(wrapped)  # type: ignore[attr-defined]
            except (ImportError, AttributeError):
                logger.warning(
                    "apply_activation_checkpointing not available; "
                    "skipping FSDP activation checkpointing."
                )

        return wrapped
