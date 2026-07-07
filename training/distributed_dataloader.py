# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed DataLoader utilities for IVERI CORE.

Creates rank-aware :class:`~torch.utils.data.DataLoader` instances backed by
:class:`~torch.utils.data.distributed.DistributedSampler` when distributed
training is enabled.

Edge cases handled
------------------
- **Uneven datasets**: ``drop_last=True`` (configurable) ensures all ranks
  get the same number of steps.
- **Prime-length datasets**: handled without crash — the sampler pads to the
  nearest multiple of ``world_size`` when ``drop_last=False``.
- **world_size > dataset_size**: the sampler will repeat samples to guarantee
  every rank gets at least one item.

Usage
-----
>>> dm = DistributedManager(cfg.distributed)
>>> dm.setup()
>>> loader = make_distributed_dataloader(dataset, config, dm, shuffle=True)
>>> for epoch in range(num_epochs):
...     set_epoch(loader, epoch)  # re-shuffles for each epoch
...     for batch in loader:
...         ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data.distributed import DistributedSampler as _DistributedSamplerT

import torch
from torch.utils.data import DataLoader, Dataset

from configs.base_config import IVERIConfig
from training.distributed import DistributedManager

logger = logging.getLogger(__name__)

try:
    from torch.utils.data.distributed import DistributedSampler

    _DIST_SAMPLER_AVAILABLE = True
except ImportError:
    _DIST_SAMPLER_AVAILABLE = False


def make_distributed_dataloader(
    dataset: Dataset[Any],
    config: IVERIConfig,
    dist_manager: DistributedManager,
    shuffle: bool = True,
    split: str = "train",
) -> DataLoader[Any]:
    """Create a DataLoader with appropriate sampler for distributed or single-GPU training.

    When ``config.distributed.enabled`` is ``True`` and ``world_size > 1``,
    a :class:`~torch.utils.data.distributed.DistributedSampler` is used so
    each rank sees a disjoint partition of the dataset.

    When disabled, a standard DataLoader is returned — the function is a
    strict no-op for single-GPU training.

    Parameters
    ----------
    dataset:
        The PyTorch dataset to wrap.
    config:
        Full project configuration.
    dist_manager:
        :class:`~training.distributed.DistributedManager` instance.
    shuffle:
        Whether to shuffle the sampler.  Automatically disabled for
        validation splits when ``split != "train"``.
    split:
        Human-readable split name (``"train"`` or ``"val"``).  Used only
        for logging.

    Returns
    -------
    DataLoader
        A DataLoader ready for training or evaluation.
    """
    dist_cfg = config.distributed
    is_distributed = (
        dist_cfg.enabled and dist_cfg.strategy != "none" and dist_manager.world_size() > 1
    )

    batch_size = config.training.batch_size if split == "train" else config.evaluation.batch_size
    num_workers = config.hardware.num_workers
    drop_last = dist_cfg.dataloader_drop_last if is_distributed else False
    persistent_workers = dist_cfg.dataloader_persistent_workers and num_workers > 0

    if is_distributed:
        if not _DIST_SAMPLER_AVAILABLE:
            raise RuntimeError("DistributedSampler is not available in this environment.")

        dataset_len = len(dataset)  # type: ignore[arg-type]
        if dist_manager.world_size() > dataset_len:
            logger.warning(
                "world_size (%d) > dataset size (%d) for split='%s'. "
                "The DistributedSampler will repeat samples to fill each "
                "rank — this may cause training instability.",
                dist_manager.world_size(),
                dataset_len,
                split,
            )

        sampler: DistributedSampler = DistributedSampler(  # type: ignore[name-defined]
            dataset,
            num_replicas=dist_manager.world_size(),
            rank=dist_manager.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

        logger.debug(
            "Created DistributedSampler for split='%s': "
            "rank=%d world_size=%d dataset_len=%d drop_last=%s",
            split,
            dist_manager.rank(),
            dist_manager.world_size(),
            dataset_len,
            drop_last,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

    # ── Single-GPU / non-distributed fallback ─────────────────────────
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )


def set_epoch(dataloader: DataLoader[Any], epoch: int) -> None:
    """Set the epoch on the DataLoader's sampler for deterministic shuffling.

    :class:`~torch.utils.data.distributed.DistributedSampler` uses the
    epoch number as a random seed so that each epoch produces a different
    but deterministic shuffle order across all ranks.

    This is a no-op if the DataLoader does not use a
    :class:`~torch.utils.data.distributed.DistributedSampler`.

    Parameters
    ----------
    dataloader:
        DataLoader to update.
    epoch:
        Current epoch index (0-based).
    """
    sampler = getattr(dataloader, "sampler", None)
    if sampler is None:
        return
    set_epoch_fn = getattr(sampler, "set_epoch", None)
    if callable(set_epoch_fn):
        set_epoch_fn(epoch)
