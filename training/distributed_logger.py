# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Distributed-aware logging proxy for IVERI CORE.

:class:`DistributedLogger` wraps the frozen
:class:`~training.logger.ExperimentLogger` and provides two logging modes
configured via :attr:`~configs.distributed_config.DistributedConfig.log_rank_zero_only`:

``log_rank_zero_only = True`` (default)
    Only rank 0 writes metrics.  All other ranks silently skip.  This is
    the production mode — all standard W&B, TensorBoard, JSONL, and CSV
    outputs are written exactly once.

``log_rank_zero_only = False`` (debug mode)
    Every rank writes to a rank-scoped JSONL file
    (``rank0.jsonl``, ``rank1.jsonl``, …) alongside the primary log.
    This is invaluable for diagnosing gradient divergence across ranks,
    identifying ranks with abnormal loss curves, or verifying that all
    ranks see equivalent data distributions.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
from typing import Any

from configs.distributed_config import DistributedConfig
from training.distributed import DistributedManager
from training.logger import ExperimentLogger


class DistributedLogger:
    """Rank-aware logging proxy.

    Parameters
    ----------
    logger:
        Underlying frozen :class:`~training.logger.ExperimentLogger`.
    dist_manager:
        :class:`~training.distributed.DistributedManager` instance.
    config:
        :class:`~configs.distributed_config.DistributedConfig` that
        controls ``log_rank_zero_only`` behaviour.
    """

    def __init__(
        self,
        logger: ExperimentLogger,
        dist_manager: DistributedManager,
        config: DistributedConfig,
    ) -> None:
        self._logger = logger
        self._dist = dist_manager
        self._config = config
        self._rank_file: pathlib.Path | None = None
        self._rank_fh: Any = None  # file handle for rank-scoped JSONL

        if not config.log_rank_zero_only:
            self._setup_rank_file(logger, dist_manager)

    def _setup_rank_file(self, logger: ExperimentLogger, dist_manager: DistributedManager) -> None:
        """Open rank-scoped JSONL file for debug mode."""
        # Try to determine the log directory from the logger's config
        try:
            log_dir = pathlib.Path(logger._config.logging.save_dir)  # type: ignore[attr-defined]
        except AttributeError:
            log_dir = pathlib.Path("logs")

        log_dir.mkdir(parents=True, exist_ok=True)
        rank = dist_manager.rank()
        self._rank_file = log_dir / f"rank{rank}.jsonl"
        self._rank_fh = self._rank_file.open("a", encoding="utf-8")

    # ── Public API ─────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics — delegates to ExperimentLogger from rank 0 only.

        In debug mode (``log_rank_zero_only=False``), every rank also
        writes to its rank-scoped JSONL file.

        Parameters
        ----------
        metrics:
            Dictionary of metric name → value.
        step:
            Current global optimiser step.
        """
        # Always write rank-scoped debug file if debug mode is enabled
        if not self._config.log_rank_zero_only and self._rank_fh is not None:
            self._write_rank_jsonl(metrics, step)

        # Delegate standard logging only from rank 0
        if self._dist.is_main_process():
            self._logger.log(metrics, step=step)

    def log_rank(self, metrics: dict[str, Any], step: int) -> None:
        """Force-write metrics to this rank's JSONL file regardless of mode.

        Useful for per-rank diagnostic information that should always be
        recorded independently of the ``log_rank_zero_only`` setting.

        Parameters
        ----------
        metrics:
            Dictionary of metric name → value.
        step:
            Current global optimiser step.
        """
        if self._rank_fh is None:
            # Debug mode not enabled — open a rank file just for this call
            self._setup_rank_file(self._logger, self._dist)
        self._write_rank_jsonl(metrics, step)

    def _write_rank_jsonl(self, metrics: dict[str, Any], step: int) -> None:
        """Append one JSONL record to the rank-scoped log file."""
        if self._rank_fh is None:
            return
        record = {"step": step, "rank": self._dist.rank(), **metrics}
        self._rank_fh.write(json.dumps(record, default=str) + "\n")
        self._rank_fh.flush()

    def shutdown(self) -> None:
        """Close the rank log file handle and shut down the underlying logger.

        The underlying ``ExperimentLogger.shutdown()`` is called only from
        rank 0 to avoid duplicate W&B / TensorBoard finalization.
        """
        if self._rank_fh is not None:
            with contextlib.suppress(Exception):
                self._rank_fh.close()
            self._rank_fh = None

        if self._dist.is_main_process():
            self._logger.shutdown()

    # ── Passthrough properties ─────────────────────────────────────────

    @property
    def rank_file(self) -> pathlib.Path | None:
        """Path to the rank-scoped debug log file, or ``None`` if not open."""
        return self._rank_file
