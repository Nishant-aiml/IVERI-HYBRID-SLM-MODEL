# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Fault tolerance utilities for distributed training in IVERI CORE.

Provides :class:`FaultToleranceHandler` which wraps training loops with:

- **Barrier timeout detection** via ``dist.monitored_barrier()``.
- **Graceful shutdown** — broadcast shutdown signal, destroy process group,
  and clean up temporary files before exiting.
- **Exception context manager** — ``protect()`` wraps the training loop,
  calling ``cleanup_on_exception()`` on any failure so that hung processes
  are not left dangling after an interrupted job.

Without this, an interrupted distributed job may leave background processes
consuming GPU memory indefinitely.

Usage
-----
>>> ft = FaultToleranceHandler(dist_manager, config.distributed)
>>> with ft.protect():
...     for epoch in range(num_epochs):
...         trainer.train_epoch()
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
from collections.abc import Generator
from typing import Any

from configs.distributed_config import DistributedConfig
from training.distributed import DistributedManager

logger = logging.getLogger(__name__)

# ── Optional distributed imports ───────────────────────────────────────────

try:
    import datetime

    import torch.distributed as dist

    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


class FaultToleranceHandler:
    """Timeout recovery, failed rank detection, and graceful shutdown.

    Parameters
    ----------
    dist_manager:
        :class:`~training.distributed.DistributedManager` instance.
    config:
        :class:`~configs.distributed_config.DistributedConfig` controlling
        timeout and graceful shutdown behaviour.
    """

    def __init__(
        self,
        dist_manager: DistributedManager,
        config: DistributedConfig,
    ) -> None:
        self._dist = dist_manager
        self._config = config
        self._shutdown_requested = False
        self._cleanup_callbacks: list[Any] = []

        # Register SIGTERM handler for graceful shutdown on cluster preemption
        if self._config.graceful_shutdown:
            self._register_signal_handler()

    # ── Public API ─────────────────────────────────────────────────────

    def check_all_ranks_alive(self) -> bool:
        """Check that all ranks are responsive via a monitored barrier.

        Uses ``dist.monitored_barrier()`` with the configured timeout.
        Returns ``True`` if all ranks respond within the timeout, ``False``
        if any rank is detected as non-responsive.

        In single-process or non-distributed mode this always returns ``True``.

        Returns
        -------
        bool
            ``True`` if all ranks are alive, ``False`` on timeout.
        """
        if not (self._dist.is_initialized() and _DIST_AVAILABLE and dist.is_initialized()):
            return True

        timeout_seconds = self._config.timeout_minutes * 60
        try:
            timeout = datetime.timedelta(seconds=timeout_seconds)
            dist.monitored_barrier(timeout=timeout)
            return True
        except Exception as exc:
            logger.error(
                "FaultToleranceHandler: monitored_barrier timed out. "
                "A rank may be hung. Details: %s",
                exc,
            )
            return False

    def graceful_shutdown(self, reason: str = "") -> None:
        """Broadcast shutdown signal, destroy process group, and cleanup.

        Safe to call from any rank.  Logs the reason on rank 0.

        Parameters
        ----------
        reason:
            Human-readable explanation for the shutdown (logged on rank 0).
        """
        if self._shutdown_requested:
            return  # Avoid double-shutdown
        self._shutdown_requested = True

        if self._dist.is_main_process() and reason:
            logger.warning(
                "FaultToleranceHandler: graceful shutdown requested. " "Reason: %s",
                reason,
            )

        self._run_cleanup_callbacks()

        with contextlib.suppress(Exception):
            self._dist.barrier()

        self._dist.teardown()
        logger.debug(
            "FaultToleranceHandler: process group destroyed on rank %d.",
            self._dist.rank(),
        )

    def cleanup_on_exception(self, exc: BaseException) -> None:
        """Handle an exception during distributed training.

        Logs the error, runs registered cleanup callbacks, and calls
        :meth:`graceful_shutdown`.

        Parameters
        ----------
        exc:
            The exception that caused training to fail.
        """
        logger.error(
            "FaultToleranceHandler: exception on rank %d: %s: %s",
            self._dist.rank(),
            type(exc).__name__,
            exc,
        )
        self.graceful_shutdown(reason=f"{type(exc).__name__}: {exc}")

    def register_cleanup(self, callback: Any) -> None:
        """Register a callable to be called during graceful shutdown.

        Parameters
        ----------
        callback:
            Zero-argument callable (e.g. close a file handle, delete a
            temp directory, finalize a logger).
        """
        self._cleanup_callbacks.append(callback)

    @contextlib.contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager that calls :meth:`cleanup_on_exception` on failure.

        Wraps the training loop so that if any exception escapes,
        ``cleanup_on_exception`` is called before the exception propagates.
        This ensures hung processes are cleaned up even when training
        crashes mid-epoch.

        Example
        -------
        >>> with ft.protect():
        ...     for epoch in range(num_epochs):
        ...         trainer.train_epoch()
        """
        try:
            yield
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            self.cleanup_on_exception(exc)
            raise

    # ── Internal helpers ───────────────────────────────────────────────

    def _run_cleanup_callbacks(self) -> None:
        """Execute all registered cleanup callbacks, logging any errors."""
        for cb in self._cleanup_callbacks:
            try:
                cb()
            except Exception as exc:
                logger.warning(
                    "FaultToleranceHandler: cleanup callback %r raised: %s",
                    cb,
                    exc,
                )

    def _register_signal_handler(self) -> None:
        """Register SIGTERM handler for cluster preemption graceful shutdown."""
        if not hasattr(signal, "SIGTERM"):
            return  # SIGTERM not available on all platforms (e.g., Windows)

        original_handler = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum: int, frame: Any) -> None:
            logger.warning(
                "FaultToleranceHandler: SIGTERM received on rank %d. "
                "Initiating graceful shutdown.",
                self._dist.rank(),
            )
            self.graceful_shutdown(reason="SIGTERM received")
            # Restore and re-raise
            signal.signal(signal.SIGTERM, original_handler)
            sys.exit(signum)

        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except (OSError, ValueError):
            # May fail in environments where signal handling is restricted
            logger.debug("FaultToleranceHandler: could not register SIGTERM handler.")
