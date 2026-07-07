"""Structured logging framework for the IVERI CORE project.

Provides two distinct logging pipelines:

1. **General logging** — timestamped, levelled messages written to both
   the console (stdout) and a rotating log file (``iveri.log``).
2. **Training-metrics logging** — a specialised logger whose records
   carry ``step``, ``loss``, ``lr``, and ``memory`` attributes so that
   metrics can be formatted in a compact, grep-friendly line format.

Typical usage::

    from utils.logging import get_logger, get_training_logger

    logger = get_logger(__name__)
    logger.info("Starting experiment …")

    train_log = get_training_logger()
    train_log.info("", extra={"step": 42, "loss": 0.3, "lr": 1e-4, "memory": 512.0})
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class IVERIFormatter(logging.Formatter):
    """Standard IVERI log formatter.

    Produces lines of the form::

        [2026-06-29 12:00:00,000] [INFO    ] [module.name] Some message
    """

    _FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FORMAT)


class TrainingMetricsFormatter(logging.Formatter):
    """Compact formatter for training-loop metric lines.

    Expects ``LogRecord`` extras: *step*, *loss*, *lr*, *memory*.
    Produces lines such as::

        [Step 42] loss=0.3000 lr=1.00e-04 mem=512.0MB
    """

    _FORMAT = "[Step %(step)d] loss=%(loss).4f lr=%(lr).2e mem=%(memory).1fMB"

    def __init__(self) -> None:
        super().__init__(fmt=self._FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, supplying safe defaults for missing extras.

        Args:
            record: The log record to format.

        Returns:
            The formatted log string.
        """
        # Guard against missing extras so the formatter never crashes.
        record.step = getattr(record, "step", 0)
        record.loss = getattr(record, "loss", 0.0)
        record.lr = getattr(record, "lr", 0.0)
        record.memory = getattr(record, "memory", 0.0)
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger factories
# ---------------------------------------------------------------------------

_LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
_LOG_BACKUP_COUNT: int = 5


def _has_handler_type(
    logger: logging.Logger,
    handler_cls: type,
) -> bool:
    """Check whether *logger* already owns a handler of *handler_cls*.

    Args:
        logger: The logger instance to inspect.
        handler_cls: The handler class to look for.

    Returns:
        ``True`` if a handler of the given type is already attached.
    """
    return any(isinstance(h, handler_cls) for h in logger.handlers)


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str | Path = "logs",
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Create (or retrieve) a fully-configured IVERI logger.

    On the first call for a given *name* the logger is set up with a
    console handler (writing to ``stdout``) and a
    :class:`RotatingFileHandler` targeting ``<log_dir>/iveri.log``.
    Subsequent calls with the same *name* return the cached logger
    **without** adding duplicate handlers.

    Args:
        name: Hierarchical logger name (typically ``__name__``).
        level: Logging level string — ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, or ``CRITICAL``.
        log_dir: Directory for log files.  Created automatically if it
            does not already exist.
        console: Whether to attach a console (stdout) handler.
        file: Whether to attach a rotating-file handler.

    Returns:
        A :class:`logging.Logger` configured with IVERI formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = IVERIFormatter()

    # --- Console handler ---------------------------------------------------
    if console and not _has_handler_type(logger, logging.StreamHandler):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)

    # --- Rotating file handler ---------------------------------------------
    if file and not _has_handler_type(logger, RotatingFileHandler):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path / "iveri.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger (avoids duplicate output).
    logger.propagate = False

    return logger


def get_training_logger(
    name: str = "training",
    log_dir: str | Path = "logs",
) -> logging.Logger:
    """Create (or retrieve) a training-metrics logger.

    Writes to ``<log_dir>/training.log`` using
    :class:`TrainingMetricsFormatter`.  Duplicate handler guards apply.

    Args:
        name: Logger name (default ``"training"``).
        log_dir: Directory for the training log file.

    Returns:
        A :class:`logging.Logger` configured for training metric lines.
    """
    logger = logging.getLogger(f"iveri.{name}")
    logger.setLevel(logging.INFO)

    if not _has_handler_type(logger, RotatingFileHandler):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            filename=log_path / "training.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(TrainingMetricsFormatter())
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Experiment bookends
# ---------------------------------------------------------------------------


def log_experiment_start(
    logger: logging.Logger,
    config_dict: dict,
) -> None:
    """Log the beginning of an experiment with its full configuration.

    Args:
        logger: Logger instance to write to.
        config_dict: Configuration dictionary to record.
    """
    logger.info("=" * 72)
    logger.info("EXPERIMENT START")
    logger.info("=" * 72)
    for key, value in sorted(config_dict.items()):
        logger.info("  %-30s : %s", key, value)
    logger.info("-" * 72)


def log_experiment_end(
    logger: logging.Logger,
    metrics: dict,
) -> None:
    """Log the end of an experiment with final metrics.

    Args:
        logger: Logger instance to write to.
        metrics: Dictionary of final metric names to values.
    """
    logger.info("-" * 72)
    logger.info("EXPERIMENT END — Final Metrics")
    logger.info("-" * 72)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info("  %-30s : %.6f", key, value)
        else:
            logger.info("  %-30s : %s", key, value)
    logger.info("=" * 72)
