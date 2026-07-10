# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Failure analyzer and recovery orchestrator for IVERI CORE training campaigns.

Classifies training exceptions by severity level and determines automatic
recovery actions (e.g., checkpoint rollback, learning rate reduction, or retry).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from training.instability_tracker import DivergenceError

logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    """Structured details of a training failure and recommended recovery actions."""
    error_class: str
    message: str
    severity: str  # "INFO" | "WARNING" | "CRITICAL"
    is_recoverable: bool
    recovery_action: str  # "NONE" | "RETRY_WITH_ROLLBACK" | "RETRY_WITH_SMALLER_BATCH" | "RETRY_AFTER_COOLDOWN"
    reason: str


class FailureAnalyzer:
    """Analyzes runtime training errors and proposes self-healing strategies."""

    def __init__(self) -> None:
        pass

    def analyze(self, error: Exception) -> FailureAnalysis:
        """Classify a training exception and return a recovery prescription.

        Args:
            error: The python exception caught during training execution.

        Returns:
            A FailureAnalysis object.
        """
        error_class = error.__class__.__name__
        msg = str(error)

        # 1. Out of Memory (OOM) Errors
        if "out of memory" in msg.lower() or error_class in ("OutOfMemoryError", "RuntimeError") and "cuda" in msg.lower() and "memory" in msg.lower():
            return FailureAnalysis(
                error_class=error_class,
                message=msg,
                severity="WARNING",
                is_recoverable=True,
                recovery_action="RETRY_WITH_SMALLER_BATCH",
                reason="GPU VRAM exceeded capacity. Recovery requires reducing batch size and clearing cache."
            )

        # 2. Gradient or Instability Divergences (NaN / Inf / DivergenceError)
        if isinstance(error, (DivergenceError, ValueError)) and any(x in msg.lower() for x in ("nan", "inf", "instability", "divergence")):
            return FailureAnalysis(
                error_class=error_class,
                message=msg,
                severity="CRITICAL",
                is_recoverable=True,
                recovery_action="RETRY_WITH_ROLLBACK",
                reason="Numerical instability or parameters exploded (NaN/Inf). Recovery requires rolling back to the previous stable checkpoint and lowering learning rate."
            )

        # 3. Connection and Networking Issues (WandB, HTTP, API keys)
        if error_class in ("ConnectionError", "HTTPError", "TimeoutError") or any(x in msg.lower() for x in ("network", "connection refused", "wandb")):
            return FailureAnalysis(
                error_class=error_class,
                message=msg,
                severity="WARNING",
                is_recoverable=True,
                recovery_action="RETRY_AFTER_COOLDOWN",
                reason="Temporary network disruption. Recovery requires sleeping and retrying connection."
            )

        # 4. Critical Configuration or Code Errors (Syntax, FileNotFoundError, etc.)
        return FailureAnalysis(
            error_class=error_class,
            message=msg,
            severity="CRITICAL",
            is_recoverable=False,
            recovery_action="NONE",
            reason="Non-transient code, configuration, or environment error. Requires manual intervention."
        )
