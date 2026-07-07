# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Campaign Health Monitor protecting run execution against stalls, NaNs, and dead nodes."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class CampaignHealthMonitor:
    """Detects anomalies (stalls, cost spikes, NaNs, or GPU hangs) and triggers pauses."""

    def __init__(self, cost_limit_usd: float = 100.0) -> None:
        self.cost_limit_usd = cost_limit_usd
        self.consecutive_stalls = 0
        self.last_loss: float | None = None

    def check_health(
        self,
        metrics: dict[str, float],
        hardware: dict[str, float],
    ) -> tuple[str, str]:
        """Assess the state of the active run.

        Checks: loss NaNs, dead GPUs, cost overruns, and stalled training steps.
        Returns:
            tuple[str, str]: (status, message_details)
        """
        # 1. NaN/Inf loss checks (Dataset corruption / gradient explosion)
        loss = metrics.get("loss")
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                return "PAUSED", "Loss is NaN or Inf (catastrophic divergence)."

        # 2. Dead GPU check (Hardware hang)
        gpu_util = hardware.get("gpu_utilization")
        if gpu_util is not None and gpu_util < 1e-3:
            return "PAUSED", "GPU utilization is 0% while steps are executing (GPU hang)."

        # 3. Cost limits check (Cost runaways)
        accumulated_cost = hardware.get("accumulated_cost_usd", 0.0)
        if accumulated_cost > self.cost_limit_usd:
            return "PAUSED", f"Accumulated cost (${accumulated_cost:.2f}) exceeded budget (${self.cost_limit_usd:.2f})."

        # 4. Stalled training check
        if loss is not None:
            if self.last_loss is not None:
                # If change is exactly zero, increment stall counter
                if abs(loss - self.last_loss) < 1e-9:
                    self.consecutive_stalls += 1
                else:
                    self.consecutive_stalls = 0

            self.last_loss = loss

            if self.consecutive_stalls >= 5:
                return "PAUSED", "Loss is completely flat for 5 consecutive epochs (stalled training)."

        return "OK", "All health metrics normal."
