# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Energy and power profiler tracking GPU wattage draw, Joules/token, and cloud training costs."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# Check for NVML binding availability
_HAS_PYNVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    pass


class EnergyProfiler:
    """Measures energy consumption and estimates operational costs of training runs."""

    def __init__(self, kwh_cost_usd: float = 0.15) -> None:
        self.kwh_cost_usd = kwh_cost_usd
        self.start_time: float = 0.0
        self.accumulated_energy_joules: float = 0.0
        self.last_poll_time: float = 0.0

    def _get_gpu_power_watts(self) -> float:
        """Fetch current GPU power draw in Watts."""
        if _HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # Returns power in milliwatts, convert to Watts
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                return power / 1000.0
            except Exception:
                pass

        # Fallback to nvidia-smi command-line execution
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False
            )
            val = res.stdout.strip()
            if val:
                return float(val)
        except Exception:
            pass

        # Return mock baseline CPU/Host power draw (e.g. 65 Watts host baseline equivalent)
        return 65.0

    def start_session(self) -> None:
        """Start profiling timer and energy logs."""
        self.start_time = time.perf_counter()
        self.last_poll_time = self.start_time
        self.accumulated_energy_joules = 0.0

    def poll(self) -> None:
        """Poll current power draw and accumulate energy usage in Joules."""
        now = time.perf_counter()
        duration = now - self.last_poll_time
        if duration <= 0:
            return

        power = self._get_gpu_power_watts()
        # Energy (Joules) = Power (Watts) * Time (Seconds)
        self.accumulated_energy_joules += power * duration
        self.last_poll_time = now

    def stop_session_and_compute(self, total_tokens: int) -> dict[str, float]:
        """Compute average energy consumption and estimated cost.

        Args:
            total_tokens: Total tokens processed during session.

        Returns:
            dict[str, float]: Energy metrics.
        """
        self.poll()
        runtime = time.perf_counter() - self.start_time

        # If zero tokens, default to 1 to avoid ZeroDivisionError
        tokens = max(1, total_tokens)

        # Convert accumulated Joules to Kilowatt-hours (1 kWh = 3.6e6 Joules)
        kwh = self.accumulated_energy_joules / 3.6e6
        total_cost = kwh * self.kwh_cost_usd

        # Estimate cloud hardware costs (e.g. standard RTX 3050 cloud tier of $0.50 per hour)
        cloud_cost_est = (runtime / 3600.0) * 0.50

        return {
            "total_runtime_seconds": runtime,
            "accumulated_energy_joules": self.accumulated_energy_joules,
            "watts_draw_average": self.accumulated_energy_joules / runtime if runtime > 0 else 0.0,
            "energy_per_token_joules": self.accumulated_energy_joules / tokens,
            "tokens_per_joule": tokens / self.accumulated_energy_joules if self.accumulated_energy_joules > 0 else 0.0,
            "electricity_cost_usd": total_cost,
            "estimated_cloud_cost_usd": cloud_cost_est + total_cost,
        }
