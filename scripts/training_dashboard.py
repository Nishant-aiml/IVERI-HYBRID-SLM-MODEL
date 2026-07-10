# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Live CLI training dashboard for IVERI CORE pretraining and SFT runs.

Monitors logs/metrics.jsonl and outputs a continuously updating progress screen,
showing current metrics, step throughput, and estimated completion time (ETA).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format duration in seconds into H:M:S format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def render_dashboard(metrics_path: Path) -> bool:
    """Parse metrics.jsonl and render the CLI dashboard.

    Returns:
        True if the metrics file is active or recently updated, False otherwise.
    """
    if not metrics_path.exists():
        print(f"Waiting for metrics log file to be created at: {metrics_path}")
        return True

    # Check file age (idle if no updates for 10 minutes)
    mtime = metrics_path.stat().st_mtime
    time_since_update = time.time() - mtime
    if time_since_update > 600:
        # Idle/finished
        status_str = "FINISHED / IDLE (No recent updates)"
    else:
        status_str = "ACTIVE"

    # Read and parse all JSON lines
    steps = []
    losses = []
    lrs = []
    tokens_per_sec = []
    timestamps = []

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Check for pretraining or sft metrics keys
                    step = data.get("step")
                    loss = data.get("train/loss") or data.get("sft/train_loss") or data.get("loss")
                    lr = data.get("train/learning_rate") or data.get("sft/learning_rate") or data.get("train/lr")
                    tps = data.get("performance/tokens_per_sec") or data.get("tokens_per_sec")
                    ts = data.get("timestamp")

                    if step is not None and loss is not None:
                        steps.append(step)
                        losses.append(loss)
                        if lr is not None:
                            lrs.append(lr)
                        if tps is not None:
                            tokens_per_sec.append(tps)
                        if ts is not None:
                            timestamps.append(ts)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return True

    if not steps:
        print("No training steps recorded yet in metrics log...")
        return True

    # Compute statistics
    curr_step = steps[-1]
    curr_loss = losses[-1]
    min_loss = min(losses)
    curr_lr = lrs[-1] if lrs else 0.0
    avg_tps = sum(tokens_per_sec[-10:]) / len(tokens_per_sec[-10:]) if tokens_per_sec else 0.0

    # Calculate time-based progress and ETA
    elapsed = 0.0
    eta_str = "Calculating..."
    if len(timestamps) > 1:
        elapsed = timestamps[-1] - timestamps[0]
        step_delta = curr_step - steps[0]
        if step_delta > 0:
            avg_step_time = elapsed / step_delta
            # Estimate total steps from verification levels if possible, default to 1000
            total_steps = 1000
            if curr_step <= 20:
                total_steps = 20
            elif curr_step <= 100:
                total_steps = 100
            elif curr_step <= 1000:
                total_steps = 1000

            remaining_steps = max(0, total_steps - curr_step)
            eta_seconds = remaining_steps * avg_step_time
            eta_str = format_duration(eta_seconds) if remaining_steps > 0 else "Complete"

    # Print dashboard layout
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print(f"                   IVERI CORE TRAINING DASHBOARD")
    print("=" * 70)
    print(f" Status:            {status_str}")
    print(f" Current Step:      {curr_step} steps")
    print(f" Learning Rate:     {curr_lr:.2e}")
    print("-" * 70)
    print(f" Current Loss:      {curr_loss:.4f}")
    print(f" Minimum Loss:      {min_loss:.4f}")
    print("-" * 70)
    print(f" Avg Throughput:    {avg_tps:.2f} tokens/sec")
    print(f" Elapsed Time:      {format_duration(elapsed)}")
    print(f" Estimated ETA:     {eta_str}")
    print("=" * 70)

    # Basic ASCII loss sparkline
    if len(losses) >= 10:
        spark_history = losses[-20:]
        max_h = max(spark_history)
        min_h = min(spark_history)
        span = max(1e-4, max_h - min_h)
        spark_chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        sparkline = ""
        for v in spark_history:
            idx = int(((v - min_h) / span) * (len(spark_chars) - 1))
            sparkline += spark_chars[idx]
        print(f" Loss Trend (last 20 logs): [{sparkline}]")
        print("=" * 70)

    return status_str == "ACTIVE"


def main() -> None:
    """Main dashboard entry point."""
    metrics_file = Path("logs/metrics.jsonl")
    try:
        while True:
            is_active = render_dashboard(metrics_file)
            if not is_active:
                print("Dashboard monitoring paused. Training is idle.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting dashboard monitor.")


if __name__ == "__main__":
    main()
