# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run CPU/GPU inference benchmark and write measured results to JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import get_base_config  # noqa: E402
from inference.benchmark import benchmark_inference  # noqa: E402
from inference.engine import InferenceEngine  # noqa: E402
from inference.loader import load_inference_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IVERI inference benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        default="reports/phase_6_3_3/inference_benchmark.json",
        help="JSON output path",
    )
    args = parser.parse_args(argv)

    cfg = get_base_config()
    cfg.hardware.device = args.device
    cfg.model.hidden_dim = 64
    cfg.model.num_layers = 2
    cfg.model.num_heads = 2
    cfg.model.mamba_ratio = 1
    cfg.model.num_experts = 2
    cfg.model.num_active_experts = 1
    cfg.model.max_recursion_depth = 2
    cfg.model.titans_memory_dim = 32
    cfg.hardware.mixed_precision = "fp32"
    cfg.validate()

    model = load_inference_model(config=cfg, device=args.device)
    engine = InferenceEngine(model)
    stats = benchmark_inference(engine, runs=args.runs, warmup=args.warmup)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": args.device,
        "model_profile": "nano-reduced (hidden=64, layers=2)",
        "provenance": "MEASURED",
        **stats,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
