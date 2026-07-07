# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for IVERI CORE inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from configs.base_config import get_base_config
from inference.engine import InferenceEngine
from inference.loader import load_inference_model
from inference.sampling import SamplingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("iveri.inference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IVERI CORE inference CLI")
    parser.add_argument("--prompt", required=True, help="Input text prompt")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    args = parser.parse_args(argv)

    cfg = get_base_config()
    cfg.hardware.device = args.device
    model = load_inference_model(args.checkpoint, config=cfg, device=args.device)
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    engine = InferenceEngine(model, sampling=sampling)

    if args.stream:
        for chunk in engine.stream(args.prompt):
            if chunk.text_delta:
                sys.stdout.write(chunk.text_delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return 0

    result = engine.generate(args.prompt)
    print(json.dumps({"text": result.text, "tokens_per_second": result.tokens_per_second}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
