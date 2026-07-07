# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""CLI for Stage 3B proprietary dataset ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.proprietary_ingest import (  # noqa: E402
    ProprietaryIngestError,
    count_proprietary_records,
    ingest_stage3b,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("iveri.ingest_stage3b")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Stage 3B proprietary JSON data")
    parser.add_argument(
        "--proprietary-dir",
        default="data/proprietary",
        help="Root directory containing university_papers/, gate_questions/, etc.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage3b",
        help="Processed output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pii", action="store_true", help="Skip PII cleaning")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Count and validate records without writing processed splits",
    )
    args = parser.parse_args(argv)

    try:
        if args.validate_only:
            counts = count_proprietary_records(args.proprietary_dir)
            total = sum(counts.values())
            print(json.dumps({"total": total, "by_source": counts}, indent=2))
            if total == 0:
                logger.warning("No proprietary records found.")
                return 1
            return 0

        report = ingest_stage3b(
            proprietary_dir=args.proprietary_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            clean_pii=not args.no_pii,
        )
        print(json.dumps(asdict(report), indent=2))
        return 0
    except ProprietaryIngestError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
