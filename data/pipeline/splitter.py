# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dataset splitting for IVERI CORE.

Splits list datasets into train, validation, and test sets using configurable
ratios and seeds to guarantee reproducible partitions.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitReport:
    """Attributes detailing dataset partitions."""

    train_count: int
    val_count: int
    test_count: int
    total_count: int
    train_bytes: int
    val_bytes: int
    test_bytes: int
    seed_used: int
    ratios: tuple[float, float, float]
    creation_time: str


class DatasetSplitter:
    """Deterministic splitter supporting normal and small-dataset modes."""

    def __init__(
        self,
        train_ratio: float = 0.98,
        val_ratio: float = 0.01,
        test_ratio: float = 0.01,
        seed: int = 42,
        small_threshold: int = 10_000,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.small_threshold = small_threshold

    def split(
        self,
        data: list[Any],
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int | None = None,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Deterministically split data into train, val, and test lists."""
        tr = train_ratio if train_ratio is not None else self.train_ratio
        vr = val_ratio if val_ratio is not None else self.val_ratio
        te = test_ratio if test_ratio is not None else self.test_ratio
        sd = seed if seed is not None else self.seed

        # Validation
        if abs((tr + vr + te) - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {tr + vr + te}")

        # Shallow copy to shuffle
        shuffled = data.copy()
        rng = random.Random(sd)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * tr)
        n_val = int(n * vr)

        # Ensure validation and test have at least 1 sample if dataset is not empty
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if (n - n_train - n_val) == 0:
                n_train -= 1

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test

    def split_small_dataset(self, data: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
        """Split using 90/5/5 ratios recommended for small local/Stage 3B data."""
        return self.split(data, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05)

    def auto_split(self, data: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
        """Choose ratio automatically based on dataset size."""
        if len(data) < self.small_threshold:
            return self.split_small_dataset(data)
        return self.split(data)

    def save_splits(
        self, train: list[Any], val: list[Any], test: list[Any], output_dir: str | Path, name: str
    ) -> None:
        """Save partitions as JSON files to output directory."""
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)

        for partition, p_data in [("train", train), ("val", val), ("test", test)]:
            path = od / f"{name}_{partition}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(p_data, f, indent=4)
            logger.info(f"Saved {partition} split to {path}")

    def load_splits(
        self, output_dir: str | Path, name: str
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Load partitions from JSON files in output directory."""
        od = Path(output_dir)
        splits = []
        for partition in ["train", "val", "test"]:
            path = od / f"{name}_{partition}.json"
            with open(path, encoding="utf-8") as f:
                splits.append(json.load(f))
        return splits[0], splits[1], splits[2]

    def generate_report(
        self, train: list[Any], val: list[Any], test: list[Any], seed: int
    ) -> SplitReport:
        """Compute statistics and return a SplitReport."""
        tr_cnt, v_cnt, te_cnt = len(train), len(val), len(test)
        total = tr_cnt + v_cnt + te_cnt

        # Calculate bytes
        def get_bytes(lst: list[Any]) -> int:
            b_len = 0
            for item in lst:
                if isinstance(item, str):
                    b_len += len(item.encode("utf-8"))
                elif isinstance(item, dict):
                    b_len += len(json.dumps(item).encode("utf-8"))
            return b_len

        tr_bytes = get_bytes(train)
        v_bytes = get_bytes(val)
        te_bytes = get_bytes(test)

        ratios = (
            tr_cnt / total if total > 0 else 0.0,
            v_cnt / total if total > 0 else 0.0,
            te_cnt / total if total > 0 else 0.0,
        )

        return SplitReport(
            train_count=tr_cnt,
            val_count=v_cnt,
            test_count=te_cnt,
            total_count=total,
            train_bytes=tr_bytes,
            val_bytes=v_bytes,
            test_bytes=te_bytes,
            seed_used=seed,
            ratios=ratios,
            creation_time=datetime.now().isoformat(),
        )
