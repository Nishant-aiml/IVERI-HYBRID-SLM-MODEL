# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Entropy analyzer for filtering low and high entropy documents in Phase 8.

Uses Shannon entropy over byte, character, and word distributions to flag
repetitive boilerplate, code sequences, or compressed/binary trash.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Entropy metrics for a single document."""

    byte_entropy: float
    char_entropy: float
    word_entropy: float
    is_low_entropy: bool
    is_high_entropy: bool
    flags: list[str]


@dataclass
class EntropyReport:
    """Aggregate entropy statistics for a batch."""

    total_docs: int
    mean_byte_entropy: float
    std_byte_entropy: float
    low_entropy_count: int
    high_entropy_count: int
    optimal_count: int
    histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_docs": self.total_docs,
            "mean_byte_entropy": round(self.mean_byte_entropy, 4),
            "std_byte_entropy": round(self.std_byte_entropy, 4),
            "low_entropy_count": self.low_entropy_count,
            "high_entropy_count": self.high_entropy_count,
            "optimal_count": self.optimal_count,
            "histogram": self.histogram,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def _shannon_entropy(data: list[Any] | bytes | str) -> float:
    """Compute Shannon entropy in bits (0 to log2(num_classes))."""
    if not data:
        return 0.0
    counter = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


class EntropyAnalyzer:
    """Analyzes character/byte/word Shannon entropy to reject repetitive or random docs."""

    LOW_ENTROPY_THRESHOLD: float = 2.0
    HIGH_ENTROPY_THRESHOLD: float = 7.5

    def __init__(
        self,
        low_threshold: float = 2.0,
        high_threshold: float = 7.5,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def analyze(self, text: str) -> EntropyResult:
        """Analyze a document string and return entropy metrics."""
        byte_data = text.encode("utf-8")
        byte_ent = _shannon_entropy(byte_data)
        char_ent = _shannon_entropy(text)

        # Word-level entropy over first 200 words to avoid OOM
        words = text.split()[:200]
        word_ent = _shannon_entropy(words)

        flags: list[str] = []
        is_low = byte_ent < self.low_threshold
        is_high = byte_ent > self.high_threshold

        if is_low:
            flags.append("low_byte_entropy")
        if is_high:
            flags.append("high_byte_entropy")

        return EntropyResult(
            byte_entropy=byte_ent,
            char_entropy=char_ent,
            word_entropy=word_ent,
            is_low_entropy=is_low,
            is_high_entropy=is_high,
            flags=flags,
        )

    def analyze_bytes(self, data: bytes) -> EntropyResult:
        """Analyze raw bytes directly."""
        byte_ent = _shannon_entropy(data)
        try:
            text = data.decode("utf-8", errors="replace")
            char_ent = _shannon_entropy(text)
            words = text.split()[:200]
            word_ent = _shannon_entropy(words)
        except Exception:
            char_ent = 0.0
            word_ent = 0.0

        is_low = byte_ent < self.low_threshold
        is_high = byte_ent > self.high_threshold

        flags = []
        if is_low:
            flags.append("low_byte_entropy")
        if is_high:
            flags.append("high_byte_entropy")

        return EntropyResult(
            byte_entropy=byte_ent,
            char_entropy=char_ent,
            word_entropy=word_ent,
            is_low_entropy=is_low,
            is_high_entropy=is_high,
            flags=flags,
        )

    def analyze_batch(
        self,
        docs: list[dict],
        text_field: str = "text",
    ) -> tuple[list[dict], EntropyReport]:
        """Analyze a batch of documents and return filtered docs + stats report."""
        kept: list[dict] = []
        low_count = 0
        high_count = 0
        opt_count = 0
        histogram = {
            "0-2": 0, "2-3": 0, "3-4": 0, "4-5": 0, "5-6": 0, "6-7": 0, "7-8": 0
        }
        total_entropy = 0.0
        entropies: list[float] = []

        for doc in docs:
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                text = ""

            res = self.analyze(text)
            e = res.byte_entropy
            total_entropy += e
            entropies.append(e)

            # Histogram binning
            if e < 2.0:
                histogram["0-2"] += 1
            elif e < 3.0:
                histogram["2-3"] += 1
            elif e < 4.0:
                histogram["3-4"] += 1
            elif e < 5.0:
                histogram["4-5"] += 1
            elif e < 6.0:
                histogram["5-6"] += 1
            elif e < 7.0:
                histogram["6-7"] += 1
            else:
                histogram["7-8"] += 1

            if res.is_low_entropy:
                low_count += 1
            elif res.is_high_entropy:
                high_count += 1
            else:
                opt_count += 1
                kept.append(doc)

        total = len(docs)
        mean_ent = total_entropy / total if total > 0 else 0.0
        variance = sum((x - mean_ent) ** 2 for x in entropies) / total if total > 0 else 0.0
        std_ent = math.sqrt(variance)

        report = EntropyReport(
            total_docs=total,
            mean_byte_entropy=mean_ent,
            std_byte_entropy=std_ent,
            low_entropy_count=low_count,
            high_entropy_count=high_count,
            optimal_count=opt_count,
            histogram=histogram,
        )

        logger.info(
            "EntropyAnalyzer: kept %d/%d (removed %d low, %d high)",
            len(kept), total, low_count, high_count,
        )
        return kept, report
