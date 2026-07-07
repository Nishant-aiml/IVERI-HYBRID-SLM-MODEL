# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte counting and sizing statistics for IVERI CORE data pipeline.

Since IVERI works at the byte level directly, all dataset sizing and capacity
calculations are done in bytes/GB rather than token counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class ByteStats:
    """Detailed byte stats for a text dataset."""

    num_documents: int
    total_bytes: int
    total_gb: float
    avg_bytes: float
    min_bytes: int
    max_bytes: int
    median_bytes: float
    p25_bytes: float
    p75_bytes: float
    p95_bytes: float
    p99_bytes: float
    empty_doc_count: int
    dataset_name: str
    stage: str


class ByteCounter:
    """Counts bytes and calculates percentiles across datasets."""

    def count(self, text: str) -> int:
        """Compute the UTF-8 byte count of a string."""
        return len(text.encode("utf-8"))

    def count_bytes(self, data: bytes) -> int:
        """Compute size of raw bytes object."""
        return len(data)

    def count_dataset(self, texts: list[str], dataset_name: str = "", stage: str = "") -> ByteStats:
        """Calculate complete statistics of a text list."""
        if not texts:
            return ByteStats(
                num_documents=0,
                total_bytes=0,
                total_gb=0.0,
                avg_bytes=0.0,
                min_bytes=0,
                max_bytes=0,
                median_bytes=0.0,
                p25_bytes=0.0,
                p75_bytes=0.0,
                p95_bytes=0.0,
                p99_bytes=0.0,
                empty_doc_count=0,
                dataset_name=dataset_name,
                stage=stage,
            )

        sizes = []
        empty = 0
        for t in texts:
            n_bytes = self.count(t)
            sizes.append(n_bytes)
            if n_bytes == 0:
                empty += 1

        sizes_arr = np.array(sizes)
        total = int(np.sum(sizes_arr))

        return ByteStats(
            num_documents=len(texts),
            total_bytes=total,
            total_gb=total / (1024**3),
            avg_bytes=float(np.mean(sizes_arr)),
            min_bytes=int(np.min(sizes_arr)),
            max_bytes=int(np.max(sizes_arr)),
            median_bytes=float(np.median(sizes_arr)),
            p25_bytes=float(np.percentile(sizes_arr, 25)),
            p75_bytes=float(np.percentile(sizes_arr, 75)),
            p95_bytes=float(np.percentile(sizes_arr, 95)),
            p99_bytes=float(np.percentile(sizes_arr, 99)),
            empty_doc_count=empty,
            dataset_name=dataset_name,
            stage=stage,
        )

    def estimate_training_bytes(
        self, stage_weights: dict[str, float], dataset_stats: dict[str, ByteStats]
    ) -> float:
        """Estimate the effective size of a mixed training dataset."""
        effective_bytes = 0.0
        for name, weight in stage_weights.items():
            if name in dataset_stats:
                effective_bytes += dataset_stats[name].total_bytes * weight
        return effective_bytes

    def format_bytes(self, n: int) -> str:
        """Convert byte integer to human-readable string."""
        val = float(n)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if val < 1024.0:
                return f"{val:.2f} {unit}"
            val /= 1024.0
        return f"{val:.2f} PB"

    def print_summary(self, stats: ByteStats) -> str:
        """Generate a structured summary string."""
        return (
            f"=== Byte Statistics: {stats.dataset_name} (Stage {stats.stage}) ===\n"
            f"Documents:      {stats.num_documents:,}\n"
            f"Total Bytes:    {stats.total_bytes:,} ({self.format_bytes(stats.total_bytes)})\n"
            f"Avg Size:       {stats.avg_bytes:.1f} bytes\n"
            f"Min/Max Size:   {stats.min_bytes:,} / {stats.max_bytes:,} bytes\n"
            f"Median Size:    {stats.median_bytes:.1f} bytes\n"
            f"P25 / P75 Size: {stats.p25_bytes:.1f} / {stats.p75_bytes:.1f} bytes\n"
            f"P95 / P99 Size: {stats.p95_bytes:.1f} / {stats.p99_bytes:.1f} bytes\n"
            f"Empty Docs:     {stats.empty_doc_count:,}\n"
            f"=================================================="
        )

    def compare_token_bytes(
        self, total_bytes: int, bytes_per_token: float = 4.0
    ) -> dict[str, float]:
        """Estimate equivalents in terms of subword tokens."""
        return {
            "estimated_tokens": total_bytes / bytes_per_token,
            "bytes_per_token": bytes_per_token,
        }
