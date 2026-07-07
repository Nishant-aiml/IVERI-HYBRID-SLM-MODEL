# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Exact and near-duplicate detection for IVERI CORE data pipeline.

Uses MD5 hashing for fast duplicate lookup, SHA256 for integrity verification.
MinHash LSH is supported as an optional feature, falling back to exact-only if
datasketch is not installed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

# Optional import of datasketch
try:
    from datasketch import MinHash, MinHashLSH

    _MINHASH_AVAILABLE = True
except ImportError:
    _MINHASH_AVAILABLE = False
    MinHash = None  # type: ignore[assignment,misc]
    MinHashLSH = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class DeduplicationReport:
    """Report summarizing the deduplication operation results."""

    total_input: int
    kept: int
    removed_exact: int
    removed_near: int
    removal_rate: float
    exact_hashes: set[str] = field(default_factory=set)
    near_dedup_available: bool = _MINHASH_AVAILABLE
    threshold_used: float = 0.8


@dataclass(frozen=False, slots=True)
class DeduplicationConfig:
    """Settings configuration for deduplication."""

    exact_enabled: bool = True
    near_enabled: bool = True
    threshold: float = 0.8
    num_perm: int = 128
    compute_sha256: bool = True


class Deduplicator:
    """Detects and removes duplicates (exact and near-duplicates)."""

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self.config = config or DeduplicationConfig()
        if self.config.near_enabled and not _MINHASH_AVAILABLE:
            logger.warning(
                "datasketch package not installed. Near-deduplication is disabled. "
                "Falling back to exact deduplication only. Run 'pip install datasketch' to enable."
            )

    def compute_md5(self, text: str) -> str:
        """Compute MD5 hash of text (used for exact duplicate lookup)."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def compute_sha256(self, text: str) -> str:
        """Compute SHA256 hash of text (used for verification/integrity)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def exact_deduplicate(self, texts: list[str]) -> tuple[list[str], DeduplicationReport]:
        """Remove exact duplicate documents based on MD5 hashes."""
        seen = set()
        kept_texts = []
        removed = 0

        for t in texts:
            h = self.compute_md5(t)
            if h not in seen:
                seen.add(h)
                kept_texts.append(t)
            else:
                removed += 1

        total = len(texts)
        report = DeduplicationReport(
            total_input=total,
            kept=len(kept_texts),
            removed_exact=removed,
            removed_near=0,
            removal_rate=(removed / total) if total > 0 else 0.0,
            exact_hashes=seen,
            near_dedup_available=_MINHASH_AVAILABLE,
            threshold_used=self.config.threshold,
        )
        return kept_texts, report

    def _get_minhash(self, text: str) -> Any:
        """Generate MinHash object from word 3-grams of the document."""
        if not _MINHASH_AVAILABLE or MinHash is None:
            return None

        # Build word 3-grams
        words = text.lower().split()
        shingles = set()
        for i in range(len(words) - 2):
            shingles.add(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Fallback to single words if text is too short
        if not shingles:
            shingles = set(words)

        m = MinHash(num_perm=self.config.num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def near_deduplicate(self, texts: list[str]) -> tuple[list[str], DeduplicationReport]:
        """Remove near-duplicate documents using MinHash LSH."""
        if not self.config.near_enabled or not _MINHASH_AVAILABLE or MinHashLSH is None:
            # Fallback to exact
            return self.exact_deduplicate(texts)

        lsh = MinHashLSH(threshold=self.config.threshold, num_perm=self.config.num_perm)
        kept_texts = []
        removed = 0
        exact_seen = set()

        for i, t in enumerate(texts):
            h_md5 = self.compute_md5(t)
            if h_md5 in exact_seen:
                removed += 1
                continue

            m = self._get_minhash(t)
            if m is None:
                exact_seen.add(h_md5)
                kept_texts.append(t)
                continue

            # Query LSH
            result = lsh.query(m)
            if not result:
                lsh.insert(f"doc_{i}", m)
                exact_seen.add(h_md5)
                kept_texts.append(t)
            else:
                removed += 1

        total = len(texts)
        report = DeduplicationReport(
            total_input=total,
            kept=len(kept_texts),
            removed_exact=0,
            removed_near=removed,
            removal_rate=(removed / total) if total > 0 else 0.0,
            exact_hashes=exact_seen,
            near_dedup_available=True,
            threshold_used=self.config.threshold,
        )
        return kept_texts, report

    def deduplicate(self, texts: list[str]) -> tuple[list[str], DeduplicationReport]:
        """Apply exact and near deduplication in sequence."""
        if not self.config.exact_enabled and not self.config.near_enabled:
            total = len(texts)
            return texts, DeduplicationReport(total, total, 0, 0, 0.0)

        if self.config.exact_enabled and not self.config.near_enabled:
            return self.exact_deduplicate(texts)

        # Near deduplication naturally handles exact duplicates, but running both in one pass
        # is implemented inside near_deduplicate. So we run near_deduplicate directly.
        return self.near_deduplicate(texts)
