# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Pluggable quality scoring backends for Phase 8.

Supports fast rule-based, perplexity-based, and LLM-based quality judges.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class QualityBackend(ABC):
    """Abstract base class for quality scoring backends."""

    @abstractmethod
    def score(self, text: str) -> float:
        """Compute a quality score in range [0.0, 1.0]."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return backend identifier name."""
        pass


class FastQualityBackend(QualityBackend):
    """Rule-based quality scoring based on alpha characters, length, and repetition."""

    def __init__(self, min_chars: int = 50, max_chars: int = 100_000) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        n = len(text)
        if n < self.min_chars or n > self.max_chars:
            return 0.1

        # Alpha ratio
        alphas = sum(1 for c in text if c.isalpha())
        alpha_ratio = alphas / n

        # Simple repetition penalty
        words = text.split()
        unique_words = len(set(words))
        word_ratio = (unique_words / len(words)) if words else 1.0

        # Combine metrics into [0, 1] range
        base_score = 0.4 * alpha_ratio + 0.6 * word_ratio
        return min(max(base_score, 0.0), 1.0)

    def name(self) -> str:
        return "fast"


class PerplexityQualityBackend(QualityBackend):
    """Mock perplexity-based scoring using character frequency variance (surrogate)."""

    def score(self, text: str) -> float:
        # Lower perplexity = higher natural language probability
        # Here we mock it: if character distribution is close to typical English distribution,
        # it scores higher.
        if not text:
            return 0.0
        # Placeholder surrogate score
        return 0.75

    def name(self) -> str:
        return "perplexity"


class LLMJudgeBackend(QualityBackend):
    """Mock LLM judge backend returning a default high score."""

    def score(self, text: str) -> float:
        return 0.85

    def name(self) -> str:
        return "llm"


class CustomQualityBackend(QualityBackend):
    """Custom backend wrapping a callable."""

    def __init__(self, score_fn: Any) -> None:
        self.score_fn = score_fn

    def score(self, text: str) -> float:
        try:
            return float(self.score_fn(text))
        except Exception:
            return 0.5

    def name(self) -> str:
        return "custom"
