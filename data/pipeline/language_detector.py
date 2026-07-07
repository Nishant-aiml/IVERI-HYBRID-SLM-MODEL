# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Language identification and filtering for IVERI CORE data pipeline.

Identifies document language using langdetect (optional).
Supports allow-lists and reject-lists for flexible filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

# Optional langdetect import
try:
    from langdetect import DetectorFactory
    from langdetect import detect as _langdetect_detect

    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class LanguageDetectionReport:
    """Report summarizing language detection and filtering results."""

    total_docs: int
    detected_counts: dict[str, int] = field(default_factory=dict)
    unknown_count: int = 0
    filtered_count: int = 0
    kept_count: int = 0
    detector_available: bool = _LANGDETECT_AVAILABLE


ALLOWED_LANGUAGES_STAGE1: frozenset[str] = frozenset({"en"})
ALLOWED_LANGUAGES_STAGE3B: frozenset[str] = frozenset({"en", "hi"})


class LanguageDetector:
    """Identifies text languages and filters documents based on lists."""

    def __init__(
        self,
        allowed: set[str] | None = None,
        reject: set[str] | None = None,
        short_text_min_chars: int = 20,
    ) -> None:
        self.allowed = allowed if allowed is not None else {"en"}
        self.reject = reject if reject is not None else set()
        self.short_text_min_chars = short_text_min_chars

        if not _LANGDETECT_AVAILABLE:
            logger.warning(
                "langdetect package not installed. Language detection is disabled. "
                "All documents will be treated as 'unknown'. Run 'pip install langdetect' to enable."
            )

    def detect(self, text: str) -> str:
        """Detect the language code (ISO 639-1) of a string."""
        if len(text.strip()) < self.short_text_min_chars:
            return "unknown"

        if not _LANGDETECT_AVAILABLE:
            return "unknown"

        try:
            return cast(str, _langdetect_detect(text))
        except Exception:
            return "unknown"

    def detect_batch(self, texts: list[str]) -> list[str]:
        """Detect language for a batch of texts."""
        return [self.detect(t) for t in texts]

    def is_allowed(self, lang: str) -> bool:
        """Check if a language code matches the allowed/rejected filters."""
        if lang == "unknown":
            # If detector is not available or text too short, let it pass
            # to avoid rejecting valid data when offline/fallback is active.
            return True

        if self.reject and lang in self.reject:
            return False

        return not (self.allowed and lang not in self.allowed)

    def filter(self, texts: list[str]) -> tuple[list[str], LanguageDetectionReport]:
        """Filter out texts that do not match the target language list."""
        kept = []
        counts: dict[str, int] = {}
        unknown = 0
        filtered = 0

        for t in texts:
            lang = self.detect(t)

            if lang == "unknown":
                unknown += 1
            else:
                counts[lang] = counts.get(lang, 0) + 1

            if self.is_allowed(lang):
                kept.append(t)
            else:
                filtered += 1

        total = len(texts)
        report = LanguageDetectionReport(
            total_docs=total,
            detected_counts=counts,
            unknown_count=unknown,
            filtered_count=filtered,
            kept_count=len(kept),
            detector_available=_LANGDETECT_AVAILABLE,
        )

        return kept, report

    def set_allowed(self, languages: set[str]) -> None:
        """Set the set of allowed language codes."""
        self.allowed = languages

    def set_reject(self, languages: set[str]) -> None:
        """Set the set of explicitly rejected language codes."""
        self.reject = languages
