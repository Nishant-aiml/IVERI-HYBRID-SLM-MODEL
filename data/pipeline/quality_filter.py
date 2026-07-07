# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Quality filtering and text normalization for IVERI CORE data pipeline.

Filters web text corpora for training quality, removing short/long docs,
HTML garbage, excessive emoji or punctuation, and repairing unicode issues.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class QualityReport:
    """Report detailing filtering actions taken on a text dataset."""

    total_input: int
    kept: int
    removed_too_short: int
    removed_too_long: int
    removed_alpha_ratio: int
    removed_line_length: int
    removed_repetition: int
    removed_html_garbage: int
    removed_control_chars: int
    removed_broken_utf8: int
    removed_emoji: int
    removed_punct: int
    removal_rate: float


@dataclass(frozen=False, slots=True)
class QualityFilterConfig:
    """Configuration for quality filters."""

    min_doc_chars: int = 100
    max_doc_chars: int = 100_000
    min_alpha_ratio: float = 0.5
    max_avg_line_length: int = 1000
    max_rep_ratio: float = 0.2
    filter_html: bool = True
    max_html_tag_ratio: float = 0.2
    remove_control_chars: bool = True
    normalize_unicode: bool = True
    repair_broken_utf8: bool = True
    filter_excessive_emoji: bool = True
    max_emoji_ratio: float = 0.1
    filter_excessive_punctuation: bool = True
    max_punct_ratio: float = 0.3


class QualityFilter:
    """Class exposing quality checks and string processing helpers."""

    def __init__(self, config: QualityFilterConfig | None = None) -> None:
        self.config = config or QualityFilterConfig()
        # Compiled patterns
        self.html_tag_pat = re.compile(r"<[^>]*>")
        self.control_char_pat = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        self.punct_pat = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")

    def normalize_unicode_text(self, text: str) -> str:
        """Perform Unicode NFKC normalization on text."""
        return unicodedata.normalize("NFKC", text)

    def remove_control_characters(self, text: str) -> str:
        """Remove control characters from string."""
        return self.control_char_pat.sub("", text)

    def repair_utf8(self, text: str) -> str:
        """Attempt to repair broken UTF-8 byte sequences in text."""
        try:
            # Re-encode and decode with replace errors
            return text.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return text

    def min_length_filter(self, text: str) -> bool:
        """Return True if text exceeds minimum length."""
        return len(text) >= self.config.min_doc_chars

    def max_length_filter(self, text: str) -> bool:
        """Return True if text is below maximum length."""
        return len(text) <= self.config.max_doc_chars

    def alpha_ratio_filter(self, text: str) -> bool:
        """Return True if alphabet character ratio exceeds threshold."""
        if not text:
            return False
        alphas = sum(1 for c in text if c.isalpha())
        return (alphas / len(text)) >= self.config.min_alpha_ratio

    def line_length_filter(self, text: str) -> bool:
        """Return True if average line length is within bounds."""
        lines = text.splitlines()
        if not lines:
            return True
        avg = sum(len(line) for line in lines) / len(lines)
        return avg <= self.config.max_avg_line_length

    def repetition_filter(self, text: str) -> bool:
        """Return True if duplicate/repetitive content is below threshold."""
        # Simple word-level repetition check
        words = text.lower().split()
        if len(words) < 10:
            return True
        unique = len(set(words))
        rep_ratio = 1.0 - (unique / len(words))
        return rep_ratio <= self.config.max_rep_ratio

    def html_garbage_filter(self, text: str) -> bool:
        """Return True if HTML tag density is within bounds."""
        if not self.config.filter_html or not text:
            return True
        tags_len = sum(len(m.group(0)) for m in self.html_tag_pat.finditer(text))
        ratio = tags_len / len(text)
        return ratio <= self.config.max_html_tag_ratio

    def emoji_ratio_filter(self, text: str) -> bool:
        """Return True if emoji density is below threshold."""
        if not self.config.filter_excessive_emoji or not text:
            return True
        emoji_count = 0
        for char in text:
            # Check unicode category 'So' (Symbol, other) for emoji
            if unicodedata.category(char) == "So":
                emoji_count += 1
        ratio = emoji_count / len(text)
        return ratio <= self.config.max_emoji_ratio

    def punctuation_ratio_filter(self, text: str) -> bool:
        """Return True if punctuation density is below threshold."""
        if not self.config.filter_excessive_punctuation or not text:
            return True
        punct_count = len(self.punct_pat.findall(text))
        ratio = punct_count / len(text)
        return ratio <= self.config.max_punct_ratio

    def apply(self, texts: list[str]) -> list[str]:
        """Apply all quality filters and return cleaned list."""
        kept, _ = self.apply_with_report(texts)
        return kept

    def apply_with_report(self, texts: list[str]) -> tuple[list[str], QualityReport]:
        """Apply all quality filters and return report along with list."""
        kept = []
        r_short = 0
        r_long = 0
        r_alpha = 0
        r_line = 0
        r_rep = 0
        r_html = 0
        r_ctrl = 0
        r_utf8 = 0
        r_emoji = 0
        r_punct = 0

        for t in texts:
            # Normalization/Repairs first
            if self.config.repair_broken_utf8:
                t = self.repair_utf8(t)
            if self.config.normalize_unicode:
                t = self.normalize_unicode_text(t)
            if self.config.remove_control_chars:
                t = self.remove_control_characters(t)

            # Filtering checks
            if not self.min_length_filter(t):
                r_short += 1
                continue
            if not self.max_length_filter(t):
                r_long += 1
                continue
            if not self.alpha_ratio_filter(t):
                r_alpha += 1
                continue
            if not self.line_length_filter(t):
                r_line += 1
                continue
            if not self.repetition_filter(t):
                r_rep += 1
                continue
            if not self.html_garbage_filter(t):
                r_html += 1
                continue
            if not self.emoji_ratio_filter(t):
                r_emoji += 1
                continue
            if not self.punctuation_ratio_filter(t):
                r_punct += 1
                continue

            kept.append(t)

        total = len(texts)
        removed_total = total - len(kept)
        report = QualityReport(
            total_input=total,
            kept=len(kept),
            removed_too_short=r_short,
            removed_too_long=r_long,
            removed_alpha_ratio=r_alpha,
            removed_line_length=r_line,
            removed_repetition=r_rep,
            removed_html_garbage=r_html,
            removed_control_chars=r_ctrl,
            removed_broken_utf8=r_utf8,
            removed_emoji=r_emoji,
            removed_punct=r_punct,
            removal_rate=(removed_total / total) if total > 0 else 0.0,
        )

        return kept, report
