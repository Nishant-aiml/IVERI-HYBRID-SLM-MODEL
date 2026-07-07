# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""PII and credential scrubbing for IVERI CORE data pipeline.

Identifies and removes personally identifiable information (emails, phones, Aadhaar,
PAN cards, credit cards, IP addresses) and leaked secrets (GitHub tokens, AWS
access/secret keys, OpenAI secret keys, JWT tokens, RSA private keys, Bearer tokens).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class PIIReport:
    """Report summarizing PII scrubbing operations."""

    total_docs: int
    docs_with_pii: int
    pii_rate: float
    match_counts: dict[str, int] = field(default_factory=dict)
    total_replacements: int = 0


class PIIRemover:
    """Scrub PII and leaked credentials using regex patterns."""

    # All target regex patterns
    PII_PATTERNS: dict[str, str] = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_india": r"\b[6-9]\d{9}\b",
        "phone_us": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "github_token": r"\bghp_[A-Za-z0-9]{36}\b",
        "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
        "aws_secret_key": r"(?i)aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]",
        "openai_key": r"\bsk-[A-Za-z0-9]{48}\b",
        "bearer_token": r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*",
        "rsa_private_key": r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        "jwt_token": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    }

    def __init__(
        self,
        replacement: str = "[REDACTED]",
        enabled_patterns: set[str] | None = None,
    ) -> None:
        self.replacement = replacement
        self.enabled = enabled_patterns or set(self.PII_PATTERNS.keys())

        # Compile enabled regex patterns
        self.patterns: dict[str, re.Pattern] = {}
        for key in self.enabled:
            if key in self.PII_PATTERNS:
                self.patterns[key] = re.compile(self.PII_PATTERNS[key])

    def remove(self, text: str) -> str:
        """Replace all identified PII with the replacement string."""
        cleaned = text
        for pat in self.patterns.values():
            cleaned = pat.sub(self.replacement, cleaned)
        return cleaned

    def has_pii(self, text: str) -> bool:
        """Return True if any enabled pattern matches within the text."""
        return any(pat.search(text) for pat in self.patterns.values())

    def remove_batch(self, texts: list[str]) -> list[str]:
        """Replace PII in a batch of texts."""
        return [self.remove(t) for t in texts]

    def audit(self, texts: list[str]) -> PIIReport:
        """Scan documents and collect counts of each PII type matched."""
        total = len(texts)
        with_pii = 0
        counts: dict[str, int] = {k: 0 for k in self.patterns}
        total_replacements = 0

        for t in texts:
            doc_had_pii = False
            for k, pat in self.patterns.items():
                matches = pat.findall(t)
                if matches:
                    counts[k] += len(matches)
                    total_replacements += len(matches)
                    doc_had_pii = True
            if doc_had_pii:
                with_pii += 1

        report = PIIReport(
            total_docs=total,
            docs_with_pii=with_pii,
            pii_rate=(with_pii / total) if total > 0 else 0.0,
            match_counts=counts,
            total_replacements=total_replacements,
        )

        return report

    def get_enabled_patterns(self) -> dict[str, str]:
        """Return dict of currently active pattern names and regexes."""
        return {k: self.PII_PATTERNS[k] for k in self.enabled}
