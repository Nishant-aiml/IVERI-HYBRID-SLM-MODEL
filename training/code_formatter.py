# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Code-specific byte formatter for IVERI CORE Phase 3.3 coding specialization.

Extends :class:`~training.conversation_formatter.ConversationFormatter` with
code-specific prefixes, language headers, and multi-format support:

- **pretrain** samples (``content`` field): raw code with language header
- **alpaca** samples (``instruction``/``output``): code Q&A with code prefixes
- **messages** samples (``messages`` list): chat-format with code-specific roles

Examples
--------
>>> from training.code_formatter import CodeFormatter, CodeFormatterConfig
>>> fmt = CodeFormatter()
>>> b, spans = fmt.format_to_bytes({"instruction": "Write hello world", "output": "print('hello')"})
>>> len(b) > 0
True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from training.conversation_formatter import (
    ConversationFormatter,
    FormatterConfig,
    TextSpan,
)

logger = logging.getLogger(__name__)

# ── Language detection heuristics ──────────────────────────────────────────

_LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "c++": "cpp",
    "c#": "csharp",
    "rs": "rust",
    "rb": "ruby",
    "go": "go",
    "java": "java",
    "sh": "bash",
    "shell": "bash",
}

_KNOWN_LANGUAGES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "cpp", "c", "java", "rust",
    "go", "ruby", "php", "swift", "kotlin", "scala", "haskell",
    "bash", "shell", "r", "matlab", "julia", "csharp",
})


# ── Config dataclass ───────────────────────────────────────────────────────


@dataclass
class CodeFormatterConfig:
    """Configuration for :class:`CodeFormatter`.

    Attributes
    ----------
    code_prefix:
        Prefix for code task instructions.
    solution_prefix:
        Prefix for code solution / response.
    language_prefix:
        Prefix for the injected language header.
    include_language_header:
        Inject a language identifier line before formatted code.
    include_explanation:
        Retain explanation text in samples that include it.
    max_turns:
        Maximum conversation turns (0 = unlimited).
    """

    code_prefix: str = "### Code Task:\n"
    solution_prefix: str = "### Solution:\n"
    language_prefix: str = "### Language: "
    include_language_header: bool = True
    include_explanation: bool = True
    max_turns: int = 0


# ── Main formatter class ───────────────────────────────────────────────────


class CodeFormatter:
    """Byte-level code formatter for IVERI CORE coding specialization.

    Wraps :class:`~training.conversation_formatter.ConversationFormatter`
    with code-specific prefixes and multi-format dispatch.

    Parameters
    ----------
    config:
        Formatter configuration.  Defaults to ``CodeFormatterConfig()``.
    """

    def __init__(self, config: CodeFormatterConfig | None = None) -> None:
        self.config = config or CodeFormatterConfig()

        # Build the underlying ConversationFormatter
        self._fmt_config = FormatterConfig(
            prompt_prefix=self.config.code_prefix,
            assistant_prefix=self.config.solution_prefix,
            max_turns=self.config.max_turns,
        )
        self._conv_formatter = ConversationFormatter(self._fmt_config)

    # ── Public API ─────────────────────────────────────────────────────

    def format_to_bytes(self, sample: dict[str, Any]) -> tuple[bytes, list[TextSpan]]:
        """Format a coding sample to UTF-8 bytes with span annotations.

        Auto-detects the sample format:
        - ``content`` field → pretrain format
        - ``messages`` field → chat / messages format
        - ``instruction`` / ``output`` → Alpaca format

        Parameters
        ----------
        sample:
            Raw dataset sample dict.

        Returns
        -------
        tuple[bytes, list[TextSpan]]
            UTF-8 encoded bytes and byte-level span annotations.
        """
        language = self.detect_language(sample)

        if "content" in sample and "messages" not in sample and "instruction" not in sample:
            return self.format_pretrain_sample(sample, language)
        else:
            return self.format_sft_sample(sample, language)

    def format_pretrain_sample(
        self,
        sample: dict[str, Any],
        language: str | None = None,
    ) -> tuple[bytes, list[TextSpan]]:
        """Format a pretrain-style raw code sample.

        The entire code content becomes the response (assistant) span so that
        setting ``train_on_prompt=True`` trains on all bytes.

        Parameters
        ----------
        sample:
            Must have a ``\"content\"`` key with the raw code string.
        language:
            Detected or provided programming language.

        Returns
        -------
        tuple[bytes, list[TextSpan]]
        """
        content = sample.get("content", "")
        if not content:
            return b"", []

        # Build text: optional language header + raw code
        text = ""
        if self.config.include_language_header and language:
            text += f"{self.config.language_prefix}{language}\n\n"

        text += content

        raw_bytes = text.encode("utf-8")
        # All bytes are the "assistant" span (response to train on)
        spans = [TextSpan(start=0, end=len(raw_bytes), role="assistant")]
        return raw_bytes, spans

    def format_sft_sample(
        self,
        sample: dict[str, Any],
        language: str | None = None,
    ) -> tuple[bytes, list[TextSpan]]:
        """Format an SFT-style coding sample (Alpaca or messages format).

        Prepends a language header if configured, then delegates to
        :class:`~training.conversation_formatter.ConversationFormatter`.

        Parameters
        ----------
        sample:
            Alpaca or messages format sample dict.
        language:
            Detected or provided programming language.

        Returns
        -------
        tuple[bytes, list[TextSpan]]
        """
        # Inject language header into instruction if configured
        working_sample = dict(sample)
        if self.config.include_language_header and language:
            lang_header = f"{self.config.language_prefix}{language}\n"
            if "messages" in working_sample:
                # Prepend to first user message
                msgs = list(working_sample["messages"])
                if msgs and msgs[0].get("role") == "user":
                    msgs[0] = dict(msgs[0])
                    msgs[0]["content"] = lang_header + msgs[0]["content"]
                    working_sample["messages"] = msgs
            elif "instruction" in working_sample:
                working_sample["instruction"] = lang_header + working_sample["instruction"]

        return self._conv_formatter.format_to_bytes(working_sample)

    def detect_language(self, sample: dict[str, Any]) -> str | None:
        """Detect the programming language from a sample dict.

        Checks, in order: ``language``, ``programming_language``, ``lang``,
        ``tags`` (list), ``metadata``.

        Parameters
        ----------
        sample:
            Raw dataset sample.

        Returns
        -------
        str | None
            Normalised language name or ``None`` if not detected.
        """
        for field_name in ("language", "programming_language", "lang"):
            val = sample.get(field_name)
            if isinstance(val, str) and val.strip():
                return self._normalize_language(val.strip())

        # Check tags list
        tags = sample.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                lang = self._normalize_language(str(tag))
                if lang:
                    return lang

        # Check metadata dict
        meta = sample.get("metadata") or {}
        if isinstance(meta, dict):
            val = meta.get("language") or meta.get("lang")
            if isinstance(val, str) and val.strip():
                return self._normalize_language(val.strip())

        return None

    def get_conversation_formatter(self) -> ConversationFormatter:
        """Return the underlying :class:`ConversationFormatter` instance.

        Used by :class:`~training.coding_dataset.CodingDatasetLoader`
        to pass directly into :class:`~training.sft_dataset.SFTByteDataset`.

        Returns
        -------
        ConversationFormatter
        """
        return self._conv_formatter

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _normalize_language(raw: str) -> str | None:
        """Normalise a language string to a canonical name."""
        lower = raw.lower().strip()
        if lower in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[lower]
        if lower in _KNOWN_LANGUAGES:
            return lower
        return None
