# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Preference formatter for Phase 3.4 Preference Optimization.

Formats chosen and rejected response pairs with identical prompt segments into
tensors for byte-level preference learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from training.conversation_formatter import ConversationFormatter, FormatterConfig, TextSpan

logger = logging.getLogger(__name__)


@dataclass
class FormattedPreferencePair:
    """Formatted bytes and spans for chosen and rejected preferences.

    Attributes
    ----------
    prompt_bytes:
        The common prompt segment in UTF-8 bytes.
    chosen_bytes:
        The chosen response segment in UTF-8 bytes (including EOS if configured).
    rejected_bytes:
        The rejected response segment in UTF-8 bytes.
    """

    prompt_bytes: bytes
    chosen_bytes: bytes
    rejected_bytes: bytes


class PreferenceFormatter:
    """Formatter to prepare chosen/rejected byte pairs for alignment training.

    Parameters
    ----------
    config:
        Optional :class:`~training.conversation_formatter.FormatterConfig` dict/object.
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()
        self.formatter = ConversationFormatter(self.config)

    def format_pair(self, sample: dict[str, Any]) -> FormattedPreferencePair:
        """Format a single preference sample into chosen and rejected byte sequences.

        Normalizes keys and parses both Alpaca-style strings and list-of-messages formats.

        Parameters
        ----------
        sample:
            Dictionary containing chosen/rejected pairs and prompt context.

        Returns
        -------
        FormattedPreferencePair

        Raises
        ------
        ValueError
            If required preference keys are missing or malformed.
        """
        # Normalize prompt key
        prompt = sample.get("prompt", sample.get("instruction", ""))
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        # ── Case A: Messages List format (UltraFeedback style) ─────────────────
        if isinstance(chosen, list) and isinstance(rejected, list):
            if not chosen or not rejected:
                raise ValueError("Empty chosen or rejected message list.")

            # Prompt is chosen messages excluding the final assistant turn
            # Chosen response is the content of the final message
            prompt_messages = chosen[:-1]
            chosen_response = chosen[-1].get("content", "")
            
            # For rejected, check if prompt matches chosen prompt
            if rejected[:-1] != prompt_messages and len(prompt_messages) > 0:
                logger.debug("Prompt mismatch between chosen and rejected messages. Using chosen prompt.")
            rejected_response = rejected[-1].get("content", "")

            # Format the prompt messages list
            formatted_prompt = self.formatter.format_messages(prompt_messages)
            # Standard SFT appends turn separator + assistant prefix at the end of prompt
            prompt_text = formatted_prompt.text + self.config.assistant_prefix
            prompt_bytes = prompt_text.encode("utf-8", errors="replace")

            # Responses
            chosen_text = chosen_response + (self.config.eos_token if self.config.add_eos else "")
            rejected_text = rejected_response + (self.config.eos_token if self.config.add_eos else "")

            return FormattedPreferencePair(
                prompt_bytes=prompt_bytes,
                chosen_bytes=chosen_text.encode("utf-8", errors="replace"),
                rejected_bytes=rejected_text.encode("utf-8", errors="replace"),
            )

        # ── Case B: String format (Alpaca style) ──────────────────────────────
        elif isinstance(chosen, str) and isinstance(rejected, str):
            instruction = str(prompt).strip()
            inp = str(sample.get("input", "")).strip()

            # Construct prompt exactly matching SFT layout
            prompt_text = self.config.prompt_prefix + instruction
            if inp:
                prompt_text += self.config.turn_separator + self.config.input_prefix + inp
            prompt_text += self.config.turn_separator + self.config.assistant_prefix
            prompt_bytes = prompt_text.encode("utf-8", errors="replace")

            # Responses
            chosen_text = str(chosen).strip() + (self.config.eos_token if self.config.add_eos else "")
            rejected_text = str(rejected).strip() + (self.config.eos_token if self.config.add_eos else "")

            return FormattedPreferencePair(
                prompt_bytes=prompt_bytes,
                chosen_bytes=chosen_text.encode("utf-8", errors="replace"),
                rejected_bytes=rejected_text.encode("utf-8", errors="replace"),
            )

        else:
            raise ValueError(
                f"Unsupported schema type: chosen is {type(chosen)}, rejected is {type(rejected)}."
            )
