# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Conversation formatter for byte-level SFT instruction tuning.

Converts structured instruction/conversation data into flat UTF-8 byte sequences
for next-byte prediction training. Supports Alpaca single-turn, Chat (user/assistant),
and Multi-turn conversation formats with configurable delimiters.

Format contract
---------------
All formatted strings are UTF-8 encoded.  Special bytes (e.g. padding) are
applied at the dataset level, not here.

Examples
--------
>>> fmt = ConversationFormatter()
>>> text = fmt.format_alpaca({"instruction": "Hello", "output": "Hi"})
>>> assert "### Instruction:" in text
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Default template tokens ────────────────────────────────────────────────

_DEFAULT_SYSTEM_PREFIX: str = "### System:\n"
_DEFAULT_PROMPT_PREFIX: str = "### Instruction:\n"
_DEFAULT_INPUT_PREFIX: str = "### Input:\n"
_DEFAULT_ASSISTANT_PREFIX: str = "### Response:\n"
_DEFAULT_USER_PREFIX: str = "### User:\n"
_DEFAULT_TURN_SEP: str = "\n\n"
_DEFAULT_EOS: str = "\n\n"


# ── Template configuration ─────────────────────────────────────────────────


@dataclass
class FormatterConfig:
    """Configuration for ConversationFormatter delimiters and template tokens.

    All strings are injected verbatim into the byte sequence.

    Attributes
    ----------
    system_prefix:
        Prefix inserted before system prompt content.
    prompt_prefix:
        Prefix inserted before user instruction/query.
    input_prefix:
        Prefix inserted before the optional ``input`` field (Alpaca only).
    assistant_prefix:
        Prefix inserted before assistant response.
    user_prefix:
        Prefix used in multi-turn chat mode before user turns.
    turn_separator:
        String inserted between consecutive conversation turns.
    eos_token:
        String appended at the very end of each formatted sample.
    add_eos:
        Whether to append ``eos_token`` at sample end.
    max_turns:
        Maximum number of dialogue turns retained. 0 = unlimited.
    """

    system_prefix: str = _DEFAULT_SYSTEM_PREFIX
    prompt_prefix: str = _DEFAULT_PROMPT_PREFIX
    input_prefix: str = _DEFAULT_INPUT_PREFIX
    assistant_prefix: str = _DEFAULT_ASSISTANT_PREFIX
    user_prefix: str = _DEFAULT_USER_PREFIX
    turn_separator: str = _DEFAULT_TURN_SEP
    eos_token: str = _DEFAULT_EOS
    add_eos: bool = True
    max_turns: int = 0


# ── Span tracking ─────────────────────────────────────────────────────────


@dataclass
class TextSpan:
    """A labelled byte range inside a formatted sequence.

    Attributes
    ----------
    start:
        Start byte index (inclusive).
    end:
        End byte index (exclusive).
    role:
        One of ``"system"``, ``"user"``, ``"assistant"``, ``"prefix"``.
    """

    start: int
    end: int
    role: str


@dataclass
class FormattedSample:
    """Result of formatting a single sample.

    Attributes
    ----------
    text:
        The complete formatted UTF-8 string.
    spans:
        Ordered list of :class:`TextSpan` instances identifying which byte
        ranges belong to which role.  Used by :class:`LossMaskBuilder`.
    format_type:
        One of ``"alpaca"``, ``"chat"``, ``"multi_turn"``.
    num_turns:
        Number of dialogue turns (1 for Alpaca).
    """

    text: str
    spans: list[TextSpan] = field(default_factory=list)
    format_type: str = "alpaca"
    num_turns: int = 1


# ── Main formatter class ───────────────────────────────────────────────────


class ConversationFormatter:
    """Convert structured instruction data into flat UTF-8 byte sequences.

    Supports three input schemas:

    Alpaca
        ``{"instruction": "...", "input": "optional", "output": "..."}``

    Chat / messages (single-turn or multi-turn)
        ``{"messages": [{"role": "user", "content": "..."}, ...]}``

    Parameters
    ----------
    config:
        :class:`FormatterConfig` controlling delimiters and template tokens.
        Defaults are the standard IVERI SFT template.
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()

    # ── Public API ─────────────────────────────────────────────────────

    def format_sample(self, sample: dict[str, Any]) -> FormattedSample:
        """Auto-detect schema and format sample.

        Parameters
        ----------
        sample:
            Raw dataset row.

        Returns
        -------
        FormattedSample
            Formatted text and span metadata.

        Raises
        ------
        ValueError
            If the schema is neither Alpaca nor messages format.
        """
        if "messages" in sample:
            return self.format_messages(sample["messages"])
        if "instruction" in sample:
            return self.format_alpaca(sample)
        # Legacy conversations format (some datasets use this key)
        if "conversations" in sample:
            msgs = _normalize_conversations(sample["conversations"])
            return self.format_messages(msgs)
        raise ValueError(
            "Unsupported sample schema: expected 'messages', 'conversations', "
            "or 'instruction' key. Got keys: " + str(list(sample.keys()))
        )

    def format_alpaca(self, sample: dict[str, Any]) -> FormattedSample:
        """Format an Alpaca-style single-turn sample.

        Expected keys: ``instruction`` (required), ``input`` (optional),
        ``output`` (required).

        Parameters
        ----------
        sample:
            Alpaca-format dict.

        Returns
        -------
        FormattedSample
            Text + spans.
        """
        cfg = self.config
        instruction = str(sample.get("instruction", "")).strip()
        inp = str(sample.get("input", "")).strip()
        output = str(sample.get("output", "")).strip()

        parts: list[tuple[str, str]] = []  # (role, text_chunk)

        # ── Prompt part ────────────────────────────────────────────────
        prompt_text = cfg.prompt_prefix + instruction
        if inp:
            prompt_text += cfg.turn_separator + cfg.input_prefix + inp
        prompt_text += cfg.turn_separator
        parts.append(("user", prompt_text))

        # ── Response part ──────────────────────────────────────────────
        response_text = cfg.assistant_prefix + output
        if cfg.add_eos:
            response_text += cfg.eos_token
        parts.append(("assistant", response_text))

        return _build_formatted_sample(parts, format_type="alpaca", num_turns=1)

    def format_messages(self, messages: list[dict[str, str]]) -> FormattedSample:
        """Format a messages list (chat or multi-turn).

        Parameters
        ----------
        messages:
            List of dicts with ``role`` and ``content`` keys.

        Returns
        -------
        FormattedSample
            Text + spans for every turn.
        """
        cfg = self.config
        msgs = list(messages)

        # Truncate if max_turns is set
        if cfg.max_turns > 0:
            msgs = _trim_to_max_turns(msgs, cfg.max_turns)

        if not msgs:
            return FormattedSample(text="", spans=[], format_type="chat", num_turns=0)

        parts: list[tuple[str, str]] = []
        num_turns = 0

        for i, msg in enumerate(msgs):
            role = str(msg.get("role", "")).lower().strip()
            content = str(msg.get("content", "")).strip()
            is_last = i == len(msgs) - 1

            if role == "system":
                chunk = cfg.system_prefix + content + cfg.turn_separator
                parts.append(("system", chunk))
            elif role == "user":
                chunk = cfg.user_prefix + content + cfg.turn_separator
                parts.append(("user", chunk))
                num_turns += 1
            elif role == "assistant":
                chunk = cfg.assistant_prefix + content
                if is_last and cfg.add_eos:
                    chunk += cfg.eos_token
                else:
                    chunk += cfg.turn_separator
                parts.append(("assistant", chunk))
            else:
                logger.debug("Skipping unknown role '%s' in messages.", role)
                continue

        fmt = "multi_turn" if num_turns > 1 else "chat"
        return _build_formatted_sample(parts, format_type=fmt, num_turns=num_turns)

    def format_to_bytes(self, sample: dict[str, Any]) -> tuple[bytes, list[TextSpan]]:
        """Format a sample directly to UTF-8 bytes plus span metadata.

        Parameters
        ----------
        sample:
            Raw dataset row.

        Returns
        -------
        tuple[bytes, list[TextSpan]]
            Encoded byte string and span list with byte-level indices.
        """
        formatted = self.format_sample(sample)
        raw = formatted.text.encode("utf-8", errors="replace")
        return raw, formatted.spans

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config_dict(cls, cfg_dict: dict[str, Any]) -> ConversationFormatter:
        """Construct from a plain dict (e.g. from :class:`InstructionConfig`).

        Parameters
        ----------
        cfg_dict:
            Dict of :class:`FormatterConfig` field names → values.

        Returns
        -------
        ConversationFormatter
        """
        known = {f.name for f in FormatterConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in cfg_dict.items() if k in known}
        return cls(FormatterConfig(**filtered))


# ── Private helpers ────────────────────────────────────────────────────────


def _build_formatted_sample(
    parts: list[tuple[str, str]],
    format_type: str,
    num_turns: int,
) -> FormattedSample:
    """Concatenate role-tagged text parts and build byte-level spans."""
    spans: list[TextSpan] = []
    cursor = 0
    full_text = ""

    for role, chunk in parts:
        byte_len = len(chunk.encode("utf-8", errors="replace"))
        spans.append(TextSpan(start=cursor, end=cursor + byte_len, role=role))
        cursor += byte_len
        full_text += chunk

    return FormattedSample(
        text=full_text,
        spans=spans,
        format_type=format_type,
        num_turns=num_turns,
    )


def _trim_to_max_turns(msgs: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    """Trim messages list to at most *max_turns* user/assistant pairs."""
    result: list[dict[str, str]] = []
    # Always keep system message at head if present
    start = 0
    if msgs and msgs[0].get("role", "").lower() == "system":
        result.append(msgs[0])
        start = 1

    pairs_seen = 0
    i = start
    while i < len(msgs) and pairs_seen < max_turns:
        msg = msgs[i]
        role = msg.get("role", "").lower()
        result.append(msg)
        if role == "assistant":
            pairs_seen += 1
        i += 1

    return result


def _normalize_conversations(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize OpenHermes-style 'conversations' key to 'messages' format.

    OpenHermes uses ``{"from": "human", "value": "..."}`` rather than
    ``{"role": "user", "content": "..."}``.
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    result = []
    for turn in conversations:
        raw_role = str(turn.get("from", turn.get("role", ""))).lower()
        content = str(turn.get("value", turn.get("content", ""))).strip()
        role = role_map.get(raw_role, raw_role)
        result.append({"role": role, "content": content})
    return result
