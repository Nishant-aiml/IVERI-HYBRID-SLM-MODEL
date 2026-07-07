# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte-level encoder for IVERI CORE.

NO TOKENIZER. IVERI processes raw UTF-8 bytes directly.
Text -> UTF-8 bytes -> token IDs (0-255 content + 256-258 specials) -> tensors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from core.byte_vocab import (
    strip_special_bytes,
    token_ids_to_content_bytes,
    validate_token_ids,
)
from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE, BYTE_VOCAB_SIZE, RAW_BYTE_VOCAB_SIZE
from data.preprocessing import pad_byte_ids, text_to_byte_ids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ByteEncoderStats:
    """Statistics for the ByteEncoder encoder operations."""

    total_docs_encoded: int
    total_bytes_encoded: int
    utf8_errors: int
    truncated_count: int
    padded_count: int


@dataclass(frozen=False, slots=True)
class ByteEncoderConfig:
    """Settings configuration for ByteEncoder."""

    seq_len: int = 512
    add_bos: bool = True
    add_eos: bool = True
    pad_value: int = PAD_BYTE


class ByteEncoder:
    """UTF-8 byte encoder and decoder without BPE/tokenization."""

    def __init__(self, config: ByteEncoderConfig | None = None) -> None:
        self.config = config or ByteEncoderConfig()
        self._total_docs = 0
        self._total_bytes = 0
        self._utf8_errors = 0
        self._truncated = 0
        self._padded = 0

    def encode(self, text: str) -> list[int]:
        """Encode string text to collision-free token IDs."""
        try:
            byte_ids = text_to_byte_ids(
                text, add_bos=self.config.add_bos, add_eos=self.config.add_eos
            )
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            self._utf8_errors += 1
            byte_ids = []

        self._total_docs += 1
        self._total_bytes += len(strip_special_bytes(byte_ids))
        return byte_ids

    def decode(self, byte_ids: list[int]) -> str:
        """Decode token IDs back to a Python string."""
        try:
            return token_ids_to_content_bytes(byte_ids).decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return ""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of strings."""
        return [self.encode(t) for t in texts]

    def encode_stream(self, texts: Iterator[str]) -> Iterator[list[int]]:
        """Memory-efficient streaming encoder."""
        for t in texts:
            yield self.encode(t)

    def decode_stream(self, byte_seqs: Iterator[list[int]]) -> Iterator[str]:
        """Streaming decoder."""
        for seq in byte_seqs:
            yield self.decode(seq)

    def validate(self, byte_ids: list[int]) -> bool:
        """Verify all integers are valid vocabulary token IDs."""
        try:
            validate_token_ids(byte_ids, context="ByteEncoder.validate")
            return True
        except ValueError:
            return False

    def validate_range(self, byte_ids: list[int]) -> bool:
        """Alias for validate."""
        return self.validate(byte_ids)

    def to_tensor(self, byte_ids: list[int], seq_len: int | None = None) -> torch.Tensor:
        """Convert byte IDs list to a PyTorch tensor, padding or truncating as needed."""
        target_len = seq_len or self.config.seq_len

        if len(byte_ids) > target_len:
            byte_ids = byte_ids[:target_len]
            self._truncated += 1
        elif len(byte_ids) < target_len:
            byte_ids = pad_byte_ids(byte_ids, target_len, self.config.pad_value)
            self._padded += 1

        return torch.tensor(byte_ids, dtype=torch.long)

    def encode_sft_sample(self, sample: dict[str, Any]) -> torch.Tensor:
        """Encode an SFT Q+A sample directly into a PyTorch tensor."""
        formatted_text = ""
        if "messages" in sample:
            parts = []
            for msg in sample["messages"]:
                role = msg["role"].capitalize()
                content = msg["content"]
                parts.append(f"### {role}:\n{content}")
            formatted_text = "\n\n".join(parts)
        elif "instruction" in sample and "output" in sample:
            instruction = sample["instruction"]
            inp = sample.get("input", "")
            output = sample["output"]
            if inp:
                instruction = f"{instruction}\n\nContext: {inp}"
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        byte_ids = self.encode(formatted_text)
        return self.to_tensor(byte_ids)

    def statistics(self) -> ByteEncoderStats:
        """Return the current operations statistics."""
        return ByteEncoderStats(
            total_docs_encoded=self._total_docs,
            total_bytes_encoded=self._total_bytes,
            utf8_errors=self._utf8_errors,
            truncated_count=self._truncated,
            padded_count=self._padded,
        )

    def reset_stats(self) -> None:
        """Reset the internal statistics counters."""
        self._total_docs = 0
        self._total_bytes = 0
        self._utf8_errors = 0
        self._truncated = 0
        self._padded = 0


class ValidationError(Exception):
    """Exception raised when SFT structure validation fails."""

    pass


# Re-export vocabulary bounds for pipeline tests.
MAX_TOKEN_ID = BYTE_VOCAB_SIZE - 1
MAX_RAW_BYTE_ID = RAW_BYTE_VOCAB_SIZE - 1
SPECIAL_TOKENS = (BOS_BYTE, PAD_BYTE, EOS_BYTE)
