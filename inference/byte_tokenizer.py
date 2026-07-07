# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte-level tokenizer wrapper for IVERI CORE inference."""

from __future__ import annotations

from core.byte_vocab import content_bytes_to_token_ids, token_ids_to_content_bytes
from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE, RAW_BYTE_VOCAB_SIZE


class ByteTokenizer:
    """Encode UTF-8 text to token IDs and decode generated IDs back to text."""

    def __init__(self, *, add_bos: bool = False) -> None:
        self.add_bos = add_bos

    def encode(self, text: str) -> list[int]:
        ids = content_bytes_to_token_ids(text.encode("utf-8"))
        if self.add_bos:
            return [BOS_BYTE] + ids
        return ids

    def decode(self, token_ids: list[int], *, errors: str = "replace") -> str:
        return token_ids_to_content_bytes(token_ids).decode("utf-8", errors=errors)

    @property
    def vocab_size(self) -> int:
        return RAW_BYTE_VOCAB_SIZE

    @property
    def eos_id(self) -> int:
        return EOS_BYTE

    @property
    def pad_id(self) -> int:
        return PAD_BYTE

    @property
    def bos_id(self) -> int:
        return BOS_BYTE
