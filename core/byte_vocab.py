# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte vocabulary helpers for collision-free special tokens (Phase 6.3.2 OBJ7).

Raw UTF-8 content bytes map 1:1 to token IDs ``0..255``.  Structural specials
(BOS, PAD, EOS) live in extended IDs ``256..258`` and never overlap valid content.
"""

from __future__ import annotations

from core.constants import (
    BOS_BYTE,
    BYTE_VOCAB_SIZE,
    EOS_BYTE,
    LEGACY_BOS_BYTE,
    LEGACY_EOS_BYTE,
    LEGACY_PAD_BYTE,
    LEGACY_SPECIAL_BYTE_IDS,
    PAD_BYTE,
    RAW_BYTE_VOCAB_SIZE,
    SPECIAL_BYTE_IDS,
)


class ByteVocabularyError(ValueError):
    """Raised when byte/token encoding violates the vocabulary contract."""


def is_raw_byte_id(token_id: int) -> bool:
    """Return True when *token_id* is a content byte (0–255)."""
    return 0 <= token_id < RAW_BYTE_VOCAB_SIZE


def is_special_byte_id(token_id: int) -> bool:
    """Return True when *token_id* is a structural special (BOS/PAD/EOS)."""
    return token_id in SPECIAL_BYTE_IDS


def is_valid_token_id(token_id: int) -> bool:
    """Return True when *token_id* is in the model vocabulary."""
    return 0 <= token_id < BYTE_VOCAB_SIZE


def validate_token_ids(token_ids: list[int], *, context: str = "") -> None:
    """Fail closed when any ID is outside ``[0, BYTE_VOCAB_SIZE)``."""
    prefix = f"{context}: " if context else ""
    for idx, token_id in enumerate(token_ids):
        if not is_valid_token_id(token_id):
            raise ByteVocabularyError(
                f"{prefix}invalid token id {token_id} at position {idx} "
                f"(expected 0..{BYTE_VOCAB_SIZE - 1})"
            )


def validate_no_legacy_special_bytes(token_ids: list[int], *, context: str = "") -> None:
    """Reject sequences that use pre-v0.2.0 structural IDs (0, 1, 2).

    Intended for legacy dataset migration checks — not for raw UTF-8 content bytes.
    """
    prefix = f"{context}: " if context else ""
    for idx, token_id in enumerate(token_ids):
        if token_id in LEGACY_SPECIAL_BYTE_IDS:
            raise ByteVocabularyError(
                f"{prefix}legacy special byte {token_id} at position {idx}; "
                f"use BOS={BOS_BYTE}, PAD={PAD_BYTE}, EOS={EOS_BYTE}"
            )


def strip_special_bytes(token_ids: list[int]) -> list[int]:
    """Remove structural specials, keeping raw content bytes only."""
    return [t for t in token_ids if t not in SPECIAL_BYTE_IDS]


def remap_legacy_token_ids(token_ids: list[int]) -> list[int]:
    """Map pre-v0.2.0 legacy specials (0,1,2) to extended collision-free IDs."""
    legacy_map = {
        LEGACY_PAD_BYTE: PAD_BYTE,
        LEGACY_BOS_BYTE: BOS_BYTE,
        LEGACY_EOS_BYTE: EOS_BYTE,
    }
    return [legacy_map.get(t, t) for t in token_ids]


def content_bytes_to_token_ids(data: bytes) -> list[int]:
    """Map UTF-8 content bytes 1:1 to token IDs 0–255."""
    return list(data)


def token_ids_to_content_bytes(token_ids: list[int]) -> bytes:
    """Decode content token IDs back to bytes (specials stripped)."""
    cleaned = strip_special_bytes(token_ids)
    for token_id in cleaned:
        if not is_raw_byte_id(token_id):
            raise ByteVocabularyError(
                f"non-content token id {token_id} cannot be decoded to bytes"
            )
    return bytes(cleaned)
