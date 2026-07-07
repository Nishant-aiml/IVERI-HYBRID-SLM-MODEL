# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte-level preprocessing utilities for IVERI CORE.

Handles conversion of text or raw documents to byte sequences, UTF-8 validation,
whitespace normalization, sequence chunking, padding, and dataset statistics.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from core.byte_vocab import content_bytes_to_token_ids, validate_token_ids
from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE, RAW_BYTE_VOCAB_SIZE


def validate_utf8(data: bytes) -> bool:
    """Check if the given byte sequence is valid UTF-8.

    Args:
        data: Raw byte sequence.

    Returns:
        True if valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def clean_invalid_bytes(data: bytes, replacement: str = "") -> bytes:
    """Clean invalid UTF-8 bytes by replacing them with a replacement character.

    Args:
        data: Raw byte sequence.
        replacement: Character to replace invalid bytes with (default is '').

    Returns:
        Cleaned byte sequence, valid UTF-8.
    """
    if validate_utf8(data):
        return data
    # Decode with 'replace' and encode back
    decoded = data.decode("utf-8", errors="replace")
    # Replace default unicode replacement char with specified one if different
    if replacement != "":
        decoded = decoded.replace("\ufffd", replacement)
    return decoded.encode("utf-8")


def normalize_whitespaces(text: str) -> str:
    """Normalize whitespace character sequences to single spaces and strip outer padding.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    # Use unicode normalization first (NFKC)
    normalized = unicodedata.normalize("NFKC", text)
    # Split on any whitespace sequence and rejoin with a single space
    return " ".join(normalized.split())


def text_to_byte_ids(text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Convert text to collision-free token IDs.

    Content UTF-8 bytes map 1:1 to IDs ``0..255``.  BOS/PAD/EOS use extended
    IDs ``256..258`` and never overlap raw byte values.
    """
    encoded = content_bytes_to_token_ids(text.encode("utf-8"))
    result: list[int] = []
    if add_bos:
        result.append(BOS_BYTE)
    result.extend(encoded)
    if add_eos:
        result.append(EOS_BYTE)
    validate_token_ids(result, context="text_to_byte_ids")
    return result


def text_to_bytes(text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Alias for :func:`text_to_byte_ids` (historical name; returns token IDs)."""
    return text_to_byte_ids(text, add_bos=add_bos, add_eos=add_eos)


def chunk_byte_ids(byte_ids: list[int], seq_len: int, overlap: int = 0) -> list[list[int]]:
    """Chunk token ID sequences into fixed-length segments."""
    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0")
    if overlap < 0 or overlap >= seq_len:
        raise ValueError("overlap must be in [0, seq_len - 1]")

    chunks: list[list[int]] = []
    step = seq_len - overlap
    i = 0
    while i < len(byte_ids):
        chunk = byte_ids[i : i + seq_len]
        chunks.append(chunk)
        if len(chunk) < seq_len:
            break
        i += step
    return chunks


def chunk_sequence(byte_data: bytes, seq_len: int, overlap: int = 0) -> list[bytes]:
    """Chunk a long byte sequence into fixed-length segments of size `seq_len`.

    Args:
        byte_data: Input byte sequence.
        seq_len: Target length for each chunk.
        overlap: Overlap size between consecutive chunks.

    Returns:
        List of chunked byte sequences.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0")
    if overlap < 0 or overlap >= seq_len:
        raise ValueError("overlap must be in [0, seq_len - 1]")

    chunks = []
    step = seq_len - overlap
    i = 0
    while i < len(byte_data):
        chunk = byte_data[i : i + seq_len]
        chunks.append(chunk)
        # If the chunk is smaller than seq_len, we stop (it will be padded downstream)
        if len(chunk) < seq_len:
            break
        i += step
    return chunks


def pad_byte_ids(byte_ids: list[int], seq_len: int, pad_val: int = PAD_BYTE) -> list[int]:
    """Pad or truncate token IDs to the target sequence length."""
    if len(byte_ids) >= seq_len:
        return byte_ids[:seq_len]
    return byte_ids + [pad_val] * (seq_len - len(byte_ids))


def pad_sequence(byte_data: bytes, seq_len: int, pad_val: int = PAD_BYTE) -> list[int]:
    """Pad or truncate a content byte sequence using collision-free PAD token IDs."""
    return pad_byte_ids(content_bytes_to_token_ids(byte_data), seq_len, pad_val=pad_val)


def get_dataset_statistics(documents: list[str] | list[bytes]) -> dict[str, Any]:
    """Compute comprehensive statistics on a corpus of documents.

    Args:
        documents: A list of strings or pre-encoded byte sequences.

    Returns:
        Dictionary containing corpus statistics.
    """
    if not documents:
        return {
            "num_documents": 0,
            "total_bytes": 0,
            "avg_bytes_per_doc": 0.0,
            "max_bytes_per_doc": 0,
            "min_bytes_per_doc": 0,
            "valid_utf8_pct": 100.0,
        }

    byte_lengths = []
    utf8_valid_count = 0
    total_docs = len(documents)

    for doc in documents:
        if isinstance(doc, str):
            encoded = doc.encode("utf-8")
            valid = True
        else:
            encoded = doc
            valid = validate_utf8(doc)

        byte_lengths.append(len(encoded))
        if valid:
            utf8_valid_count += 1

    total_bytes = sum(byte_lengths)
    return {
        "num_documents": total_docs,
        "total_bytes": total_bytes,
        "avg_bytes_per_doc": total_bytes / total_docs,
        "max_bytes_per_doc": max(byte_lengths),
        "min_bytes_per_doc": min(byte_lengths),
        "valid_utf8_pct": (utf8_valid_count / total_docs) * 100.0,
        "raw_byte_vocab_size": RAW_BYTE_VOCAB_SIZE,
    }
