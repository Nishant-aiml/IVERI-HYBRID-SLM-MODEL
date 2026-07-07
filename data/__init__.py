# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Data loading, preprocessing, and byte-level dataset utilities."""

from __future__ import annotations

from data.dataloader import ByteDataset, StreamingByteDataset, get_dataloader
from data.dataset_utils import (
    detect_duplicates,
    find_text_files,
    generate_dataset_metadata,
    load_raw_text_file,
    stream_documents_from_files,
)
from data.preprocessing import (
    chunk_sequence,
    clean_invalid_bytes,
    get_dataset_statistics,
    normalize_whitespaces,
    pad_sequence,
    text_to_bytes,
    validate_utf8,
)

__all__ = [
    "ByteDataset",
    "StreamingByteDataset",
    "get_dataloader",
    "clean_invalid_bytes",
    "chunk_sequence",
    "get_dataset_statistics",
    "normalize_whitespaces",
    "pad_sequence",
    "text_to_bytes",
    "validate_utf8",
    "find_text_files",
    "detect_duplicates",
    "load_raw_text_file",
    "stream_documents_from_files",
    "generate_dataset_metadata",
]
