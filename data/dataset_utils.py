# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset utilities for IVERI CORE.

Provides file system helpers, corpus indexing, duplicate detection, and streaming generators.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Generator
from typing import Any


def find_text_files(
    directory: str | pathlib.Path, extensions: list[str] | None = None
) -> list[pathlib.Path]:
    """Find all text or data files in a directory recursively.

    Args:
        directory: Path to the search directory.
        extensions: List of extensions to filter by (default: ['.txt', '.json', '.jsonl']).

    Returns:
        List of pathlib.Path file locations.
    """
    path = pathlib.Path(directory)
    if not path.exists() or not path.is_dir():
        return []

    if extensions is None:
        extensions = [".txt", ".json", ".jsonl"]

    files: list[pathlib.Path] = []
    for ext in extensions:
        # Support case-insensitive extension matching
        pattern = f"**/*{ext}"
        files.extend(path.rglob(pattern))
        pattern_upper = f"**/*{ext.upper()}"
        files.extend(path.rglob(pattern_upper))

    # De-duplicate list and sort
    return sorted(list(set(files)))


def detect_duplicates(documents: list[str] | list[bytes]) -> list[int]:
    """Identify indices of duplicate documents in a list based on MD5 hashes.

    Args:
        documents: List of string documents or byte sequences.

    Returns:
        List of indices that represent duplicate documents (retaining first occurrence).
    """
    seen_hashes = set()
    duplicate_indices = []

    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            encoded = doc.encode("utf-8")
        else:
            assert isinstance(doc, bytes)
            encoded = doc

        h = hashlib.md5(encoded).hexdigest()
        if h in seen_hashes:
            duplicate_indices.append(i)
        else:
            seen_hashes.add(h)

    return duplicate_indices


def load_raw_text_file(file_path: str | pathlib.Path) -> list[str]:
    """Load text documents from a file.

    Supports:
    - `.jsonl`: assumes each line is a JSON object with a "text" or "content" field.
    - `.json`: parses as list of strings or dict of documents.
    - Plain text files: returns the whole file as a single string document (or splits on double newlines).

    Args:
        file_path: Path to the target file.

    Returns:
        List of raw text documents extracted from the file.
    """
    path = pathlib.Path(file_path)
    if not path.exists() or not path.is_file():
        return []

    documents = []
    ext = path.suffix.lower()

    if ext == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # Try common fields
                        text = obj.get("text", obj.get("content", obj.get("body", "")))
                        if text:
                            documents.append(text)
                    elif isinstance(obj, str):
                        documents.append(obj)
                except json.JSONDecodeError:
                    # Fallback to plain line
                    documents.append(line)
    elif ext == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            documents.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text", item.get("content", ""))
                            if text:
                                documents.append(text)
                elif isinstance(data, dict):
                    # Try direct text or fields
                    text = data.get("text", data.get("content", ""))
                    if text:
                        documents.append(text)
            except json.JSONDecodeError:
                pass
    else:
        # Default plain text: read whole file
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
            if content:
                # If double newlines exist, treat as paragraph-level documents
                if "\n\n" in content:
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    documents.extend(paragraphs)
                else:
                    documents.append(content)

    return [d for d in documents if d]  # filter out empty documents


def stream_documents_from_files(
    file_paths: list[pathlib.Path],
) -> Generator[str, None, None]:
    """A streaming generator yielding documents from a list of files sequentially.

    Prevents loading the entire corpus into memory at once.

    Args:
        file_paths: List of file locations.

    Yields:
        Raw string documents.
    """
    for path in file_paths:
        docs = load_raw_text_file(path)
        yield from docs


def generate_dataset_metadata(
    directory: str | pathlib.Path,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Generate metadata structure describing the dataset corpus.

    Args:
        directory: Root directory path.
        stats: Computed dataset statistics dictionary.

    Returns:
        Metadata dictionary.
    """
    path = pathlib.Path(directory)
    return {
        "dataset_name": path.name if path.exists() else "unknown",
        "root_directory": str(path.resolve()),
        "statistics": stats,
        "format_version": "1.0",
    }
