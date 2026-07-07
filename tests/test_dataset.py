# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, integration, stress, and performance tests for IVERI CORE Data Pipeline (Phase 2.1)."""

from __future__ import annotations

import pathlib
import tempfile
import time

import pytest
import torch

from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE
from data.dataloader import ByteDataset, StreamingByteDataset, get_dataloader
from data.dataset_utils import (
    detect_duplicates,
    find_text_files,
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


def get_test_devices() -> list[str]:
    """Get list of available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


# --- Preprocessing Tests ---


def test_validate_utf8_valid() -> None:
    """Verify validate_utf8 works correctly on valid UTF-8."""
    assert validate_utf8(b"Hello, world!") is True
    assert validate_utf8("यह परीक्षण है।".encode()) is True
    assert validate_utf8("测试句子".encode()) is True


def test_validate_utf8_invalid() -> None:
    """Verify validate_utf8 flags invalid UTF-8 bytes."""
    assert validate_utf8(bytes([0xFF, 0xFE, 0xFD])) is False


def test_clean_invalid_bytes() -> None:
    """Verify clean_invalid_bytes replaces bad bytes correctly."""
    bad_bytes = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xFF, 0x21])
    cleaned = clean_invalid_bytes(bad_bytes, replacement="?")
    assert validate_utf8(cleaned) is True
    assert b"Hello" in cleaned
    assert b"!" in cleaned


def test_normalize_whitespaces() -> None:
    """Verify normalization of whitespace character sequences."""
    text = "   Hello    world!\nThis \t is   a test.   "
    expected = "Hello world! This is a test."
    assert normalize_whitespaces(text) == expected


def test_text_to_bytes() -> None:
    """Verify conversion of string to collision-free token IDs with BOS/EOS."""
    text = "Hello"
    res = text_to_bytes(text, add_bos=True, add_eos=True)
    assert res[0] == BOS_BYTE
    assert res[-1] == EOS_BYTE
    assert res[1:-1] == list(b"Hello")


def test_chunk_sequence() -> None:
    """Verify chunking sequence into fixed-length segments."""
    data = b"abcdefgh"
    # test size 3, overlap 0
    chunks = chunk_sequence(data, seq_len=3, overlap=0)
    assert chunks == [b"abc", b"def", b"gh"]

    # test size 3, overlap 1
    chunks = chunk_sequence(data, seq_len=3, overlap=1)
    assert chunks == [b"abc", b"cde", b"efg", b"gh"]


def test_pad_sequence() -> None:
    """Verify padding uses collision-free PAD token IDs."""
    data = b"abc"
    padded = pad_sequence(data, seq_len=5, pad_val=PAD_BYTE)
    assert padded == [97, 98, 99, PAD_BYTE, PAD_BYTE]

    # truncates if longer
    padded_short = pad_sequence(data, seq_len=2)
    assert padded_short == [97, 98]


def test_dataset_statistics() -> None:
    """Verify computation of corpus statistics."""
    docs = ["Hello", "यह", "测试"]
    stats = get_dataset_statistics(docs)
    assert stats["num_documents"] == 3
    assert stats["total_bytes"] == 5 + 6 + 6  # "Hello"=5, "यह"=6, "测试"=6
    assert stats["max_bytes_per_doc"] == 6
    assert stats["min_bytes_per_doc"] == 5
    assert stats["valid_utf8_pct"] == 100.0


# --- Dataset Utilities Tests ---


def test_detect_duplicates() -> None:
    """Verify MD5-based duplicate detection."""
    docs = ["Doc A", "Doc B", "Doc A", "Doc C", "Doc B"]
    dups = detect_duplicates(docs)
    assert dups == [2, 4]  # indices of duplicate items


def test_find_text_files_and_loaders() -> None:
    """Verify finding text files and loading documents from files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        # Create text file
        txt_path = root / "doc1.txt"
        txt_path.write_text("Hello World\n\nSecond Document", encoding="utf-8")

        # Create jsonl file
        jsonl_path = root / "doc2.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write('{"text": "JSONL document 1"}\n')
            f.write('{"content": "JSONL document 2"}\n')

        files = find_text_files(root, extensions=[".txt", ".jsonl"])
        assert len(files) == 2

        # Test loading plain text
        txt_docs = load_raw_text_file(txt_path)
        assert len(txt_docs) == 2
        assert txt_docs[0] == "Hello World"
        assert txt_docs[1] == "Second Document"

        # Test loading jsonl
        jsonl_docs = load_raw_text_file(jsonl_path)
        assert len(jsonl_docs) == 2
        assert jsonl_docs[0] == "JSONL document 1"
        assert jsonl_docs[1] == "JSONL document 2"

        # Test stream
        streamed = list(stream_documents_from_files(files))
        assert len(streamed) == 4


# --- DataLoader Tests ---


def test_byte_dataset_shapes_and_types() -> None:
    """Verify map-style ByteDataset output shape and type contracts."""
    docs = ["Hello, this is a test document.", "Short doc."]
    seq_len = 8
    dataset = ByteDataset(docs, seq_len=seq_len, add_bos=True, add_eos=True)

    # seq_len=8 -> inputs must be shape (8,), targets shape (8,)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # verify targets are shifted inputs
    # chunk must be inputs + last target byte
    chunk_bytes = dataset.chunks[0]
    assert torch.equal(x, torch.tensor(list(chunk_bytes[:-1]), dtype=torch.long))
    assert torch.equal(y, torch.tensor(list(chunk_bytes[1:]), dtype=torch.long))


def test_streaming_byte_dataset() -> None:
    """Verify iterable-style StreamingByteDataset outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        path = root / "data.txt"
        path.write_text("Hello world! This is a long streaming test doc.", encoding="utf-8")

        seq_len = 16
        dataset = StreamingByteDataset([path], seq_len=seq_len)
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)

        for x, y in loader:
            assert x.shape[0] <= 2
            assert x.shape[1] == seq_len
            assert y.shape[1] == seq_len
            assert x.dtype == torch.long
            assert y.dtype == torch.long
            break


def test_dataloader_seed_determinism() -> None:
    """Verify that dataloader yields deterministic batches under seed control."""
    docs = [f"This is document number {i} in the determinism test." for i in range(20)]
    dataset = ByteDataset(docs, seq_len=12)

    g1 = torch.Generator().manual_seed(42)
    loader1 = get_dataloader(dataset, batch_size=4, shuffle=True, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    loader2 = get_dataloader(dataset, batch_size=4, shuffle=True, generator=g2)

    batches1 = list(loader1)
    batches2 = list(loader2)

    assert len(batches1) == len(batches2)
    for (x1, y1), (x2, y2) in zip(batches1, batches2, strict=False):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Hello English world!", "English"),
        ("नमस्ते हिंदी दुनिया!", "Hindi"),
        ("你好中文世界！", "Chinese"),
        ("مرحبا بالعالم العربي!", "Arabic"),
        ("こんにちは日本！", "Japanese"),
        ("안녕하세요 한국!", "Korean"),
        ("🌍🚀🔥", "Emoji"),
    ],
)
def test_multilingual_dataset_validation(text: str, lang: str) -> None:
    """Verify that dataset handles diverse multilingual UTF-8 strings correctly."""
    dataset = ByteDataset([text], seq_len=8)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)


# --- Stress & Edge Case Tests ---


def test_dataset_empty_docs() -> None:
    """Verify dataset can handle empty documents without crashing."""
    dataset = ByteDataset(["", "Hello", "", "World"], seq_len=4)
    # empty docs should yield at least BOS/EOS/PAD sequence
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (4,)


def test_single_sample_batch() -> None:
    """Verify dataloader with batch_size=1 works."""
    dataset = ByteDataset(["Single sample text document."], seq_len=8)
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)
    for x, y in loader:
        assert x.shape == (1, 8)
        assert y.shape == (1, 8)
        break


# --- Performance Verification ---


def test_dataset_performance() -> None:
    """Measure batch generation speed, throughput, and memory stats."""
    # Generate 1000 synthetic documents
    docs = [f"This is document {i} for benchmarking data loading throughput." for i in range(1000)]

    t0 = time.perf_counter()
    dataset = ByteDataset(docs, seq_len=32)
    t_preprocess = time.perf_counter() - t0

    loader = get_dataloader(dataset, batch_size=64, shuffle=False)

    t1 = time.perf_counter()
    batch_count = 0
    total_bytes = 0
    for x, _ in loader:
        batch_count += 1
        total_bytes += x.numel() * 8  # int64 elements = 8 bytes each
    t_load = time.perf_counter() - t1

    samples_sec = len(dataset) / t_load if t_load > 0 else 0
    bytes_sec = total_bytes / t_load if t_load > 0 else 0

    print("\n[Performance Benchmark]")
    print(f"  Preprocessing time: {t_preprocess:.4f}s")
    print(f"  Batch loading time: {t_load:.4f}s")
    print(f"  Throughput:         {samples_sec:.2f} samples/sec")
    print(f"  Data rate:          {bytes_sec / 1024 / 1024:.2f} MB/sec")

    assert batch_count > 0
