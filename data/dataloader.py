# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""DataLoader infrastructure for IVERI CORE.

Provides map-style and iterable-style PyTorch Datasets representing raw byte sequence data
for autoregressive next-byte prediction pretraining.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from core.byte_vocab import content_bytes_to_token_ids
from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE
from data.dataset_utils import stream_documents_from_files
from data.preprocessing import chunk_byte_ids, pad_byte_ids, text_to_byte_ids
from utils.validation import validate_dtype, validate_shape


def _prepare_byte_ids(
    doc: str | bytes,
    *,
    add_bos: bool,
    add_eos: bool,
) -> list[int]:
    if isinstance(doc, str):
        return text_to_byte_ids(doc, add_bos=add_bos, add_eos=add_eos)
    byte_ids = content_bytes_to_token_ids(doc)
    if add_bos:
        byte_ids = [BOS_BYTE, *byte_ids]
    if add_eos:
        byte_ids = [*byte_ids, EOS_BYTE]
    return byte_ids


class ByteDataset(Dataset):
    """Map-style PyTorch Dataset for loading and pre-processing byte-level text data.

    Accepts raw string documents or pre-encoded byte sequences, chunks them to
    seq_len + 1 size, and returns (input_ids, labels) tuples.
    """

    def __init__(
        self,
        documents: list[str] | list[bytes],
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """Initialize the ByteDataset.

        Args:
            documents: List of raw string documents or encoded byte sequences.
            seq_len: Target sequence length for training.
            add_bos: Whether to prepend BOS_BYTE.
            add_eos: Whether to append EOS_BYTE.
        """
        self.seq_len = seq_len
        self.chunks: list[list[int]] = []

        # Autoregressive modeling requires inputs of size seq_len and targets of size seq_len.
        # This requires chunks of size seq_len + 1.
        chunk_len = seq_len + 1

        for doc in documents:
            byte_ids = _prepare_byte_ids(doc, add_bos=add_bos, add_eos=add_eos)
            doc_chunks = chunk_byte_ids(byte_ids, seq_len=chunk_len, overlap=0)
            for chunk in doc_chunks:
                if len(chunk) < chunk_len:
                    chunk = pad_byte_ids(chunk, seq_len=chunk_len, pad_val=PAD_BYTE)
                self.chunks.append(chunk)

    def __len__(self) -> int:
        """Get total number of chunks in the dataset."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single chunk of sequence data.

        Args:
            idx: index of chunk.

        Returns:
            Tuple of (input_ids, labels) both of shape (S,) and type torch.int64.
        """
        chunk = self.chunks[idx]
        inputs_raw = chunk[:-1]
        targets_raw = chunk[1:]

        input_ids = torch.tensor(inputs_raw, dtype=torch.long)
        labels = torch.tensor(targets_raw, dtype=torch.long)

        validate_shape(input_ids, (self.seq_len,), name="input_ids")
        validate_shape(labels, (self.seq_len,), name="labels")
        validate_dtype(input_ids, torch.long, name="input_ids")
        validate_dtype(labels, torch.long, name="labels")

        return input_ids, labels


class StreamingByteDataset(IterableDataset):
    """Iterable-style PyTorch Dataset for streaming raw byte-level text datasets.

    Streams documents from file paths, dynamically chunks the UTF-8 bytes to
    seq_len + 1 size, and yields (input_ids, labels) tuples.
    """

    def __init__(
        self,
        file_paths: list[pathlib.Path],
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """Initialize the StreamingByteDataset.

        Args:
            file_paths: List of file path locations.
            seq_len: Target sequence length for training.
            add_bos: Whether to prepend BOS_BYTE.
            add_eos: Whether to append EOS_BYTE.
        """
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterable interface yielding (input_ids, labels) sequentially.

        Correctly handles PyTorch multi-worker dataloading by partitioning the
        file list among workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_files = self.file_paths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_files = [
                path for i, path in enumerate(self.file_paths) if i % num_workers == worker_id
            ]

        chunk_len = self.seq_len + 1
        buffer: list[int] = []

        for doc in stream_documents_from_files(worker_files):
            buffer.extend(
                _prepare_byte_ids(doc, add_bos=self.add_bos, add_eos=self.add_eos)
            )

            while len(buffer) >= chunk_len:
                chunk = buffer[:chunk_len]
                del buffer[:chunk_len]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                validate_shape(input_ids, (self.seq_len,), name="input_ids")
                validate_shape(labels, (self.seq_len,), name="labels")

                yield input_ids, labels

        if buffer:
            padded = pad_byte_ids(buffer, seq_len=chunk_len, pad_val=PAD_BYTE)
            input_ids = torch.tensor(padded[:-1], dtype=torch.long)
            labels = torch.tensor(padded[1:], dtype=torch.long)

            validate_shape(input_ids, (self.seq_len,), name="input_ids")
            validate_shape(labels, (self.seq_len,), name="labels")

            yield input_ids, labels


def get_dataloader(
    dataset: Dataset | IterableDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a standardized PyTorch DataLoader for IVERI datasets.

    Args:
        dataset: The ByteDataset or StreamingByteDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle (only compatible with Map-style Dataset).
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.
        drop_last: Set to True to drop the last incomplete batch.
        generator: PyTorch Generator for reproducibility.

    Returns:
        torch.utils.data.DataLoader instance.
    """
    if isinstance(dataset, IterableDataset) and shuffle:
        raise ValueError("IterableDataset does not support shuffle=True")

    persistent_workers = num_workers > 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        generator=generator,
    )
