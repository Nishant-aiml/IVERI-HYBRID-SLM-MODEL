# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Production-grade byte-level DataLoaders for IVERI CORE training stages.

NO TOKENIZER. IVERI processes raw UTF-8 bytes directly.
Provides a clean hierarchy:
BaseByteDataset -> PretrainByteDataset, SFTByteDataset, CodingByteDataset.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.constants import PAD_BYTE
from training.loss_mask import LossMaskBuilder, MaskStrategy


# Optional import of HuggingFace datasets
import os
if os.environ.get("IVERI_DISABLE_HF", "0") == "1":
    _HF_DATASETS_AVAILABLE = False
else:
    try:
        from datasets import load_from_disk

        _HF_DATASETS_AVAILABLE = True
    except (ImportError, Exception):
        _HF_DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseByteDataset(Dataset, ABC):
    """Abstract base class for byte-level datasets in the pipeline."""

    def __init__(self, seq_len: int = 512, split: str = "train") -> None:
        self.seq_len = seq_len
        self.split = split
        self._total_bytes = 0

    @abstractmethod
    def _load_data(self, data_path: str | Path) -> None:
        """Internal method to load data from path."""
        pass

    @property
    @abstractmethod
    def num_bytes(self) -> int:
        """Return total bytes in dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total samples/chunks in dataset."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Return statistics dict for the dataset."""
        n_bytes = self.num_bytes
        return {
            "num_samples": len(self),
            "seq_len": self.seq_len,
            "split": self.split,
            "num_bytes": n_bytes,
            "num_gb": n_bytes / (1024**3),
        }


class PretrainByteDataset(BaseByteDataset):
    """Stage 1 Foundation Pretraining byte-level dataset.

    Returns input/target byte ID sequences for next-byte prediction.
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int = 512,
        split: str = "train",
        text_field: str = "text",
    ) -> None:
        super().__init__(seq_len=seq_len, split=split)
        self.text_field = text_field
        self.data = np.array([], dtype=np.uint8)
        self._load_data(data_path)

    def _load_data(self, data_path: str | Path) -> None:
        path = Path(data_path)
        all_bytes = []

        # 1. Try HuggingFace load_from_disk if available
        if _HF_DATASETS_AVAILABLE and path.exists() and (path / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(path))
                logger.info(f"Loaded HF dataset from disk: {path}")
                for example in ds:
                    text = example.get(self.text_field, example.get("content", ""))
                    if text:
                        all_bytes.extend(text.encode("utf-8"))
                self.data = np.array(all_bytes, dtype=np.uint8)
                self._total_bytes = len(self.data)
                return
            except Exception as e:
                logger.warning(f"Failed to load as HF dataset: {e}. Falling back to file scan.")

        # 2. Fall back to scanning files
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                list(path.glob("**/*.jsonl"))
                + list(path.glob("**/*.json"))
                + list(path.glob("**/*.txt"))
            )
        else:
            files = []

        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                sample = json.loads(line)
                                text = sample.get(self.text_field, sample.get("content", ""))
                                if text:
                                    all_bytes.extend(text.encode("utf-8"))
                elif f.suffix == ".json":
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            for sample in data:
                                text = sample.get(self.text_field, sample.get("content", ""))
                                if text:
                                    all_bytes.extend(text.encode("utf-8"))
                        elif isinstance(data, dict):
                            text = data.get(self.text_field, data.get("content", ""))
                            if text:
                                all_bytes.extend(text.encode("utf-8"))
                elif f.suffix == ".txt":
                    with open(f, encoding="utf-8") as fh:
                        content = fh.read()
                        if content:
                            all_bytes.extend(content.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to load file {f}: {e}")

        self.data = np.array(all_bytes, dtype=np.uint8)
        self._total_bytes = len(self.data)

    @property
    def num_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


class SFTByteDataset(BaseByteDataset):
    """Stage 2 Instruction Tuning byte-level dataset.

    Formats Q+A pairs into structured byte sequences using user/assistant templates.
    """

    PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int = 512,
        split: str = "train",
        train_on_prompt: bool = False,
    ) -> None:
        super().__init__(seq_len=seq_len, split=split)
        self.train_on_prompt = train_on_prompt
        self.samples: list[bytes] = []
        self.mask_builder = LossMaskBuilder(
            strategy=MaskStrategy.TRAIN_ENTIRE_SEQUENCE if train_on_prompt else MaskStrategy.CUSTOM
        )
        self._load_data(data_path)


    def _format_sample(self, example: dict[str, Any]) -> str:
        """Format an SFT sample as a single string."""
        # Multi-turn messages format
        if "messages" in example:
            parts = []
            for msg in example["messages"]:
                role = msg["role"].capitalize()
                content = msg["content"]
                parts.append(f"### {role}:\n{content}")
            return "\n\n".join(parts)

        # Single-turn Alpaca format
        elif "instruction" in example and "output" in example:
            instruction = example["instruction"]
            inp = example.get("input", "")
            output = example["output"]

            if inp:
                instruction = f"{instruction}\n\nContext: {inp}"

            return self.PROMPT_TEMPLATE.format(instruction=instruction, output=output)

        return ""

    def _load_data(self, data_path: str | Path) -> None:
        path = Path(data_path)
        raw_samples = []

        # 1. Try HF dataset load_from_disk
        if _HF_DATASETS_AVAILABLE and path.exists() and (path / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(path))
                logger.info(f"Loaded HF dataset from disk: {path}")
                for example in ds:
                    text = self._format_sample(example)
                    if text:
                        raw_samples.append(text.encode("utf-8"))
                self.samples = raw_samples
                self._total_bytes = sum(len(x) for x in self.samples)
                return
            except Exception as e:
                logger.warning(f"Failed to load as HF dataset: {e}. Falling back to file scan.")

        # 2. File scan
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(list(path.glob("**/*.jsonl")) + list(path.glob("**/*.json")))
        else:
            files = []

        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                sample = json.loads(line)
                                text = self._format_sample(sample)
                                if text:
                                    raw_samples.append(text.encode("utf-8"))
                elif f.suffix == ".json":
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            for sample in data:
                                text = self._format_sample(sample)
                                if text:
                                    raw_samples.append(text.encode("utf-8"))
                        elif isinstance(data, dict):
                            text = self._format_sample(data)
                            if text:
                                raw_samples.append(text.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to load file {f}: {e}")

        self.samples = raw_samples
        self._total_bytes = sum(len(x) for x in self.samples)

    @property
    def num_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        byte_list = list(self.samples[idx])
        target_len = self.seq_len + 1

        sample_bytes = self.samples[idx]
        response_start = 0
        sep_alpaca = b"\n\n### Response:\n"
        sep_msg = b"### Assistant:\n"

        if sep_alpaca in sample_bytes:
            response_start = sample_bytes.rfind(sep_alpaca) + len(sep_alpaca)
        elif sep_msg in sample_bytes:
            response_start = sample_bytes.rfind(sep_msg) + len(sep_msg)

        full_mask = torch.zeros(len(byte_list), dtype=torch.bool)
        full_mask[response_start:] = True

        if len(byte_list) > target_len:
            byte_list = byte_list[:target_len]
            full_mask = full_mask[:target_len]
        else:
            pad_len = target_len - len(byte_list)
            byte_list = byte_list + [PAD_BYTE] * pad_len
            full_mask = torch.cat([full_mask, torch.zeros(pad_len, dtype=torch.bool)])

        tokens = torch.tensor(byte_list, dtype=torch.long)
        x = tokens[:-1].clone()
        y = tokens[1:].clone()

        if self.train_on_prompt:
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
        else:
            loss_mask = full_mask[1:].clone()

        result = self.mask_builder.build(
            y,
            custom_mask=loss_mask,
            seq_len=self.seq_len,
        )
        loss_mask = result.mask

        return x, y, loss_mask



class CodingByteDataset(BaseByteDataset):
    """Stage 3A Coding Specialization byte-level dataset.

    Loads code repositories and allows filtering based on languages and licenses.
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int = 512,
        split: str = "train",
        language_filter: str | None = None,
        license_filter: list[str] | None = None,
    ) -> None:
        super().__init__(seq_len=seq_len, split=split)
        self.language_filter = language_filter
        self.license_filter = license_filter
        self.data = np.array([], dtype=np.uint8)
        self._load_data(data_path)

    def _load_data(self, data_path: str | Path) -> None:
        path = Path(data_path)
        all_bytes = []

        # 1. Try HF dataset load_from_disk
        if _HF_DATASETS_AVAILABLE and path.exists() and (path / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(path))
                logger.info(f"Loaded HF dataset from disk: {path}")
                for example in ds:
                    # Filter check
                    if self.language_filter and example.get("language") != self.language_filter:
                        continue
                    if self.license_filter and example.get("license") not in self.license_filter:
                        continue

                    text = example.get("content", example.get("text", ""))
                    if text:
                        all_bytes.extend(text.encode("utf-8"))
                self.data = np.array(all_bytes, dtype=np.uint8)
                self._total_bytes = len(self.data)
                return
            except Exception as e:
                logger.warning(f"Failed to load as HF dataset: {e}. Falling back to file scan.")

        # 2. File scan
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                list(path.glob("**/*.jsonl"))
                + list(path.glob("**/*.json"))
                + list(path.glob("**/*.py"))
            )
        else:
            files = []

        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                sample = json.loads(line)
                                if (
                                    self.language_filter
                                    and sample.get("language") != self.language_filter
                                ):
                                    continue
                                if (
                                    self.license_filter
                                    and sample.get("license") not in self.license_filter
                                ):
                                    continue
                                text = sample.get("content", sample.get("text", ""))
                                if text:
                                    all_bytes.extend(text.encode("utf-8"))
                elif f.suffix == ".json":
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            for sample in data:
                                if (
                                    self.language_filter
                                    and sample.get("language") != self.language_filter
                                ):
                                    continue
                                if (
                                    self.license_filter
                                    and sample.get("license") not in self.license_filter
                                ):
                                    continue
                                text = sample.get("content", sample.get("text", ""))
                                if text:
                                    all_bytes.extend(text.encode("utf-8"))
                elif f.suffix == ".py":
                    with open(f, encoding="utf-8") as fh:
                        content = fh.read()
                        if content:
                            all_bytes.extend(content.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to load file {f}: {e}")

        self.data = np.array(all_bytes, dtype=np.uint8)
        self._total_bytes = len(self.data)

    @property
    def num_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


# ── Factory Functions ────────────────────────────────────────────────────────


def get_pretrain_dataloader(
    data_path: str | Path,
    batch_size: int = 8,
    seq_len: int = 512,
    num_workers: int = 2,
    split: str = "train",
    pin_memory: bool = True,
    drop_last: bool = True,
    text_field: str = "text",
) -> DataLoader:
    """Helper factory to instantiate a PretrainByteDataset dataloader."""
    dataset = PretrainByteDataset(data_path, seq_len=seq_len, split=split, text_field=text_field)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_sft_dataloader(
    data_path: str | Path,
    batch_size: int = 4,
    seq_len: int = 512,
    num_workers: int = 2,
    split: str = "train",
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Helper factory to instantiate an SFTByteDataset dataloader."""
    dataset = SFTByteDataset(data_path, seq_len=seq_len, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_coding_dataloader(
    data_path: str | Path,
    batch_size: int = 4,
    seq_len: int = 512,
    num_workers: int = 2,
    split: str = "train",
    pin_memory: bool = True,
    drop_last: bool = True,
    language_filter: str | None = None,
    license_filter: list[str] | None = None,
) -> DataLoader:
    """Helper factory to instantiate a CodingByteDataset dataloader."""
    dataset = CodingByteDataset(
        data_path,
        seq_len=seq_len,
        split=split,
        language_filter=language_filter,
        license_filter=license_filter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
