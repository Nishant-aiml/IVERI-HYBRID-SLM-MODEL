# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SFT byte-level dataset with loss masking for IVERI CORE instruction tuning.

This module provides :class:`SFTByteDataset` — an enhanced version of the
base SFT dataset in ``data/pipeline/dataloader.py``.  Key differences:

* Full :class:`~training.conversation_formatter.ConversationFormatter` support.
* Per-sample :class:`~training.loss_mask.LossMaskBuilder` integration.
* ``train_on_prompt=False`` to compute loss only on assistant responses.
* Attention mask support.
* Sequence packing and truncation.

Byte encoding contract
----------------------
All text is UTF-8 encoded.  Padding uses collision-free ``PAD_BYTE`` (257).
Content bytes map 1:1 to IDs 0–255; vocabulary size is 259.

Examples
--------
>>> from training.sft_dataset import SFTByteDataset
>>> ds = SFTByteDataset(samples=[{"instruction": "Hello", "output": "Hi"}], seq_len=64)
>>> x, y, mask = ds[0]
>>> assert x.shape == (63,)
>>> assert y.shape == (63,)
>>> assert mask.shape == (63,)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from core.constants import PAD_BYTE, RAW_BYTE_VOCAB_SIZE
from training.conversation_formatter import ConversationFormatter, FormatterConfig, TextSpan
from training.loss_mask import LossMaskBuilder, MaskStrategy, apply_mask_to_loss

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

from core.constants import PAD_BYTE, RAW_BYTE_VOCAB_SIZE

MAX_BYTE_VALUE: int = RAW_BYTE_VOCAB_SIZE - 1


# ── Main dataset class ─────────────────────────────────────────────────────


class SFTByteDataset(Dataset):
    """Byte-level SFT dataset for IVERI CORE instruction tuning.

    Each item returns a triple ``(x, y, loss_mask)`` where:

    * ``x`` — input bytes of shape ``(seq_len - 1,)``
    * ``y`` — target bytes of shape ``(seq_len - 1,)`` (shifted by 1)
    * ``loss_mask`` — bool mask of shape ``(seq_len - 1,)`` for masked CE loss

    Parameters
    ----------
    samples:
        List of raw dataset dicts.  Each must be compatible with
        :class:`~training.conversation_formatter.ConversationFormatter`.
    seq_len:
        Target padded sequence length (including the autoregressive shift).
    formatter:
        :class:`~training.conversation_formatter.ConversationFormatter` instance.
        A default Alpaca-style formatter is used when absent.
    train_on_prompt:
        If ``False``, prompt bytes are masked out of the loss (recommended
        for instruction tuning).  If ``True``, the model is trained on all bytes.
    packing:
        If ``True``, pack multiple short samples into each ``seq_len`` window
        (not yet implemented; reserved for future use).
    shuffle:
        Whether to shuffle the sample list at init time.
    seed:
        Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        seq_len: int = 512,
        formatter: ConversationFormatter | None = None,
        train_on_prompt: bool = False,
        packing: bool = False,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.train_on_prompt = train_on_prompt
        self.packing = packing

        self.formatter = formatter or ConversationFormatter()
        self.mask_builder = LossMaskBuilder(
            strategy=(
                MaskStrategy.TRAIN_ENTIRE_SEQUENCE
                if train_on_prompt
                else MaskStrategy.TRAIN_ONLY_ASSISTANT
            ),
            pad_byte=PAD_BYTE,
        )

        # Pre-encode all samples
        self._encoded: list[tuple[bytes, list[TextSpan]]] = []
        n_skip = 0
        for sample in samples:
            try:
                raw_bytes, spans = self.formatter.format_to_bytes(sample)
                if len(raw_bytes) == 0:
                    n_skip += 1
                    continue
                self._encoded.append((raw_bytes, spans))
            except Exception as exc:
                logger.debug("Skipping malformed sample: %s", exc)
                n_skip += 1

        if n_skip:
            logger.warning("SFTByteDataset: skipped %d malformed/empty samples.", n_skip)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._encoded)

        logger.info(
            "SFTByteDataset initialized: %d samples, seq_len=%d, train_on_prompt=%s",
            len(self._encoded),
            seq_len,
            train_on_prompt,
        )

    # ── Dataset protocol ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x, y, loss_mask)`` tensors for the given index.

        Parameters
        ----------
        idx:
            Sample index.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            * x — shape ``(seq_len-1,)`` dtype long
            * y — shape ``(seq_len-1,)`` dtype long
            * loss_mask — shape ``(seq_len-1,)`` dtype bool
        """
        raw_bytes, spans = self._encoded[idx]
        # Truncate or pad to seq_len
        padded = _truncate_or_pad(raw_bytes, self.seq_len)

        # Build tokens (seq_len,)
        token_ids = torch.tensor(list(padded), dtype=torch.long)

        # Autoregressive shift: x = tokens[:-1], y = tokens[1:]
        x = token_ids[:-1].clone()
        y = token_ids[1:].clone()

        # Build loss mask on the unshifted padded sequence, then shift
        if self.train_on_prompt:
            # Full mask — no masking
            loss_mask = torch.ones(self.seq_len - 1, dtype=torch.bool)
        else:
            # Response-only mask on full seq_len; shift by 1 to align with y
            full_mask_result = self.mask_builder.build(
                padded, spans=spans, seq_len=self.seq_len
            )
            # Shift mask: y[i] = token[i+1], so mask[i] = mask_full[i+1]
            loss_mask = full_mask_result.mask[1:]

        # Mask out pad positions in y
        pad_positions = y == PAD_BYTE
        loss_mask = loss_mask & ~pad_positions

        return x, y, loss_mask

    # ── Utilities ──────────────────────────────────────────────────────

    @property
    def num_samples(self) -> int:
        """Number of encoded samples."""
        return len(self._encoded)

    def get_sample_text(self, idx: int) -> str:
        """Return the raw formatted text for a sample (for debugging).

        Parameters
        ----------
        idx:
            Sample index.

        Returns
        -------
        str
            UTF-8 decoded formatted text.
        """
        raw_bytes, _ = self._encoded[idx]
        return raw_bytes.decode("utf-8", errors="replace")

    def get_span_stats(self) -> dict[str, float]:
        """Compute average prompt/response ratio across all samples.

        Returns
        -------
        dict[str, float]
            Keys: ``prompt_ratio``, ``response_ratio``, ``avg_length``.
        """
        total_prompt = 0
        total_response = 0
        total_len = 0
        for raw_bytes, spans in self._encoded:
            n = len(raw_bytes)
            total_len += n
            resp_bytes = sum(
                min(s.end, n) - min(s.start, n)
                for s in spans
                if s.role == "assistant"
            )
            total_prompt += n - resp_bytes
            total_response += resp_bytes
        total = total_prompt + total_response or 1
        return {
            "prompt_ratio": total_prompt / total,
            "response_ratio": total_response / total,
            "avg_length": total_len / max(len(self._encoded), 1),
        }


# ── Factory function ───────────────────────────────────────────────────────


def make_sft_dataloader(
    samples: list[dict[str, Any]],
    batch_size: int = 4,
    seq_len: int = 512,
    formatter: ConversationFormatter | None = None,
    train_on_prompt: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for SFT byte-level training.

    Parameters
    ----------
    samples:
        List of raw dataset dicts.
    batch_size:
        Mini-batch size.
    seq_len:
        Padded sequence length.
    formatter:
        Custom formatter.  Default is Alpaca-style.
    train_on_prompt:
        Whether to compute loss on prompt bytes.
    shuffle:
        Whether to shuffle per epoch.
    num_workers:
        DataLoader workers.  Set to 0 on Windows to avoid deadlocks.
    pin_memory:
        Pin memory for GPU transfer.
    drop_last:
        Drop incomplete final batch.
    seed:
        Shuffle seed.

    Returns
    -------
    DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = SFTByteDataset(
        samples=samples,
        seq_len=seq_len,
        formatter=formatter,
        train_on_prompt=train_on_prompt,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# ── Private helpers ────────────────────────────────────────────────────────


def _truncate_or_pad(raw_bytes: bytes, target_len: int) -> list[int]:
    """Truncate or right-pad with collision-free PAD token IDs."""
    from core.byte_vocab import content_bytes_to_token_ids

    ids = content_bytes_to_token_ids(raw_bytes)
    if len(ids) >= target_len:
        return ids[:target_len]
    return ids + [PAD_BYTE] * (target_len - len(ids))
