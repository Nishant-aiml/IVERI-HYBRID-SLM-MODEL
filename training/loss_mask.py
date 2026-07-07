# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Loss mask builder for SFT byte-level instruction tuning.

Generates per-byte binary masks that control which positions contribute to
the cross-entropy loss.  Three strategies are supported:

``train_only_assistant``
    Only assistant-response bytes are unmasked (loss computed on responses only).
    This is the recommended default for instruction tuning.

``train_entire_sequence``
    All bytes contribute to loss (equivalent to standard language modelling).

``custom``
    Caller supplies a pre-built mask tensor directly.

The mask is a ``torch.BoolTensor`` of shape ``(seq_len,)`` where
``True`` means "this position contributes to loss" (unmasked).

Examples
--------
>>> builder = LossMaskBuilder()
>>> mask = builder.build_response_mask(spans, seq_len=512)
>>> assert mask.shape == (512,)
>>> assert mask.dtype == torch.bool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from core.constants import PAD_BYTE
from training.conversation_formatter import TextSpan

logger = logging.getLogger(__name__)

# ── Masking strategies ─────────────────────────────────────────────────────


class MaskStrategy(str, Enum):
    """Loss masking strategy enumeration."""

    TRAIN_ONLY_ASSISTANT = "train_only_assistant"
    TRAIN_ENTIRE_SEQUENCE = "train_entire_sequence"
    CUSTOM = "custom"


# ── Main builder class ─────────────────────────────────────────────────────


@dataclass
class MaskResult:
    """Output of :meth:`LossMaskBuilder.build`.

    Attributes
    ----------
    mask:
        Bool tensor of shape ``(seq_len,)``.  ``True`` = contributes to loss.
    prompt_ratio:
        Fraction of bytes that are prompt (masked out when training on responses only).
    response_ratio:
        Fraction of bytes that are response (unmasked).
    strategy:
        The masking strategy applied.
    """

    mask: torch.Tensor
    prompt_ratio: float
    response_ratio: float
    strategy: str


class LossMaskBuilder:
    """Build per-byte boolean loss masks for SFT training.

    Parameters
    ----------
    strategy:
        Default masking strategy.  Can be overridden per call.
    pad_byte:
        Byte value used for padding.  Padding positions are always masked.
    """

    def __init__(
        self,
        strategy: str | MaskStrategy = MaskStrategy.TRAIN_ONLY_ASSISTANT,
        pad_byte: int = PAD_BYTE,
    ) -> None:
        self.strategy = MaskStrategy(strategy)
        self.pad_byte = pad_byte

    # ── Public API ─────────────────────────────────────────────────────

    def build(
        self,
        byte_sequence: bytes | torch.Tensor,
        spans: list[TextSpan] | None = None,
        seq_len: int | None = None,
        custom_mask: torch.Tensor | None = None,
        strategy: str | MaskStrategy | None = None,
    ) -> MaskResult:
        """Build a loss mask for a single byte sequence.

        Parameters
        ----------
        byte_sequence:
            Raw bytes or integer tensor of shape ``(seq_len,)`` containing
            byte values 0–255.
        spans:
            List of :class:`TextSpan` objects from the formatter.  Required
            when *strategy* is ``train_only_assistant``.
        seq_len:
            Override sequence length.  Derived from *byte_sequence* when absent.
        custom_mask:
            Pre-built bool tensor.  Required when *strategy* is ``custom``.
        strategy:
            Override default strategy for this call.

        Returns
        -------
        MaskResult
            Mask tensor and statistics.

        Raises
        ------
        ValueError
            For invalid strategy / missing arguments.
        """
        effective_strategy = MaskStrategy(strategy) if strategy else self.strategy

        # Resolve sequence length
        if seq_len is None:
            if isinstance(byte_sequence, torch.Tensor):
                seq_len = byte_sequence.size(0)
            else:
                seq_len = len(byte_sequence)

        if effective_strategy == MaskStrategy.TRAIN_ENTIRE_SEQUENCE:
            mask = self.build_full_mask(byte_sequence, seq_len)
        elif effective_strategy == MaskStrategy.TRAIN_ONLY_ASSISTANT:
            mask = self.build_response_mask(spans or [], seq_len)
        elif effective_strategy == MaskStrategy.CUSTOM:
            if custom_mask is None:
                raise ValueError("custom_mask must be provided for strategy='custom'")
            mask = custom_mask.bool()
            if mask.size(0) != seq_len:
                mask = _pad_or_trim_mask(mask, seq_len)
        else:
            raise ValueError(f"Unknown strategy: {effective_strategy!r}")

        # Always mask padding bytes
        mask = self._mask_padding(mask, byte_sequence, seq_len)

        # Compute statistics
        total = mask.numel()
        n_response = int(mask.sum().item())
        n_prompt = total - n_response

        return MaskResult(
            mask=mask,
            prompt_ratio=n_prompt / total if total > 0 else 0.0,
            response_ratio=n_response / total if total > 0 else 0.0,
            strategy=effective_strategy.value,
        )

    def build_full_mask(
        self,
        byte_sequence: bytes | torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Return an all-True mask (train on every byte).

        Parameters
        ----------
        byte_sequence:
            Raw bytes or long tensor.
        seq_len:
            Sequence length.

        Returns
        -------
        torch.Tensor
            Bool tensor of shape ``(seq_len,)`` — all True.
        """
        return torch.ones(seq_len, dtype=torch.bool)

    def build_response_mask(
        self,
        spans: list[TextSpan],
        seq_len: int,
    ) -> torch.Tensor:
        """Return a mask that is True only at assistant response spans.

        Parameters
        ----------
        spans:
            Span list produced by :class:`ConversationFormatter`.
        seq_len:
            Sequence length (post truncation/padding).

        Returns
        -------
        torch.Tensor
            Bool tensor of shape ``(seq_len,)`` — True at assistant bytes.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for span in spans:
            if span.role == "assistant":
                start = min(span.start, seq_len)
                end = min(span.end, seq_len)
                if start < end:
                    mask[start:end] = True
        return mask

    def build_prompt_mask(
        self,
        spans: list[TextSpan],
        seq_len: int,
    ) -> torch.Tensor:
        """Return a mask that is True only at prompt (non-assistant) spans.

        Parameters
        ----------
        spans:
            Span list produced by :class:`ConversationFormatter`.
        seq_len:
            Sequence length.

        Returns
        -------
        torch.Tensor
            Bool tensor of shape ``(seq_len,)`` — True at prompt bytes.
        """
        full = torch.ones(seq_len, dtype=torch.bool)
        response = self.build_response_mask(spans, seq_len)
        return full & ~response

    def build_batch_masks(
        self,
        byte_sequences: list[bytes],
        spans_list: list[list[TextSpan]],
        seq_len: int,
        strategy: str | MaskStrategy | None = None,
    ) -> torch.Tensor:
        """Build a batch of masks stacked into shape ``(B, seq_len)``.

        Parameters
        ----------
        byte_sequences:
            List of raw byte strings.
        spans_list:
            Corresponding list of span lists.
        seq_len:
            Padded sequence length.
        strategy:
            Override default strategy.

        Returns
        -------
        torch.Tensor
            Bool tensor of shape ``(B, seq_len)``.
        """
        masks = []
        for bseq, spans in zip(byte_sequences, spans_list):
            result = self.build(bseq, spans=spans, seq_len=seq_len, strategy=strategy)
            masks.append(result.mask)
        return torch.stack(masks, dim=0)

    # ── Private helpers ─────────────────────────────────────────────────

    def _mask_padding(
        self,
        mask: torch.Tensor,
        byte_sequence: bytes | torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Set mask to False wherever the byte is the padding byte."""
        if isinstance(byte_sequence, (bytes, bytearray)):
            arr = torch.tensor(list(byte_sequence), dtype=torch.long)
        elif isinstance(byte_sequence, list):
            arr = torch.tensor(byte_sequence, dtype=torch.long)
        else:
            arr = byte_sequence.long()

        if arr.size(0) != seq_len:
            arr = _pad_or_trim_tensor(arr, seq_len, fill_value=self.pad_byte)

        pad_positions = arr == self.pad_byte
        mask = mask.clone()
        mask[pad_positions] = False
        return mask


# ── Module-level convenience functions ────────────────────────────────────


def make_response_only_mask(spans: list[TextSpan], seq_len: int) -> torch.Tensor:
    """Convenience: create a response-only loss mask without instantiating builder.

    Parameters
    ----------
    spans:
        Spans from the formatter.
    seq_len:
        Padded sequence length.

    Returns
    -------
    torch.Tensor
        Bool tensor of shape ``(seq_len,)``.
    """
    return LossMaskBuilder(MaskStrategy.TRAIN_ONLY_ASSISTANT).build_response_mask(spans, seq_len)


def apply_mask_to_loss(
    loss_tensor: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Apply a boolean mask to a per-token loss tensor.

    Parameters
    ----------
    loss_tensor:
        Unreduced per-token CE loss of shape ``(B * seq_len,)`` or ``(B, seq_len)``.
    mask:
        Bool mask of same shape as *loss_tensor*.
    reduction:
        ``"mean"`` or ``"sum"``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    flat_loss = loss_tensor.view(-1)
    flat_mask = mask.view(-1).to(flat_loss.device)
    masked = flat_loss * flat_mask.float()
    if reduction == "sum":
        return masked.sum()
    n = flat_mask.float().sum().clamp(min=1.0)
    return masked.sum() / n


# ── Private utilities ──────────────────────────────────────────────────────


def _pad_or_trim_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.size(0) >= target_len:
        return mask[:target_len]
    pad = torch.zeros(target_len - mask.size(0), dtype=torch.bool)
    return torch.cat([mask, pad])


def _pad_or_trim_tensor(
    t: torch.Tensor, target_len: int, fill_value: int = 0
) -> torch.Tensor:
    if t.size(0) >= target_len:
        return t[:target_len]
    pad = torch.full((target_len - t.size(0),), fill_value, dtype=t.dtype)
    return torch.cat([t, pad])
