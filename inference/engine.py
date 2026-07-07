# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Production inference engine with streaming and batching."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from core.constants import EOS_BYTE
from inference.byte_tokenizer import ByteTokenizer
from inference.sampling import Sampler, SamplingConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    token_id: int
    text_delta: str
    finished: bool = False


@dataclass
class InferenceResult:
    token_ids: list[int]
    text: str
    latency_seconds: float
    tokens_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """High-level inference API over IVERIModel (no architecture changes)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer: ByteTokenizer | None = None,
        sampling: SamplingConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or ByteTokenizer()
        self.sampling = sampling or SamplingConfig()
        self.sampler = Sampler(self.sampling)
        self.device = device or next(model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> InferenceResult:
        """Generate completion for a text prompt."""
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )
        max_tokens = max_new_tokens or self.sampling.max_new_tokens
        stop_bytes = [s.encode("utf-8") for s in (stop_sequences or [])]

        t0 = time.perf_counter()
        generated: list[int] = []
        curr = input_ids

        for _ in range(max_tokens):
            outputs = self.model(curr, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token = self.sampler.sample_next_token(logits, curr)
            token_id = int(next_token.item())
            generated.append(token_id)
            curr = torch.cat([curr, next_token], dim=-1)

            if token_id == EOS_BYTE:
                break
            if stop_bytes and any(
                bytes(generated).endswith(sb) for sb in stop_bytes
            ):
                break

        elapsed = time.perf_counter() - t0
        text = self.tokenizer.decode(generated)
        tps = len(generated) / elapsed if elapsed > 0 else 0.0
        return InferenceResult(
            token_ids=generated,
            text=text,
            latency_seconds=elapsed,
            tokens_per_second=tps,
        )

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield incremental decoded chunks during generation."""
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )
        max_tokens = max_new_tokens or self.sampling.max_new_tokens
        curr = input_ids
        prev_text = ""

        for _ in range(max_tokens):
            outputs = self.model(curr, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_token = self.sampler.sample_next_token(logits, curr)
            token_id = int(next_token.item())
            curr = torch.cat([curr, next_token], dim=-1)

            gen_ids = curr[0, input_ids.size(1) :].tolist()
            full_text = self.tokenizer.decode(gen_ids)
            delta = full_text[len(prev_text) :]
            prev_text = full_text
            finished = token_id == EOS_BYTE
            yield StreamChunk(token_id=token_id, text_delta=delta, finished=finished)
            if finished:
                break

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[InferenceResult]:
        """Sequential batch inference (memory-safe; no architecture change)."""
        return [self.generate(p, **kwargs) for p in prompts]
