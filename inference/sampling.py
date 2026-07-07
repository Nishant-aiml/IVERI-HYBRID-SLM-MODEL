# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Sampling utilities for inference (wraps evaluation.generation)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.generation import GenerationEvaluator


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128


class Sampler:
    """Applies temperature, top-k, top-p, and repetition penalty to logits."""

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()
        self._evaluator = GenerationEvaluator()

    def sample_next_token(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """Sample a single next-token ID from last-position logits."""
        next_logits = logits[:, -1, :].clone()

        if self.config.repetition_penalty != 1.0 and generated.numel() > 0:
            for token_id in generated[0].unique().tolist():
                if 0 <= token_id < next_logits.size(-1):
                    next_logits[0, int(token_id)] /= self.config.repetition_penalty

        if self.config.temperature <= 1e-5:
            return next_logits.argmax(dim=-1, keepdim=True)

        next_logits = next_logits / max(self.config.temperature, 1e-5)
        next_logits = self._evaluator.apply_top_k(next_logits, self.config.top_k)
        next_logits = self._evaluator.apply_top_p(next_logits, self.config.top_p)
        probs = torch.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
