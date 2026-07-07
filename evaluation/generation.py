# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Generation evaluation and text decoding capabilities for IVERI CORE.

Supports greedy decoding, temperature scaling, top-k filtering, and nucleus top-p sampling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from core.constants import EOS_BYTE


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Dataclass holding generation metrics and outputs."""

    output_ids: torch.Tensor
    latency_seconds: float
    bytes_per_second: float
    avg_generated_length: float
    early_stopped_ratio: float


class GenerationEvaluator:
    """Orchestrates autoregressive text generation and records inference metrics."""

    def __init__(self, eos_token_id: int = EOS_BYTE) -> None:
        """Initialize the GenerationEvaluator.

        Args:
            eos_token_id: End of sequence token ID to trigger stopping.
        """
        self.eos_token_id = eos_token_id

    @staticmethod
    def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits keeping only the top k values."""
        if top_k <= 0:
            return logits
        # Keep at least 1 element and at most vocab size
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits

    @staticmethod
    def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits keeping only cumulative top p probability distribution."""
        if top_p <= 0.0 or top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter indices to remove back to original logits shape
        for i in range(logits.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = float("-inf")

        return logits

    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_bytes: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: torch.device | None = None,
    ) -> GenerationResult:
        """Perform autoregressive generation starting from input_ids.

        Args:
            model: IVERI model.
            input_ids: Prompt token IDs of shape (B, S).
            max_new_bytes: Limit on maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 for greedy).
            top_k: Top-k filtering threshold.
            top_p: Top-p (nucleus) filtering threshold.
            device: Accelerator device to use.

        Returns:
            GenerationResult containing generated tokens and telemetry.
        """
        model.eval()
        device_resolved = device or next(model.parameters()).device
        input_ids = input_ids.to(device_resolved)

        batch_size = input_ids.size(0)
        curr_ids = input_ids.clone()
        generated_count = torch.zeros(batch_size, dtype=torch.long, device=device_resolved)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device_resolved)

        t0 = time.perf_counter()

        with torch.no_grad():
            for _ in range(max_new_bytes):
                # Forward pass: obtain logits of the last token
                outputs = model(curr_ids, return_dict=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                next_token_logits = logits[:, -1, :].clone()

                if temperature <= 1e-5:
                    # Greedy decoding
                    next_tokens = next_token_logits.argmax(dim=-1)
                else:
                    # Temperature scaling
                    next_token_logits = next_token_logits / temperature
                    # Apply top-k
                    if top_k > 0:
                        next_token_logits = self.apply_top_k(next_token_logits, top_k)
                    # Apply top-p
                    if 0.0 < top_p < 1.0:
                        next_token_logits = self.apply_top_p(next_token_logits, top_p)

                    # Sample from softmax probability distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Mask out tokens for already finished sequences
                next_tokens = torch.where(finished, torch.full_like(next_tokens, self.eos_token_id), next_tokens)

                # Append tokens
                curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(-1)], dim=-1)

                # Update states
                generated_count += (~finished).long()
                finished = finished | (next_tokens == self.eos_token_id)

                if finished.all():
                    break

        elapsed = time.perf_counter() - t0
        total_generated_bytes = generated_count.sum().item()

        # Metrics calculations
        bytes_per_sec = (total_generated_bytes / elapsed) if elapsed > 0 else 0.0
        avg_len = (total_generated_bytes / batch_size) if batch_size > 0 else 0.0
        early_stopped = (finished.sum().item() / batch_size) if batch_size > 0 else 0.0

        return GenerationResult(
            output_ids=curr_ids,
            latency_seconds=elapsed,
            bytes_per_second=bytes_per_sec,
            avg_generated_length=avg_len,
            early_stopped_ratio=early_stopped,
        )
