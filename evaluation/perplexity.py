# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Language modeling metrics: Cross Entropy, Negative Log Likelihood, and Perplexity.

This module provides numerically stable evaluation of language modeling performance,
compatible with batch evaluation, streaming evaluation, and mixed precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.mixed_precision import PrecisionHandler


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """Dataclass holding language modeling evaluation results."""

    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int


class PerplexityEvaluator:
    """Computes perplexity and cross entropy loss over datasets and batches."""

    def __init__(self, ignore_index: int = -100) -> None:
        """Initialize the PerplexityEvaluator.

        Args:
            ignore_index: Target index to ignore in cross entropy computation.
        """
        self.ignore_index = ignore_index

    def evaluate_batch(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
        """Compute metrics for a single batch of predictions.

        Args:
            logits: Predicted logits of shape (B, S, V) or (B * S, V).
            targets: Target labels of shape (B, S) or (B * S).

        Returns:
            Dictionary containing 'loss', 'num_tokens', and 'nll'.
        """
        if logits.dim() == 3:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
        else:
            flat_logits = logits
            flat_targets = targets

        # Cast to float32 for numerical stability
        flat_logits = flat_logits.float()

        # Compute cross entropy with standard PyTorch implementation
        loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.ignore_index,
            reduction="sum",
        )

        # Count active tokens (not ignored)
        if self.ignore_index != -100:
            active_mask = flat_targets != self.ignore_index
            num_tokens = active_mask.sum().item()
        else:
            num_tokens = flat_targets.numel()

        loss_val = loss.item()

        # Handle edge cases (empty batch or zero active tokens)
        if num_tokens == 0:
            return {"loss": 0.0, "num_tokens": 0, "nll": 0.0}

        mean_nll = loss_val / num_tokens

        # Check for NaN / Inf and sanitize
        if math.isnan(mean_nll) or math.isinf(mean_nll):
            mean_nll = 0.0
            loss_val = 0.0

        return {
            "loss": loss_val,
            "num_tokens": int(num_tokens),
            "nll": float(mean_nll),
        }

    def evaluate_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        precision_handler: PrecisionHandler,
        device: torch.device,
        max_batches: int | None = None,
    ) -> PerplexityResult:
        """Evaluate the model over a validation dataset.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader providing evaluation batches.
            precision_handler: PrecisionHandler for mixed-precision context.
            device: Accelerator device to place tensors on.
            max_batches: Optional limit on evaluation iterations.

        Returns:
            PerplexityResult container.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Parse standard dataset inputs and targets
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs = batch["input_ids"]
                    targets = batch["labels"]
                else:
                    inputs = batch
                    targets = batch

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with precision_handler.autocast_context():
                    outputs = model(inputs, return_dict=True)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    batch_metrics = self.evaluate_batch(logits, targets)

                total_loss += batch_metrics["loss"]
                total_tokens += batch_metrics["num_tokens"]
                num_batches += 1

        if total_tokens == 0:
            return PerplexityResult(loss=0.0, perplexity=0.0, num_tokens=0, num_batches=num_batches)

        final_loss = total_loss / total_tokens

        # Guard perplexity against extremely high exponent/Inf
        try:
            perplexity = math.exp(final_loss)
            if math.isnan(perplexity) or math.isinf(perplexity):
                perplexity = 0.0
        except OverflowError:
            perplexity = 0.0

        if math.isnan(final_loss) or math.isinf(final_loss):
            final_loss = 0.0

        return PerplexityResult(
            loss=float(final_loss),
            perplexity=float(perplexity),
            num_tokens=int(total_tokens),
            num_batches=int(num_batches),
        )
