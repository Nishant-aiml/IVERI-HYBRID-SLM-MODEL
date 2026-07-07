# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Validation evaluator for pretraining convergence in IVERI CORE.

Computes next-byte perplexity, Bits-Per-Byte (BPB), top-1 next-byte prediction accuracy,
and top-5 next-byte prediction accuracy.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from evaluation.evaluator import Evaluator


class PretrainingEvaluator:
    """Orchestrates pretraining validation passes and calculates byte-level accuracy metrics."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.model = evaluator.model
        self.config = evaluator.config
        self.device = evaluator.device
        self.precision_handler = evaluator.precision_handler

    def evaluate_pretraining(self, val_dataloader: DataLoader) -> dict[str, float]:
        """Run validation evaluation pass and compute CE, Perplexity, BPB, and Accuracies."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_correct_top1 = 0
        total_correct_top5 = 0
        count = 0
        t0 = time.perf_counter()

        max_batches = self.config.evaluation.max_eval_batches

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs = batch["input_ids"]
                    targets = batch["labels"]
                else:
                    inputs = batch
                    targets = batch

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with self.precision_handler.autocast_context():
                    outputs = self.model(inputs, return_dict=True)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                    # Flat shapes
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = targets.view(-1)

                    # Compute cross entropy loss
                    ce_loss = torch.nn.functional.cross_entropy(
                        flat_logits, flat_targets, reduction="sum"
                    )

                    # Top-1 accuracy
                    preds_top1 = flat_logits.argmax(dim=-1)
                    correct_top1 = (preds_top1 == flat_targets).sum().item()

                    # Top-5 accuracy
                    _, preds_top5 = torch.topk(flat_logits, k=5, dim=-1)
                    correct_top5 = (preds_top5 == flat_targets.unsqueeze(-1)).any(dim=-1).sum().item()

                total_loss += ce_loss.item()
                total_tokens += flat_targets.numel()
                total_correct_top1 += correct_top1
                total_correct_top5 += correct_top5
                count += 1

        elapsed = time.perf_counter() - t0

        if total_tokens > 0:
            final_loss = total_loss / total_tokens
            # Bits-per-byte (BPB) is log_2(perplexity) = loss / ln(2)
            bpb = final_loss / math.log(2)
            try:
                perplexity = math.exp(final_loss)
            except OverflowError:
                perplexity = 0.0

            top1_acc = total_correct_top1 / total_tokens
            top5_acc = total_correct_top5 / total_tokens
        else:
            final_loss = 0.0
            bpb = 0.0
            perplexity = 0.0
            top1_acc = 0.0
            top5_acc = 0.0

        return {
            "val_loss": final_loss,
            "perplexity": perplexity,
            "bits_per_byte": bpb,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "eval_throughput_bytes_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
            "eval_latency_seconds": elapsed,
        }
