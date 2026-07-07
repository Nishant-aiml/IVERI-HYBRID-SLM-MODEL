# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SFT validation evaluator for IVERI CORE instruction tuning.

Extends the pretraining evaluation framework with SFT-specific metrics:
qualitative generation quality, response inspector, prompt suite evaluation,
and response length/latency tracking.

Metrics
-------
- Validation cross-entropy loss (on response bytes only)
- Perplexity
- Top-1 byte accuracy (on response bytes)
- Top-5 byte accuracy (on response bytes)
- Average response length (bytes)
- Generation latency (seconds)
- Bytes per second throughput
- Response quality score (0–1)
- Collapse / repetition / empty / UTF-8 error rates

Examples
--------
>>> from evaluation.sft_evaluator import SFTEvaluator
>>> # evaluator = SFTEvaluator(base_evaluator, config)
>>> # metrics = evaluator.evaluate_sft(val_loader, prompt_suite, model)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from evaluation.evaluator import Evaluator
from evaluation.prompt_suite import PromptSuite
from evaluation.response_inspector import ResponseInspector
from core.constants import PAD_BYTE

logger = logging.getLogger(__name__)

# ── Byte-level generation ──────────────────────────────────────────────────

_DEFAULT_MAX_NEW_BYTES: int = 128
_DEFAULT_TEMPERATURE: float = 0.8
_DEFAULT_TOP_K: int = 50


# ── Main evaluator class ───────────────────────────────────────────────────


class SFTEvaluator:
    """Orchestrates SFT validation and qualitative generation evaluation.

    Parameters
    ----------
    evaluator:
        Base :class:`~evaluation.evaluator.Evaluator` instance.
    config:
        Master IVERI configuration.
    response_inspector:
        Optional :class:`~evaluation.response_inspector.ResponseInspector`.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: IVERIConfig,
        response_inspector: ResponseInspector | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.model = evaluator.model
        self.config = config
        self.device = evaluator.device
        self.precision_handler = evaluator.precision_handler
        self.inspector = response_inspector or ResponseInspector()

    # ── Public API ─────────────────────────────────────────────────────

    def evaluate_sft(
        self,
        val_dataloader: DataLoader,
        use_loss_mask: bool = True,
    ) -> dict[str, float]:
        """Run SFT validation evaluation.

        Parameters
        ----------
        val_dataloader:
            DataLoader returning ``(x, y, loss_mask)`` triples from
            :class:`~training.sft_dataset.SFTByteDataset`.
        use_loss_mask:
            If ``True``, compute loss only on unmasked (response) positions.

        Returns
        -------
        dict[str, float]
            Evaluation metrics.
        """
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

                x, y, loss_mask = _unpack_sft_batch(batch)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                loss_mask = loss_mask.to(self.device, non_blocking=True)

                with self.precision_handler.autocast_context():
                    outputs = self.model(x, return_dict=True)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                    # Shape: (B, seq_len-1, 256)
                    B, S, V = logits.shape

                    flat_logits = logits.reshape(-1, V)
                    flat_targets = y.reshape(-1)
                    flat_mask = loss_mask.reshape(-1)

                    if use_loss_mask and flat_mask.any():
                        # Only compute metrics on response positions
                        sel_logits = flat_logits[flat_mask]
                        sel_targets = flat_targets[flat_mask]
                    else:
                        sel_logits = flat_logits
                        sel_targets = flat_targets

                    if sel_targets.numel() == 0:
                        continue

                    ce_loss = F.cross_entropy(sel_logits, sel_targets, reduction="sum")

                    preds_top1 = sel_logits.argmax(dim=-1)
                    correct_top1 = (preds_top1 == sel_targets).sum().item()

                    _, preds_top5 = torch.topk(sel_logits, k=min(5, V), dim=-1)
                    correct_top5 = (
                        preds_top5 == sel_targets.unsqueeze(-1)
                    ).any(dim=-1).sum().item()

                total_loss += ce_loss.item()
                total_tokens += sel_targets.numel()
                total_correct_top1 += correct_top1
                total_correct_top5 += correct_top5
                count += 1

        elapsed = time.perf_counter() - t0

        if total_tokens > 0:
            final_loss = total_loss / total_tokens
            bpb = final_loss / math.log(2)
            try:
                perplexity = math.exp(final_loss)
            except OverflowError:
                perplexity = float("inf")
            top1_acc = total_correct_top1 / total_tokens
            top5_acc = total_correct_top5 / total_tokens
        else:
            final_loss = bpb = perplexity = top1_acc = top5_acc = 0.0

        return {
            "sft/val_loss": final_loss,
            "sft/perplexity": perplexity,
            "sft/bits_per_byte": bpb,
            "sft/top1_accuracy": top1_acc,
            "sft/top5_accuracy": top5_acc,
            "sft/eval_batches": count,
            "sft/eval_tokens": total_tokens,
            "sft/eval_latency_sec": elapsed,
            "sft/eval_throughput_bps": total_tokens / elapsed if elapsed > 0 else 0.0,
        }

    def evaluate_prompt_suite(
        self,
        prompt_suite: PromptSuite,
        max_new_bytes: int = _DEFAULT_MAX_NEW_BYTES,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run qualitative generation on the fixed prompt suite.

        Parameters
        ----------
        prompt_suite:
            :class:`~evaluation.prompt_suite.PromptSuite` instance.
        max_new_bytes:
            Maximum new bytes to generate per prompt.
        temperature:
            Sampling temperature.
        top_k:
            Top-K sampling.
        seed:
            Random seed for deterministic sampling.

        Returns
        -------
        dict[str, Any]
            Per-prompt results and aggregate statistics.
        """
        self.model.eval()
        torch.manual_seed(seed)

        prompts = prompt_suite.get_all()
        results: list[dict[str, Any]] = []
        raw_responses: list[bytes] = []
        total_latency = 0.0

        with torch.no_grad():
            for ep in prompts:
                # Format prompt as Alpaca-style bytes
                prompt_text = f"### Instruction:\n{ep.instruction}\n\n### Response:\n"
                if ep.context:
                    prompt_text = (
                        f"### Instruction:\n{ep.instruction}\n\n"
                        f"### Input:\n{ep.context}\n\n### Response:\n"
                    )

                prompt_bytes = prompt_text.encode("utf-8")
                t_gen0 = time.perf_counter()
                response_bytes = self._generate(
                    prompt_bytes, max_new_bytes, temperature, top_k
                )
                gen_latency = time.perf_counter() - t_gen0
                total_latency += gen_latency

                raw_responses.append(response_bytes)
                insp = self.inspector.inspect_bytes(response_bytes)
                quality_score = self.inspector.score_response(response_bytes)

                # Keyword scoring
                response_text = response_bytes.decode("utf-8", errors="replace").lower()
                keyword_hits = sum(
                    1 for kw in ep.expected_keywords
                    if kw.lower() in response_text
                )
                keyword_ratio = keyword_hits / max(len(ep.expected_keywords), 1)

                results.append({
                    "prompt_id": ep.prompt_id,
                    "category": ep.category,
                    "difficulty": ep.difficulty,
                    "instruction": ep.instruction,
                    "response": response_bytes.decode("utf-8", errors="replace"),
                    "response_length": insp.length,
                    "is_valid": insp.is_valid,
                    "issues": insp.issues,
                    "entropy": insp.entropy,
                    "quality_score": quality_score,
                    "keyword_ratio": keyword_ratio,
                    "latency_sec": gen_latency,
                })

        # Aggregate
        batch_stats = self.inspector.inspect_batch(raw_responses)
        n = len(results)

        return {
            "suite_version": prompt_suite.version,
            "num_prompts": n,
            "avg_quality_score": sum(r["quality_score"] for r in results) / max(n, 1),
            "avg_keyword_ratio": sum(r["keyword_ratio"] for r in results) / max(n, 1),
            "avg_response_length": sum(r["response_length"] for r in results) / max(n, 1),
            "avg_latency_sec": total_latency / max(n, 1),
            "avg_entropy": batch_stats["avg_entropy"],
            "valid_ratio": batch_stats["valid_ratio"],
            "issue_counts": batch_stats["issue_counts"],
            "per_prompt": results,
        }

    # ── Private: byte-level generation ────────────────────────────────

    def _generate(
        self,
        prompt_bytes: bytes,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
    ) -> bytes:
        """Generate new bytes autoregressively from a prompt.

        Parameters
        ----------
        prompt_bytes:
            Raw UTF-8 prompt bytes.
        max_new_bytes:
            Maximum new bytes to generate.
        temperature:
            Sampling temperature.
        top_k:
            Top-K filter.

        Returns
        -------
        bytes
            Generated response bytes (not including prompt).
        """
        seq_len = self.config.training.seq_len
        ctx = list(prompt_bytes[-seq_len:])  # Truncate prompt to seq_len
        generated: list[int] = []

        for _ in range(max_new_bytes):
            ctx_tensor = torch.tensor(
                ctx[-seq_len:], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            with self.precision_handler.autocast_context():
                outputs = self.model(ctx_tensor, return_dict=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                # Take last position logits
                next_logits = logits[0, -1, :]  # (256,)

            next_byte = _sample_byte(next_logits, temperature=temperature, top_k=top_k)
            generated.append(next_byte)
            ctx.append(next_byte)

            # Stop if we generate a double newline (end of response)
            if (
                len(generated) >= 2
                and generated[-1] == ord("\n")
                and generated[-2] == ord("\n")
            ):
                break

        return bytes(generated)


# ── Private helpers ────────────────────────────────────────────────────────


def _unpack_sft_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a batch from SFTByteDataset into (x, y, loss_mask)."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        mask = torch.ones_like(y, dtype=torch.bool)
        return x, y, mask
    if isinstance(batch, dict):
        x = batch["input_ids"]
        y = batch["labels"]
        mask = batch.get("loss_mask", torch.ones_like(y, dtype=torch.bool))
        return x, y, mask
    raise ValueError(f"Cannot unpack batch of type {type(batch)}")


def _sample_byte(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> int:
    """Sample a byte from logits with temperature and top-k filtering.

    Parameters
    ----------
    logits:
        Shape ``(256,)`` raw logits.
    temperature:
        Sampling temperature.
    top_k:
        Top-K filter.

    Returns
    -------
    int
        Sampled byte value 0–255.
    """
    if temperature <= 0.0:
        return int(logits.argmax().item())

    scaled = logits / temperature

    if top_k > 0:
        top_k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, top_k)
        min_val = values[-1]
        scaled = scaled.masked_fill(scaled < min_val, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
