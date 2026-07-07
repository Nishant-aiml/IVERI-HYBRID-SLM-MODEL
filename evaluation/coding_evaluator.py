# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Coding specialization evaluator for IVERI CORE Phase 3.3.

Extends the SFT evaluation framework with coding-specific metrics:
syntax validity (multi-language), code quality analysis, security scanning,
compilation/execution checks, and generation quality inspection.

Metric namespace: ``coding/``

Examples
--------
>>> from evaluation.coding_evaluator import CodingEvaluator
>>> # evaluator = CodingEvaluator(base_evaluator, config)
>>> # metrics = evaluator.evaluate_coding(val_loader)
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

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_BYTES: int = 256
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_TOP_K: int = 20


# ── Main evaluator class ───────────────────────────────────────────────────


class CodingEvaluator:
    """Orchestrates coding specialization validation and qualitative evaluation.

    Parameters
    ----------
    evaluator:
        Base :class:`~evaluation.evaluator.Evaluator` instance.
    config:
        Master IVERI configuration.
    code_inspector:
        Optional :class:`~evaluation.code_inspector.CodeInspector`.
    code_quality_analyzer:
        Optional :class:`~evaluation.code_quality_analyzer.CodeQualityAnalyzer`.
    security_scanner:
        Optional :class:`~evaluation.security_scanner.SecurityScanner`.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: IVERIConfig,
        code_inspector: Any | None = None,
        code_quality_analyzer: Any | None = None,
        security_scanner: Any | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.model = evaluator.model
        self.config = config
        self.device = evaluator.device
        self.precision_handler = evaluator.precision_handler

        # Lazily import optional components
        if code_inspector is None:
            try:
                from evaluation.code_inspector import CodeInspector
                code_inspector = CodeInspector()
            except ImportError:
                logger.debug("CodeInspector not available.")
        self.inspector = code_inspector

        if code_quality_analyzer is None:
            try:
                from evaluation.code_quality_analyzer import CodeQualityAnalyzer
                code_quality_analyzer = CodeQualityAnalyzer()
            except ImportError:
                logger.debug("CodeQualityAnalyzer not available.")
        self.quality_analyzer = code_quality_analyzer

        if security_scanner is None:
            try:
                from evaluation.security_scanner import SecurityScanner
                security_scanner = SecurityScanner()
            except ImportError:
                logger.debug("SecurityScanner not available.")
        self.security_scanner = security_scanner

    # ── Public API ─────────────────────────────────────────────────────

    def evaluate_coding(
        self,
        val_dataloader: DataLoader,
        use_loss_mask: bool = True,
        curriculum_stage: int = 0,
    ) -> dict[str, float]:
        """Run coding validation evaluation on the val dataloader.

        Parameters
        ----------
        val_dataloader:
            DataLoader returning ``(x, y, loss_mask)`` triples.
        use_loss_mask:
            Compute loss only on response positions.
        curriculum_stage:
            Current curriculum stage index (logged as telemetry).

        Returns
        -------
        dict[str, float]
            All coding evaluation metrics with ``coding/`` prefix.
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

                x, y, loss_mask = _unpack_batch(batch)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                loss_mask = loss_mask.to(self.device, non_blocking=True)

                with self.precision_handler.autocast_context():
                    outputs = self.model(x, return_dict=True)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                    B, S, V = logits.shape
                    flat_logits = logits.reshape(-1, V)
                    flat_targets = y.reshape(-1)
                    flat_mask = loss_mask.reshape(-1)

                    if use_loss_mask and flat_mask.any():
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
            "coding/val_loss": final_loss,
            "coding/perplexity": perplexity,
            "coding/bits_per_byte": bpb,
            "coding/top1_accuracy": top1_acc,
            "coding/top5_accuracy": top5_acc,
            "coding/eval_batches": count,
            "coding/eval_tokens": total_tokens,
            "coding/eval_latency_sec": elapsed,
            "coding/eval_throughput_bps": total_tokens / elapsed if elapsed > 0 else 0.0,
            "coding/curriculum_stage": float(curriculum_stage),
        }

    def evaluate_code_prompt_suite(
        self,
        prompt_suite: Any,
        max_new_bytes: int = _DEFAULT_MAX_NEW_BYTES,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run qualitative generation on the coding prompt suite.

        Parameters
        ----------
        prompt_suite:
            :class:`~evaluation.coding_prompt_suite.CodingPromptSuite`.
        max_new_bytes:
            Maximum new bytes per prompt.
        temperature:
            Sampling temperature (lower = more deterministic code).
        top_k:
            Top-K sampling.
        seed:
            Random seed.

        Returns
        -------
        dict[str, Any]
            Aggregate stats + per-prompt results + code quality aggregates.
        """
        self.model.eval()
        torch.manual_seed(seed)

        prompts = prompt_suite.get_all()
        results: list[dict[str, Any]] = []
        raw_responses: list[bytes] = []
        total_latency = 0.0

        seq_len = self.config.training.seq_len
        coding_cfg = getattr(self.config, "coding", None)
        code_prefix = getattr(coding_cfg, "code_prefix", "### Code Task:\n")
        solution_prefix = getattr(coding_cfg, "solution_prefix", "### Solution:\n")

        with torch.no_grad():
            for ep in prompts:
                prompt_text = f"{code_prefix}{ep.instruction}\n\n{solution_prefix}"
                if ep.context:
                    prompt_text = (
                        f"{code_prefix}{ep.instruction}\n\n"
                        f"### Context:\n{ep.context}\n\n{solution_prefix}"
                    )

                prompt_bytes = prompt_text.encode("utf-8")
                t0 = time.perf_counter()
                response_bytes = self._generate(prompt_bytes, max_new_bytes, temperature, top_k)
                gen_latency = time.perf_counter() - t0
                total_latency += gen_latency
                raw_responses.append(response_bytes)

                response_text = response_bytes.decode("utf-8", errors="replace")

                # Code inspection
                insp_result = None
                quality_score = 0.5
                if self.inspector is not None:
                    insp_result = self.inspector.inspect_bytes(response_bytes)
                    quality_score = self.inspector.score_response(response_bytes)

                # Code quality analysis
                quality_metrics: dict[str, Any] = {}
                if self.quality_analyzer is not None:
                    try:
                        q = self.quality_analyzer.analyze(response_text)
                        quality_metrics = {
                            "cyclomatic_complexity": q.cyclomatic_complexity,
                            "function_count": q.function_count,
                            "comment_ratio": q.comment_ratio,
                            "docstring_ratio": q.docstring_ratio,
                        }
                    except Exception as exc:
                        logger.debug("Quality analysis failed: %s", exc)

                # Security scan
                security_flags: list[str] = []
                if self.security_scanner is not None:
                    try:
                        sec = self.security_scanner.scan(response_text)
                        security_flags = sec.flagged_patterns
                    except Exception as exc:
                        logger.debug("Security scan failed: %s", exc)

                # Keyword scoring
                keyword_hits = sum(
                    1 for kw in getattr(ep, "expected_keywords", [])
                    if kw.lower() in response_text.lower()
                )
                keyword_ratio = keyword_hits / max(len(getattr(ep, "expected_keywords", []) or [1]), 1)

                # Syntax check
                syntax_valid = None
                if self.inspector is not None and insp_result is not None:
                    syntax_valid = insp_result.syntax_valid

                results.append({
                    "prompt_id": getattr(ep, "prompt_id", "?"),
                    "category": getattr(ep, "category", "?"),
                    "difficulty": getattr(ep, "difficulty", "medium"),
                    "instruction": ep.instruction,
                    "response": response_text,
                    "response_length": len(response_bytes),
                    "is_valid": insp_result.is_valid if insp_result else True,
                    "syntax_valid": syntax_valid,
                    "entropy": insp_result.entropy if insp_result else 0.0,
                    "quality_score": quality_score,
                    "keyword_ratio": keyword_ratio,
                    "security_flags": security_flags,
                    "latency_sec": gen_latency,
                    **quality_metrics,
                })

        n = len(results) or 1

        # Aggregate syntax validity (only where not None)
        syntax_checked = [r["syntax_valid"] for r in results if r["syntax_valid"] is not None]
        syntax_valid_ratio = sum(1 for s in syntax_checked if s) / max(len(syntax_checked), 1) if syntax_checked else 0.0

        # Security aggregate
        security_issue_ratio = sum(1 for r in results if r["security_flags"]) / n

        # Quality aggregate
        avg_cyclomatic = _safe_avg([r.get("cyclomatic_complexity", 0.0) for r in results])
        avg_function_count = _safe_avg([r.get("function_count", 0.0) for r in results])
        avg_comment_ratio = _safe_avg([r.get("comment_ratio", 0.0) for r in results])

        return {
            "suite_version": getattr(prompt_suite, "version", "3A-v1.0"),
            "num_prompts": n,
            "avg_quality_score": sum(r["quality_score"] for r in results) / n,
            "avg_keyword_ratio": sum(r["keyword_ratio"] for r in results) / n,
            "avg_response_length": sum(r["response_length"] for r in results) / n,
            "avg_latency_sec": total_latency / n,
            "avg_entropy": sum(r["entropy"] for r in results) / n,
            "valid_ratio": sum(r["is_valid"] for r in results) / n,
            "syntax_valid_ratio": syntax_valid_ratio,
            "security_issue_ratio": security_issue_ratio,
            "avg_cyclomatic_complexity": avg_cyclomatic,
            "avg_function_count": avg_function_count,
            "avg_comment_ratio": avg_comment_ratio,
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
        """Autoregressive byte-level generation from a prompt."""
        seq_len = self.config.training.seq_len
        ctx = list(prompt_bytes[-seq_len:])
        generated: list[int] = []

        for _ in range(max_new_bytes):
            ctx_tensor = torch.tensor(
                ctx[-seq_len:], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            with self.precision_handler.autocast_context():
                outputs = self.model(ctx_tensor, return_dict=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                next_logits = logits[0, -1, :]

            next_byte = _sample_byte(next_logits, temperature=temperature, top_k=top_k)
            generated.append(next_byte)
            ctx.append(next_byte)

            if (
                len(generated) >= 2
                and generated[-1] == ord("\n")
                and generated[-2] == ord("\n")
            ):
                break

        return bytes(generated)


# ── Private helpers ────────────────────────────────────────────────────────


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a batch into ``(x, y, loss_mask)``."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, loss_mask = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        loss_mask = None
    elif isinstance(batch, dict):
        x = batch["input_ids"]
        y = batch["labels"]
        loss_mask = batch.get("loss_mask", None)
    else:
        x, y = batch, batch
        loss_mask = None

    # Defensive conversion to tensor
    if not isinstance(x, torch.Tensor):
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            x = torch.stack(list(x))
        else:
            x = torch.tensor(x, dtype=torch.long)

    if not isinstance(y, torch.Tensor):
        if isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], torch.Tensor):
            y = torch.stack(list(y))
        else:
            y = torch.tensor(y, dtype=torch.long)

    if loss_mask is None:
        loss_mask = torch.ones_like(y, dtype=torch.bool)
    elif not isinstance(loss_mask, torch.Tensor):
        if isinstance(loss_mask, (list, tuple)) and len(loss_mask) > 0 and isinstance(loss_mask[0], torch.Tensor):
            loss_mask = torch.stack(list(loss_mask))
        else:
            loss_mask = torch.tensor(loss_mask, dtype=torch.bool)

    return x, y, loss_mask


def _sample_byte(logits: torch.Tensor, temperature: float = 0.2, top_k: int = 20) -> int:
    """Sample a byte value from logits with temperature and top-k."""
    if temperature <= 0.0:
        return int(logits.argmax().item())
    scaled = logits / temperature
    if top_k > 0:
        top_k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, top_k)
        scaled = scaled.masked_fill(scaled < values[-1], float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _safe_avg(values: list[Any]) -> float:
    """Compute mean of a list, filtering None and non-numeric values."""
    nums = [v for v in values if isinstance(v, (int, float)) and v == v]
    return sum(nums) / len(nums) if nums else 0.0
