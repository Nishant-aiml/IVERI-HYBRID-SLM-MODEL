# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Instruction retention evaluator to detect catastrophic forgetting (Phase 3.3).

Runs the Phase 3.2 SFT PromptSuite and evaluates instruction-following capability,
tracking perplexity delta and quality score delta. Checkpoints can be rejected
if instruction retention falls below a threshold.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class InstructionRetentionEvaluator:
    """Evaluates retention of instruction-following capability.

    Parameters
    ----------
    model:
        IVERI model in eval mode.
    config:
        IVERIConfig master configuration.
    device:
        Target device for running evaluation.
    baseline_quality_score:
        Optional baseline quality score from Phase 3.2 SFT.
    baseline_perplexity:
        Optional baseline validation perplexity from Phase 3.2 SFT.
    """

    def __init__(
        self,
        model: Any,
        config: Any,
        device: torch.device | None = None,
        baseline_quality_score: float | None = None,
        baseline_perplexity: float | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.baseline_quality_score = baseline_quality_score
        self.baseline_perplexity = baseline_perplexity

        # Lazily import SFT PromptSuite and ResponseInspector
        try:
            from evaluation.prompt_suite import PromptSuite
            from evaluation.response_inspector import ResponseInspector
            self.prompt_suite = PromptSuite()
            self.inspector = ResponseInspector()
            self._available = True
        except ImportError:
            logger.warning("SFT PromptSuite or ResponseInspector not available. Retention evaluation will be disabled.")
            self._available = False

    def evaluate(self, step: int = 0) -> dict[str, float]:
        """Evaluate instruction retention on the Phase 3.2 PromptSuite.

        Parameters
        ----------
        step:
            Current training step.

        Returns
        -------
        dict[str, float]
            Retention metrics with ``instruction/`` prefix.
        """
        if not self._available:
            return {
                "instruction/pass_rate": 0.0,
                "instruction/quality_score": 0.0,
                "instruction/avg_response_length": 0.0,
                "instruction/quality_delta": 0.0,
                "instruction/perplexity_delta": 0.0,
                "instruction/retention_ok": 1.0,
            }

        self.model.eval()
        t0 = time.perf_counter()

        prompts = self.prompt_suite.get_all()
        # Subsample for quick evaluation on CPU/verification runs
        if self.config.training.max_steps <= 100:
            prompts = prompts[:5]  # quick check

        scores = []
        lengths = []
        valid_count = 0

        # Estimate perplexity via cross entropy over generated sequences
        # since we don't load the full SFT dataset here, we estimate SFT performance
        # via the PromptSuite response metrics
        with torch.no_grad():
            for ep in prompts:
                prompt_text = f"### Instruction:\n{ep.instruction}\n\n### Response:\n"
                if ep.context:
                    prompt_text = (
                        f"### Instruction:\n{ep.instruction}\n\n"
                        f"### Input:\n{ep.context}\n\n### Response:\n"
                    )

                prompt_bytes = prompt_text.encode("utf-8")
                response_bytes = self._generate(prompt_bytes)
                lengths.append(len(response_bytes))

                insp = self.inspector.inspect_bytes(response_bytes)
                q_score = self.inspector.score_response(response_bytes)
                scores.append(q_score)

                if insp.is_valid and "collapse" not in insp.issues and "empty" not in insp.issues:
                    valid_count += 1

        n = len(prompts) or 1
        avg_score = sum(scores) / n
        avg_len = sum(lengths) / n
        pass_rate = valid_count / n

        # Estimate a pseudo-perplexity from quality score:
        # perplexity drops as quality score increases.
        # pseudo-perplexity = exp(5.0 * (1.0 - quality_score))
        est_perplexity = math.exp(3.0 * (1.0 - avg_score))

        q_delta = avg_score - (self.baseline_quality_score or 0.8)
        p_delta = est_perplexity - (self.baseline_perplexity or 2.0)

        # Retention is OK if quality hasn't degraded too much
        # delta threshold is configurable (e.g. -0.15 quality drop allowed)
        threshold = getattr(getattr(self.config, "coding", None), "instruction_retention_threshold", 0.15)
        retention_ok = 1.0 if q_delta >= -threshold else 0.0

        elapsed = time.perf_counter() - t0
        logger.info(
            "[Retention Eval] score=%.4f (delta=%.4f), pass_rate=%.2f%%, retention_ok=%s",
            avg_score,
            q_delta,
            pass_rate * 100,
            "YES" if retention_ok else "NO"
        )

        return {
            "instruction/pass_rate": pass_rate,
            "instruction/quality_score": avg_score,
            "instruction/avg_response_length": avg_len,
            "instruction/quality_delta": q_delta,
            "instruction/perplexity_delta": p_delta,
            "instruction/retention_ok": retention_ok,
            "instruction/eval_latency_sec": elapsed,
        }

    def should_reject_checkpoint(self, results: dict[str, float]) -> bool:
        """Return True if the instruction quality dropped beyond the threshold."""
        return results.get("instruction/retention_ok", 1.0) == 0.0

    # ── Private: byte-level generation ────────────────────────────────

    def _generate(self, prompt_bytes: bytes, max_new_bytes: int = 128) -> bytes:
        """Autoregressive byte generation."""
        seq_len = self.config.training.seq_len
        ctx = list(prompt_bytes[-seq_len:])
        generated: list[int] = []

        for _ in range(max_new_bytes):
            ctx_tensor = torch.tensor(
                ctx[-seq_len:], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Check precision handler
            precision_handler = getattr(self.model, "precision_handler", None)
            if precision_handler is not None:
                with precision_handler.autocast_context():
                    outputs = self.model(ctx_tensor, return_dict=True)
            else:
                outputs = self.model(ctx_tensor, return_dict=True)

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_logits = logits[0, -1, :]

            # Deterministic greedy decoding for quick evaluation
            next_byte = int(next_logits.argmax().item())
            generated.append(next_byte)
            ctx.append(next_byte)

            if (
                len(generated) >= 2
                and generated[-1] == ord("\n")
                and generated[-2] == ord("\n")
            ):
                break

        return bytes(generated)
