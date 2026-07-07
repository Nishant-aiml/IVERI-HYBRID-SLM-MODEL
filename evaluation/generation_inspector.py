# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Generation inspector for monitoring text convergence in IVERI CORE pretraining.

Saves prompt seeds, generated raw bytes, decoded strings, invalid UTF-8 counts,
average entropy, and speed to reports/live_training/generation_samples.md.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from core.constants import EOS_BYTE
from core.byte_vocab import ByteVocabularyError, token_ids_to_content_bytes
from evaluation.generation import GenerationEvaluator

logger = logging.getLogger(__name__)


class GenerationInspector:
    """Periodically generates samples and evaluates language/numerical safety."""

    def __init__(
        self,
        config: Any,
        log_dir: str | Path = "reports/live_training",
    ) -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.samples_file = self.log_dir / "generation_samples.md"

        self.generation_evaluator = GenerationEvaluator(eos_token_id=EOS_BYTE)

    def inspect(
        self,
        model: nn.Module,
        step: int,
        prompt_text: str = "Once upon a time, there was a little",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Perform generation, compute metrics, and append to markdown file."""
        device = next(model.parameters()).device
        model.eval()

        # Encode prompt
        prompt_bytes = prompt_text.encode("utf-8")
        input_ids = torch.tensor(list(prompt_bytes), dtype=torch.long, device=device).unsqueeze(0)

        # Stochastic or greedy run
        t0 = time.perf_counter()
        entropy_sum = 0.0
        entropy_count = 0
        curr_ids = input_ids.clone()
        finished = False

        max_new_bytes = self.config.evaluation.generation_max_new_bytes

        with torch.no_grad():
            for _ in range(max_new_bytes):
                outputs = model(curr_ids, return_dict=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                next_token_logits = logits[:, -1, :].clone()

                # Calculate step entropy before sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                step_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
                entropy_sum += step_entropy
                entropy_count += 1

                if temperature <= 1e-5:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token_logits = next_token_logits / temperature
                    probs_sampled = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs_sampled, num_samples=1)

                curr_ids = torch.cat([curr_ids, next_token], dim=-1)

                if next_token.item() == EOS_BYTE:
                    finished = True
                    break

        elapsed = time.perf_counter() - t0
        generated_ids = curr_ids[0, len(prompt_bytes) :].cpu().tolist()

        try:
            generated_bytes = token_ids_to_content_bytes(generated_ids)
        except ByteVocabularyError:
            generated_bytes = bytes(b for b in generated_ids if 0 <= b < 256)

        # Decode string and check for invalid UTF-8 replacement chars (\ufffd)
        decoded_text = generated_bytes.decode("utf-8", errors="replace")
        invalid_utf8_count = decoded_text.count("\ufffd")

        # Average entropy
        avg_entropy = entropy_sum / max(1, entropy_count)
        gen_speed = len(generated_ids) / elapsed if elapsed > 0 else 0.0

        # Repetition check (heuristic to check collapse)
        # Check if same byte pattern of length 3 is repeated excessive times
        is_repetitive = False
        if len(generated_ids) >= 12:
            quads = [tuple(generated_ids[i : i + 3]) for i in range(len(generated_ids) - 2)]
            if len(quads) > 0:
                most_freq = max(quads.count(q) for q in set(quads))
                if most_freq > len(quads) * 0.4:
                    is_repetitive = True
                    logger.warning(
                        f"[GenerationInspector] WARNING: Step {step} generation exhibits high repetition collapse."
                    )

        # Log statistics
        results = {
            "step": step,
            "prompt": prompt_text,
            "raw_bytes": generated_ids,
            "decoded_text": decoded_text,
            "invalid_utf8_count": invalid_utf8_count,
            "generation_speed_bytes_per_sec": gen_speed,
            "average_entropy": avg_entropy,
            "repetition_collapse": is_repetitive,
        }

        # Write to Markdown
        self._write_markdown(step, results, temperature, seed)

        return results

    def _write_markdown(self, step: int, results: dict[str, Any], temp: float, seed: int) -> None:
        """Append entry to generation_samples.md."""
        file_exists = self.samples_file.exists()

        lines = []
        if not file_exists:
            lines.extend([
                "# IVERI CORE — Pretraining Generation Inspection Log",
                "",
                "This file documents the evolution of model generation samples over training steps.",
                "",
            ])

        lines.extend([
            f"## Step {step}",
            f"- **Temperature**: {temp}",
            f"- **Seed**: {seed}",
            f"- **Speed**: {results['generation_speed_bytes_per_sec']:.2f} bytes/sec",
            f"- **Average Entropy**: {results['average_entropy']:.4f}",
            f"- **Invalid UTF-8 Count**: {results['invalid_utf8_count']}",
            f"- **Repetition Collapse**: {results['repetition_collapse']}",
            "",
            "### Prompt",
            f"> {results['prompt']}",
            "",
            "### Generated Output",
            "```text",
            results["decoded_text"],
            "```",
            "",
            "---",
            "",
        ])

        with open(self.samples_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
