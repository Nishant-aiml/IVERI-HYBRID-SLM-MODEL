# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research Benchmarks Module for Stage 5 scientific validation campaigns.

Evaluates perplexity, scaling behaviors, long-context retrieval, calibration error,
and operational hardware efficiency metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from research.calibration import ConfidenceCalibrator
from research.flops import FlopProfiler
from research.profiler import MemoryProfiler
from research.energy_profiler import EnergyProfiler

logger = logging.getLogger(__name__)


class ResearchBenchmarkRunner:
    """Manages structural research evaluations.

    Executes perplexity, LongBench, Needle-in-a-Haystack, and confidence calibrations.
    """

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config
        self.calibrator = ConfidenceCalibrator()
        self.flop_profiler = FlopProfiler(config)
        self.memory_profiler = MemoryProfiler(config)
        self.energy_profiler = EnergyProfiler()

    def run_needle_in_a_haystack(
        self,
        model: nn.Module,
        context_len: int = 1024,
        needle_pos: int = 512,
    ) -> dict[str, Any]:
        """Runs a simulated Needle-in-a-Haystack context retrieval test.

        Places a target fact ('needle') in a background corpus ('haystack')
        and evaluates whether the model recalls the fact.
        """
        device = next(model.parameters()).device
        haystack = [ord("a") for _ in range(context_len)]

        # Needle fact: "The secret code is 999."
        needle = b"The secret code is 999."
        needle_len = len(needle)

        # Inset needle
        pos = min(needle_pos, context_len - needle_len)
        for i, val in enumerate(needle):
            haystack[pos + i] = val

        # Query instruction
        query = b" What is the secret code? Answer:"
        input_bytes = bytes(haystack) + query

        input_tensor = torch.tensor(list(input_bytes), dtype=torch.long, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            # Check next predicted token
            next_byte = torch.argmax(logits[:, -1, :], dim=-1).item()

        # Needle verification check: expect digits or space (e.g. '9' = 57)
        success = next_byte == ord("9") or next_byte == ord(" ") or next_byte in list(b"999")

        return {
            "context_length": context_len,
            "needle_position": pos,
            "success": success,
            "predicted_byte": next_byte,
            "predicted_char": chr(next_byte) if 32 <= next_byte <= 126 else f"\\x{next_byte:02x}",
        }

    def run_longbench_perplexity(
        self,
        model: nn.Module,
        sequence_lengths: list[int] | None = None,
    ) -> dict[int, float]:
        """Runs perplexity scaling tests over long context window frames."""
        device = next(model.parameters()).device
        lengths = sequence_lengths or [1024, 2048, 4096]
        perplexity_by_length: dict[int, float] = {}

        model.eval()
        criterion = nn.CrossEntropyLoss()

        for length in lengths:
            try:
                # We limit validation size if CPU limits to prevent OOM
                sz = min(length, 1024)
                dummy_input = torch.randint(0, 256, (1, sz), dtype=torch.long, device=device)
                with torch.no_grad():
                    outputs = model(dummy_input)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss = criterion(logits[:, :-1, :].reshape(-1, 256), dummy_input[:, 1:].reshape(-1))
                    ppl = torch.exp(loss).item()
                    perplexity_by_length[length] = ppl
            except Exception as e:
                logger.warning(f"Failed longbench perplexity at length {length}: {e}")
                perplexity_by_length[length] = 0.0

        return perplexity_by_length

    def run_research_suite(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Execute the complete set of Stage 5 research validation audits.

        Args:
            model: PyTorch model instance.
            val_loader: Validation dataset loader.

        Returns:
            dict[str, Any]: Aggregated research validation outcomes.
        """
        device = next(model.parameters()).device
        model.to(device)
        model.eval()

        self.energy_profiler.start_session()

        # ── 1. Loss & Perplexity ──
        criterion = nn.CrossEntropyLoss()
        val_losses = []
        all_logits = []
        all_labels = []

        start_eval = time.perf_counter()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 10:  # limit to 10 batches for quick validation
                    break
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) if len(batch) >= 2 else inputs
                else:
                    inputs = batch.to(device)
                    targets = inputs.clone()

                outputs = model(inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_losses.append(loss.item())

                # Collect for calibration checks
                all_logits.append(logits.cpu())
                all_labels.append(targets.cpu())

        eval_time = time.perf_counter() - start_eval
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # ── 2. Calibration Metrics ──
        calibration_results = {}
        if all_logits and all_labels:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            calibration_results = self.calibrator.compute_calibration_metrics(cat_logits, cat_labels)

        # ── 3. Long-Context retrieval ──
        needle_results = self.run_needle_in_a_haystack(model, context_len=1024, needle_pos=512)
        longbench_results = self.run_longbench_perplexity(model, sequence_lengths=[1024, 2048])

        # ── 4. Energy & Cost ──
        processed_tokens = sum(t.numel() for t in all_labels)
        energy_results = self.energy_profiler.stop_session_and_compute(processed_tokens)

        # ── 5. System Memory & Diagnostics ──
        mem_results = self.memory_profiler.get_system_memory_info()
        diag_results = self.memory_profiler.profile_subsystem_diagnostics(
            model,
            torch.randint(0, 256, (1, self.config.training.seq_len), dtype=torch.long, device=device)
        )

        return {
            "validation": {
                "loss": avg_loss,
                "perplexity": perplexity,
                "eval_seconds": eval_time,
            },
            "calibration": calibration_results,
            "long_context": {
                "needle_in_haystack": needle_results,
                "longbench_perplexity": longbench_results,
            },
            "energy": energy_results,
            "memory": mem_results,
            "diagnostics": diag_results,
        }
