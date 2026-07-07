# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""LaTeX table generator exporting formatted strings for paper publications."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PaperTableGenerator:
    """Generates clean publication-ready LaTeX tables matching academic conventions."""

    def __init__(self) -> None:
        pass

    def generate_benchmark_table(
        self,
        iveri_metrics: dict[str, float],
        transformer_metrics: dict[str, float],
        mamba_metrics: dict[str, float],
        hybrid_metrics: dict[str, float],
    ) -> str:
        """Generate LaTeX table comparing core benchmark accuracies."""
        latex = r"""\begin{table}[t]
\centering
\caption{Downstream capability benchmarks compared under matched parameter and compute budgets. Bold indicates superior performance.}
\label{tab:capability_benchmarks}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{HumanEval (Pass@1)} & \textbf{MBPP (Pass@1)} & \textbf{GSM8K (Acc)} & \textbf{Perplexity} \\
\midrule
"""
        latex += f"Vanilla Transformer & {transformer_metrics.get('humaneval', 0.0):.2f} & {transformer_metrics.get('mbpp', 0.0):.2f} & {transformer_metrics.get('gsm8k', 0.0):.2f} & {transformer_metrics.get('perplexity', 0.0):.2f} \\\\\n"
        latex += f"Pure Mamba2 & {mamba_metrics.get('humaneval', 0.0):.2f} & {mamba_metrics.get('mbpp', 0.0):.2f} & {mamba_metrics.get('gsm8k', 0.0):.2f} & {mamba_metrics.get('perplexity', 0.0):.2f} \\\\\n"
        latex += f"Mamba-Attention Hybrid & {hybrid_metrics.get('humaneval', 0.0):.2f} & {hybrid_metrics.get('mbpp', 0.0):.2f} & {hybrid_metrics.get('gsm8k', 0.0):.2f} & {hybrid_metrics.get('perplexity', 0.0):.2f} \\\\\n"
        latex += f"\\textbf{{IVERI CORE (Ours)}} & \\textbf{{{iveri_metrics.get('humaneval', 0.0):.2f}}} & \\textbf{{{iveri_metrics.get('mbpp', 0.0):.2f}}} & \\textbf{{{iveri_metrics.get('gsm8k', 0.0):.2f}}} & \\textbf{{{iveri_metrics.get('perplexity', 0.0):.2f}}} \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_ablation_table(self, ablation_metrics: dict[str, dict[str, float]]) -> str:
        """Generate LaTeX table compiling architectural ablation outcomes."""
        latex = r"""\begin{table}[t]
\centering
\caption{Architectural ablation study evaluating contribution of model sub-components.}
\label{tab:ablation_study}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Perplexity} & \textbf{Latency (ms)} & \textbf{VRAM (MB)} \\
\midrule
"""
        for config_name, metrics in ablation_metrics.items():
            latex += f"{config_name} & {metrics.get('perplexity', 0.0):.2f} & {metrics.get('latency_ms', 0.0):.1f} & {metrics.get('vram_mb', 0.0):.1f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_efficiency_table(
        self,
        iveri_eff: dict[str, float],
        transformer_eff: dict[str, float],
        mamba_eff: dict[str, float],
    ) -> str:
        """Generate LaTeX table compiling hardware and cost efficiencies."""
        latex = r"""\begin{table}[t]
\centering
\caption{Hardware throughput, energy footprint, and cloud costs metrics.}
\label{tab:efficiency_table}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Tokens/Second} & \textbf{Tokens/Joule} & \textbf{Cost/1M Tokens (USD)} \\
\midrule
"""
        latex += f"Vanilla Transformer & {transformer_eff.get('tps', 0.0):.1f} & {transformer_eff.get('tpj', 0.0):.1f} & {transformer_eff.get('cost_per_m', 0.0):.4f} \\\\\n"
        latex += f"Pure Mamba2 & {mamba_eff.get('tps', 0.0):.1f} & {mamba_eff.get('tpj', 0.0):.1f} & {mamba_eff.get('cost_per_m', 0.0):.4f} \\\\\n"
        latex += f"\\textbf{{IVERI CORE (Ours)}} & \\textbf{{{iveri_eff.get('tps', 0.0):.1f}}} & \\textbf{{{iveri_eff.get('tpj', 0.0):.1f}}} & \\textbf{{{iveri_eff.get('cost_per_m', 0.0):.4f}}} \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex
