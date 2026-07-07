# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Publication Figure Generator module generating vector SVG, PDF, and PNG plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Try importing plotting dependencies with graceful fallbacks
_HAS_PLOT = False
try:
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_PLOT = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PaperFigureGenerator:
    """Generates vector publication-quality charts using Matplotlib.

    Publication mode is fail-closed: matplotlib must be available to emit figures.
    """

    def __init__(self, output_dir: str = "reports/phase_3_5/figures/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_PLOT:
            try:
                plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
                plt.rcParams.update({
                    "font.family": "sans-serif",
                    "font.size": 10,
                    "axes.labelsize": 11,
                    "axes.titlesize": 12,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "figure.titlesize": 14,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                })
            except Exception:
                pass

    def save_figure(self, fig: Any, filename: str) -> list[Path]:
        """Save a figure in PDF, SVG, and PNG formats."""
        if not _HAS_PLOT or fig is None:
            raise RuntimeError(
                "Publication blocked: matplotlib is required to generate publication figures."
            )
        saved_paths = []
        for ext in [".pdf", ".svg", ".png"]:
            path = self.output_dir / f"{filename}{ext}"
            try:
                fig.savefig(path, format=ext.strip("."), dpi=300, bbox_inches="tight")
            except Exception as e:
                raise RuntimeError(f"Publication blocked: failed to save figure ({e})") from e
            saved_paths.append(path)
        plt.close(fig)
        return saved_paths

    def plot_loss_curves(
        self,
        steps: list[int],
        iveri_losses: list[float],
        transformer_losses: list[float],
        mamba_losses: list[float],
        hybrid_losses: list[float],
    ) -> list[Path]:
        """Plot loss comparison curves over steps."""
        if not _HAS_PLOT:
            raise RuntimeError(
                "Publication blocked: matplotlib is required for loss curve figures."
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, iveri_losses, label="IVERI CORE (Ours)", color="#8a2be2", linewidth=2.0)
        ax.plot(steps, transformer_losses, label="Vanilla Transformer", color="#ff7f50", linestyle="--")
        ax.plot(steps, mamba_losses, label="Pure Mamba2", color="#4682b4", linestyle=":")
        ax.plot(steps, hybrid_losses, label="Mamba-Attention Hybrid", color="#3cb371", linestyle="-.")

        ax.set_title("Training Loss Convergence Comparison")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", frameon=True)
        ax.set_xlim(min(steps), max(steps))

        return self.save_figure(fig, "loss_convergence_comparison")

    def plot_radar_chart(
        self,
        categories: list[str],
        iveri_scores: list[float],
        transformer_scores: list[float],
        mamba_scores: list[float],
    ) -> list[Path]:
        """Plot a multi-dimensional radar comparison chart."""
        if not _HAS_PLOT:
            raise RuntimeError(
                "Publication blocked: matplotlib is required for radar chart figures."
            )

        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the loop
        angles += angles[:1]
        iveri_scores = iveri_scores + iveri_scores[:1]
        transformer_scores = transformer_scores + transformer_scores[:1]
        mamba_scores = mamba_scores + mamba_scores[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.plot(angles, iveri_scores, color="#8a2be2", linewidth=2, label="IVERI CORE (Ours)")
        ax.fill(angles, iveri_scores, color="#8a2be2", alpha=0.15)

        ax.plot(angles, transformer_scores, color="#ff7f50", linewidth=1.5, linestyle="--", label="Transformer")
        ax.fill(angles, transformer_scores, color="#ff7f50", alpha=0.05)

        ax.plot(angles, mamba_scores, color="#4682b4", linewidth=1.5, linestyle=":", label="Mamba2")
        ax.fill(angles, mamba_scores, color="#4682b4", alpha=0.05)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment("center")
            elif 0 < angle < np.pi:
                label.set_horizontalalignment("left")
            else:
                label.set_horizontalalignment("right")

        ax.set_title("Comparative Model Capabilities", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        return self.save_figure(fig, "capability_radar_chart")

    def plot_context_throughput_curve(
        self,
        context_lengths: list[int],
        iveri_tps: list[float],
        transformer_tps: list[float],
        mamba_tps: list[float],
    ) -> list[Path]:
        """Plot decode throughput vs context length."""
        if not _HAS_PLOT:
            raise RuntimeError(
                "Publication blocked: matplotlib is required for throughput figures."
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(context_lengths, iveri_tps, label="IVERI CORE (Ours)", color="#8a2be2", marker="o", linewidth=2.0)
        ax.plot(context_lengths, transformer_tps, label="Vanilla Transformer", color="#ff7f50", marker="s", linestyle="--")
        ax.plot(context_lengths, mamba_tps, label="Pure Mamba2", color="#4682b4", marker="^", linestyle=":")

        ax.set_title("Throughput Scalability vs Context Length")
        ax.set_xlabel("Context Length (Tokens)")
        ax.set_ylabel("Throughput (Tokens/Second)")
        ax.legend(loc="lower left", frameon=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(context_lengths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        return self.save_figure(fig, "throughput_context_scaling")
