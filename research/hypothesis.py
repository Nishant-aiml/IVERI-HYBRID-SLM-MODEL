# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research Hypothesis Engine mapping architectural claims to statistical validation results."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResearchHypothesisEngine:
    """Validates specific research claims (H1 to H10) using statistical outputs.

    Assigns outcomes strictly from: SUPPORTED, REFUTED, INCONCLUSIVE.
    """

    def __init__(self) -> None:
        self.hypotheses = {
            "H1": {
                "desc": "Entropy Routing improves validation perplexity compared to matched vanilla hybrid baseline.",
                "null": "Entropy routing does not improve perplexity compared to vanilla hybrid baselines.",
                "alternative": "Entropy routing reduces perplexity with statistical significance (p < 0.05).",
                "metric_key": "perplexity",
            },
            "H2": {
                "desc": "Titans Neural Memory reduces long-context degradation over sequence length scaling.",
                "null": "Titans neural memory does not reduce long-context degradation over sequence length scaling.",
                "alternative": "Titans memory increases long-context recall accuracy with statistical significance.",
                "metric_key": "long_context_recall",
            },
            "H3": {
                "desc": "MoR reduces computation without harming representation quality.",
                "null": "MoR recursion depth increases perplexity or does not reduce computation.",
                "alternative": "MoR recursion improves throughput while maintaining validation perplexity.",
                "metric_key": "latency",
            },
            "H4": {
                "desc": "BLT compression improves decode throughput.",
                "null": "BLT patching does not increase throughput compared to standard byte models.",
                "alternative": "BLT patching improves decoding speed (tokens/sec) with statistical significance.",
                "metric_key": "decode_speed_tps",
            },
            "H5": {
                "desc": "IVERI converges faster than parameter/FLOP-matched Transformer baseline.",
                "null": "IVERI does not converge faster than matched Transformer baselines.",
                "alternative": "IVERI achieves lower validation loss in fewer training steps compared to matched Transformers.",
                "metric_key": "val_loss",
            },
            "H6": {
                "desc": "IVERI converges faster than parameter/FLOP-matched Mamba2 baseline.",
                "null": "IVERI does not converge faster than matched Mamba2 baselines.",
                "alternative": "IVERI achieves lower validation loss in fewer training steps compared to matched Mamba2.",
                "metric_key": "val_loss",
            },
            "H7": {
                "desc": "IVERI achieves lower Bits-Per-Byte (BPB) profile than baselines.",
                "null": "IVERI does not achieve a lower Bits-Per-Byte (BPB) profile than baselines.",
                "alternative": "IVERI achieves a statistically lower BPB compared to matched baselines.",
                "metric_key": "perplexity",
            },
            "H8": {
                "desc": "IVERI specialized coding model maintains instruction following capability.",
                "null": "Coding specialization degrades instruction-following capabilities beyond threshold.",
                "alternative": "Coding specialized models maintain general instruction follow metrics (> 95%).",
                "metric_key": "instruction_score",
            },
            "H9": {
                "desc": "Preference optimization (SimPO/DPO) improves quality without forgetting.",
                "null": "Preference alignment degrades coding or instruction capacities below threshold.",
                "alternative": "Preference tuning improves alignment win rate without regressing base metrics.",
                "metric_key": "humaneval_pass_rate",
            },
            "H10": {
                "desc": "Model scaling follows a predictable power-law.",
                "null": "Model validation loss scaling does not follow a predictable power-law.",
                "alternative": "Validation loss vs parameter size matches power-law regression (R^2 >= 0.95).",
                "metric_key": "val_loss",
            }
        }

    def evaluate_hypothesis(
        self,
        label: str,
        delta_pct: float,
        p_value: float | None,
        num_seeds: int,
        cohens_d: float | None = None,
        ci: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Assess the state of a research hypothesis.

        Outcomes: SUPPORTED, REFUTED, INCONCLUSIVE.
        """
        hyp = self.hypotheses.get(label)
        if not hyp:
            raise ValueError(f"Unknown hypothesis label: {label}")

        evidence_items = [
            f"Delta={delta_pct * 100:.2f}%",
            f"p-value={p_value}",
            f"seeds={num_seeds}"
        ]
        if cohens_d is not None:
            evidence_items.append(f"d={cohens_d:.2f}")
        if ci is not None:
            evidence_items.append(f"CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

        evidence = ", ".join(evidence_items)

        # ── Outcome classification rules ──
        # SUPPORTED: p-value < 0.05, improvement is positive (delta_pct > 0 or decrease/increase matches direction)
        # REFUTED: p-value is significant but delta goes in negative direction, or metrics degrade significantly (delta_pct < -0.05)
        # INCONCLUSIVE: p-value >= 0.05 or insufficient seeds, representing no statistical certainty
        if num_seeds < 5 or p_value is None:
            outcome = "INCONCLUSIVE"
        elif delta_pct < -0.01:
            outcome = "REFUTED"
        elif p_value < 0.05 and delta_pct > 0.0:
            outcome = "SUPPORTED"
        elif p_value >= 0.05:
            # High p-value (null hypothesis cannot be rejected)
            if delta_pct < 0.0:
                outcome = "REFUTED"
            else:
                outcome = "INCONCLUSIVE"
        else:
            outcome = "INCONCLUSIVE"

        return {
            "hypothesis_label": label,
            "description": hyp["desc"],
            "null_hypothesis": hyp["null"],
            "alternative_hypothesis": hyp["alternative"],
            "evidence": evidence,
            "outcome": outcome,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "confidence_interval": ci,
        }
