# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Evaluation report generation in Markdown, JSON, and CSV formats (Phase 2.5)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Consolidates metrics and metadata into Markdown, JSON, and CSV summaries."""

    def __init__(self, output_dir: str | Path = "reports/evaluation") -> None:
        """Initialize the ReportGenerator.

        Args:
            output_dir: Directory where report files should be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
        """Flatten a nested dictionary into a single level with separated keys."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def generate_report(
        self,
        evaluation_data: dict[str, Any],
        filename_prefix: str = "eval_summary",
    ) -> dict[str, Path]:
        """Generate evaluation reports in JSON, CSV, and Markdown.

        Args:
            evaluation_data: Combined dictionary containing metrics, metadata, and config.
            filename_prefix: Prefix used for output files.

        Returns:
            Dictionary mapping format extension (json, csv, md) to Path object.
        """
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{filename_prefix}.json"
        csv_path = self.output_dir / f"{filename_prefix}.csv"
        md_path = self.output_dir / f"{filename_prefix}.md"

        # 1. Write JSON Report
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, default=str)

        # 2. Write CSV Report
        flat_data = self._flatten_dict(evaluation_data)
        # Ensure values are simple primitives
        flat_data = {k: str(v) for k, v in flat_data.items() if not isinstance(v, (list, tuple))}
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in sorted(flat_data.items()):
                writer.writerow([k, v])

        # 3. Write Markdown Report
        md_content = self._build_markdown(evaluation_data)
        md_path.write_text(md_content, encoding="utf-8")

        return {
            "json": json_path,
            "csv": csv_path,
            "md": md_path,
        }

    def _build_markdown(self, data: dict[str, Any]) -> str:
        """Construct the Markdown report template."""
        meta = data.get("metadata", {})
        perplexity = data.get("perplexity", {})
        generation = data.get("generation", {})
        benchmark = data.get("benchmark", {})
        memory = data.get("memory", {})
        arch = data.get("architecture", {})

        md = []
        md.append("# IVERI CORE — Evaluation Report")
        md.append(f"**Generated:** {meta.get('timestamp', 'N/A')}")
        md.append(f"**Reproduction Seed:** {meta.get('random_seed', 'N/A')}")
        md.append("")

        # Metadata section
        md.append("## 1. Run Metadata")
        md.append("| Attribute | Value |")
        md.append("|---|---|")
        md.append(f"| Git Commit | `{meta.get('git_commit', 'N/A')}` |")
        md.append(f"| Architecture Version | `{meta.get('architecture_version', 'N/A')}` |")
        md.append(f"| Device / Hardware | `{meta.get('device', 'cpu')}` |")
        md.append(f"| Dtype | `{meta.get('dtype', 'float32')}` |")
        md.append(f"| PyTorch Version | `{meta.get('pytorch_version', 'N/A')}` |")
        md.append(f"| CUDA Version | `{meta.get('cuda_version', 'N/A')}` |")
        md.append(f"| Evaluation Duration | `{meta.get('evaluation_duration_seconds', 0.0):.2f}s` |")
        md.append("")

        # Language modeling section
        if perplexity:
            md.append("## 2. Language Modeling Performance")
            md.append("| Metric | Value |")
            md.append("|---|---|")
            md.append(f"| Cross Entropy Loss | `{perplexity.get('loss', 0.0):.4f}` |")
            md.append(f"| Perplexity | `{perplexity.get('perplexity', 0.0):.4f}` |")
            md.append(f"| Total Tokens evaluated | `{perplexity.get('num_tokens', 0)}` |")
            md.append(f"| Total Batches processed | `{perplexity.get('num_batches', 0)}` |")
            md.append("")

        # Generation section
        if generation:
            md.append("## 3. Generative Decoding Performance")
            md.append("| Metric | Value |")
            md.append("|---|---|")
            md.append(f"| Latency (s) | `{generation.get('latency_seconds', 0.0):.4f}s` |")
            md.append(f"| Throughput (bytes/s) | `{generation.get('bytes_per_second', 0.0):.2f} bytes/sec` |")
            md.append(f"| Average length | `{generation.get('avg_generated_length', 0.0):.1f} bytes` |")
            md.append(f"| Early exit ratio | `{generation.get('early_stopped_ratio', 0.0):.2%}` |")
            md.append("")

        # Benchmark section
        if benchmark:
            md.append("## 4. Inference Performance Benchmarks")
            md.append("| Metric | Latency / Throughput |")
            md.append("|---|---|")
            md.append(f"| Warmup Latency | `{benchmark.get('warmup_latency_ms', 0.0):.2f} ms` |")
            md.append(f"| Average Latency | `{benchmark.get('latency_mean_ms', 0.0):.2f} ms` |")
            md.append(f"| Median Latency | `{benchmark.get('latency_median_ms', 0.0):.2f} ms` |")
            md.append(f"| P90 / P95 / P99 | `{benchmark.get('latency_p90_ms', 0.0):.2f} / {benchmark.get('latency_p95_ms', 0.0):.2f} / {benchmark.get('latency_p99_ms', 0.0):.2f} ms` |")
            md.append(f"| Min / Max Latency | `{benchmark.get('latency_min_ms', 0.0):.2f} / {benchmark.get('latency_max_ms', 0.0):.2f} ms` |")
            md.append(f"| Samples / sec | `{benchmark.get('samples_per_sec', 0.0):.2f} samples/sec` |")
            md.append(f"| Tokens / sec | `{benchmark.get('tokens_per_sec', 0.0):.2f} tokens/sec` |")
            md.append(f"| Estimated FLOPs | `{benchmark.get('estimated_flops', 0.0):.2e} FLOPs` |")
            md.append(f"| Model Parameter Count | `{benchmark.get('parameter_count', 0):,}` |")
            md.append("")

        # Memory section
        if memory:
            md.append("## 5. Memory Consumption Benchmarks")
            md.append("| Resource | Allocated | Reserved / Peak |")
            md.append("|---|---|---|")
            md.append(f"| GPU Memory | `{memory.get('gpu_allocated_mb', 0.0):.1f} MB` | `{memory.get('gpu_reserved_mb', 0.0):.1f} MB / {memory.get('gpu_peak_mb', 0.0):.1f} MB` |")
            md.append(f"| CPU System RAM | `{memory.get('cpu_ram_mb', 0.0):.1f} MB` | `{memory.get('cpu_peak_ram_mb', 0.0):.1f} MB` |")
            md.append(f"| Parameter Memory | `{memory.get('parameter_mb', 0.0):.2f} MB` | - |")
            md.append(f"| Activation Memory (Est) | `{memory.get('activation_mb', 0.0):.2f} MB` | - |")
            md.append(f"| Memory Fragmentation Ratio | `{memory.get('fragmentation_ratio', 0.0):.2%}` | - |")
            md.append(f"| Memory Growth Delta | `{memory.get('growth_mb', 0.0):.2f} MB` | - |")
            md.append("")

        # Architecture Telemetry
        if arch:
            md.append("## 6. Architecture Subsystem Telemetry")

            # MoR
            mor = arch.get("mor", {})
            if mor:
                md.append("### Mixture of Recursions (MoR)")
                md.append(f"- Average Depth: `{mor.get('average_depth', 0.0):.2f}`")
                md.append(f"- Median Depth: `{mor.get('median_depth', 0.0):.2f}`")
                md.append(f"- 95th Percentile Depth: `{mor.get('p95_depth', 0.0):.2f}`")
                md.append(f"- Maximum Depth: `{mor.get('max_depth_observed', 0.0)}`")
                md.append(f"- FLOPs Saved Ratio: `{mor.get('flops_saved_ratio', 0.0):.2%}`")
                md.append("")

            # MoE
            moe = arch.get("moe", {})
            if moe:
                md.append("### Mixture of Experts (MoE)")
                md.append(f"- Expert utilization histogram: `{moe.get('expert_utilization_histogram')}`")
                md.append(f"- Unused experts count: `{moe.get('unused_experts_count')}`")
                md.append(f"- Max load / Min load: `{moe.get('max_load')} / {moe.get('min_load')}`")
                md.append(f"- Imbalance Ratio: `{moe.get('imbalance_ratio', 0.0):.3f}`")
                md.append(f"- Routing Entropy: `{moe.get('routing_entropy', 0.0):.3f}`")
                md.append(f"- Expert Collapse Score: `{moe.get('expert_collapse_score', 0.0):.3f}`")
                md.append("")

            # BLT
            blt = arch.get("blt", {})
            if blt:
                md.append("### Byte Latent Transformer (BLT)")
                md.append(f"- Average Byte Entropy: `{blt.get('average_byte_entropy', 0.0):.3f}`")
                md.append(f"- Average Patch Entropy: `{blt.get('average_patch_entropy', 0.0):.3f}`")
                md.append(f"- Average / Median Patch Count: `{blt.get('average_patch_count', 0.0):.1f} / {blt.get('median_patch_count', 0.0):.1f}`")
                md.append(f"- Compression Ratio (seq_len / patches): `{blt.get('compression_ratio', 0.0):.2f}`")
                md.append(f"- Boundary Frequency: `{blt.get('boundary_frequency', 0.0):.2%}`")
                md.append("")

            # Titans
            titans = arch.get("titans", {})
            if titans:
                md.append("### Titans Gated Neural Memory")
                md.append(f"- Gated Memory reads / writes: `{titans.get('memory_reads')} / {titans.get('memory_writes')}`")
                md.append(f"- Memory Update Norm: `{titans.get('update_norm', 0.0):.4f}`")
                md.append(f"- Memory Retrieval Norm: `{titans.get('retrieval_norm', 0.0):.4f}`")
                md.append("")

            # Mamba2
            mamba = arch.get("mamba2", {})
            if mamba:
                md.append("### Mamba2 Structured State Space Duality (SSD)")
                mamba_norms = mamba.get("hidden_state_norms", {})
                md.append(f"- Hidden State Norm (mean): `{mamba_norms.get('mean', 0.0):.4f}`")
                md.append(f"- State Update Norm: `{mamba.get('state_update_norm', 0.0):.4f}`")
                md.append(f"- State Variance: `{mamba.get('state_variance', 0.0):.4e}`")
                md.append(f"- SSM throughput: `{mamba.get('throughput_tokens_per_sec', 0.0):.2f} tokens/sec`")
                md.append("")

            # Attention & Backbone
            attn = arch.get("attention", {})
            backbone = arch.get("backbone", {})
            if attn or backbone:
                md.append("### Attention & Backbone stack")
                md.append(f"- Attention Backend Selected: `{attn.get('backend_selected', 'N/A')}`")
                md.append(f"- Backbone layer latency (list of layer runtimes): `{backbone.get('layer_latencies')}`")
                md.append(f"- Backbone residual norm (mean): `{backbone.get('residual_norms', {}).get('mean', 0.0):.4f}`")
                md.append(f"- Backbone activation norm (mean): `{backbone.get('activation_norms', {}).get('mean', 0.0):.4f}`")
                md.append("")

        return "\n".join(md)
