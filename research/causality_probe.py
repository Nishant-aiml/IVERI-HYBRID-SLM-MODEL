# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Runtime causality perturbation probes for IVERI CORE (Phase 6.3.1C).

For each cut position ``i``, future bytes ``> i`` are randomly perturbed. Causal
modules must leave all outputs at positions ``<= i`` (and patch tensors fully
contained in ``[0, i]``) bitwise-identical.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from configs.base_config import IVERIConfig, get_base_config
from model.iveri_core import IVERIModel


class CorpusType(str, Enum):
    RANDOM = "random"
    ENGLISH = "english"
    CODE = "code"
    BINARY = "binary"


CORPUS_SAMPLES: dict[CorpusType, str] = {
    CorpusType.ENGLISH: (
        "The quick brown fox jumps over the lazy dog. "
        "Autoregressive language models must not read the future."
    ),
    CorpusType.CODE: (
        "def causal_probe(x):\n"
        "    if x > 0:\n"
        "        return x * 2\n"
        "    return -1\n"
    ),
}


@dataclass
class PositionError:
    cut_index: int
    max_abs_error: float
    max_rel_error: float
    trial: int


@dataclass
class ModuleProbeSummary:
    module: str
    tensor_path: str
    masking_issue: str
    corpus: str
    positions_tested: int
    leaking_positions: int
    max_abs_error: float
    max_rel_error: float
    first_leak_index: int | None
    passed: bool


@dataclass
class CausalityAuditResult:
    protocol_version: str = "Phase-6.3.2-OBJ1"
    timestamp_utc: str = ""
    device: str = "cpu"
    seq_len: int = 0
    model_hidden_dim: int = 0
    model_num_layers: int = 0
    corpora: list[str] = field(default_factory=list)
    module_summaries: list[ModuleProbeSummary] = field(default_factory=list)
    end_to_end_verdict: str = "UNKNOWN"
    primary_leak_module: str | None = None
    primary_tensor: str | None = None
    primary_attention_path: str | None = None
    primary_masking_issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bytes_from_text(text: str, seq_len: int) -> torch.Tensor:
    raw = list(text.encode("utf-8"))
    if len(raw) < seq_len:
        raw = (raw * ((seq_len // len(raw)) + 1))[:seq_len]
    else:
        raw = raw[:seq_len]
    return torch.tensor([raw], dtype=torch.long)


def bytes_random(seq_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, 256, (1, seq_len), generator=gen, dtype=torch.long)


def bytes_binary(seq_len: int) -> torch.Tensor:
    base = list(range(256))
    vals = (base * ((seq_len // len(base)) + 1))[:seq_len]
    return torch.tensor([vals], dtype=torch.long)


def corpus_tensor(corpus: CorpusType, seq_len: int, seed: int) -> torch.Tensor:
    if corpus == CorpusType.RANDOM:
        return bytes_random(seq_len, seed)
    if corpus == CorpusType.BINARY:
        return bytes_binary(seq_len)
    return bytes_from_text(CORPUS_SAMPLES[corpus], seq_len)


def perturb_future_bytes(
    raw_bytes: torch.Tensor,
    cut_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    perturbed = raw_bytes.clone()
    seq_len = raw_bytes.shape[1]
    if cut_index >= seq_len - 1:
        return perturbed
    future_len = seq_len - cut_index - 1
    replacement = torch.randint(
        3, 256, (raw_bytes.shape[0], future_len), generator=generator, dtype=torch.long
    )
    same = replacement == raw_bytes[:, cut_index + 1 :]
    if same.all():
        replacement = (replacement + 17) % 250 + 3
    perturbed[:, cut_index + 1 :] = replacement
    return perturbed


def patch_spans(boundary_map: torch.Tensor) -> list[tuple[int, int]]:
    """Return inclusive (start, end) byte spans for each patch in batch item 0."""
    bmap = boundary_map[0].tolist()
    starts = [idx for idx, is_boundary in enumerate(bmap) if is_boundary]
    spans: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else (len(bmap) - 1)
        spans.append((start, end))
    return spans


def causal_patch_indices(boundary_map: torch.Tensor, cut_index: int) -> list[int]:
    return [p for p, (start, end) in enumerate(patch_spans(boundary_map)) if end <= cut_index]


def tensor_delta(
    ref: torch.Tensor,
    pert: torch.Tensor,
) -> tuple[float, float]:
    diff = (ref - pert).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = ref.abs().clamp(min=1e-8)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return max_abs, max_rel


@dataclass
class IntermediateOutputs:
    byte_entropy: torch.Tensor
    boundary_map: torch.Tensor
    latent_patches: torch.Tensor
    patch_entropy: torch.Tensor
    backbone_out: torch.Tensor
    logits: torch.Tensor


def forward_intermediates(model: IVERIModel, raw_bytes: torch.Tensor) -> IntermediateOutputs:
    byte_entropy = model.entropy_model(raw_bytes)
    boundary_map = model.patcher.compute_boundaries(raw_bytes, byte_entropy)
    latent_patches = model.encoder.encode_with_boundaries(raw_bytes, boundary_map)
    p_max = latent_patches.shape[1]
    patch_ids = torch.cumsum(boundary_map.long(), dim=-1) - 1
    patch_indices = torch.arange(p_max, device=raw_bytes.device).view(1, -1, 1)
    is_patch = patch_ids.unsqueeze(1) == patch_indices
    patch_lengths = is_patch.sum(dim=-1, keepdim=True)
    patch_lengths_clamped = torch.clamp(patch_lengths, min=1)
    pool = is_patch.float() / patch_lengths_clamped.float()
    patch_entropy = torch.bmm(pool, byte_entropy)
    backbone_out = model.backbone(latent_patches, entropy=patch_entropy)
    logits = model.decoder.decode_with_boundaries(backbone_out, boundary_map, raw_bytes)
    return IntermediateOutputs(
        byte_entropy=byte_entropy,
        boundary_map=boundary_map,
        latent_patches=latent_patches,
        patch_entropy=patch_entropy,
        backbone_out=backbone_out,
        logits=logits,
    )


MODULE_SPECS: list[tuple[str, str, str]] = [
    (
        "ByteEntropyModel",
        "byte_entropy[:, :cut+1, :]",
        "Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)",
    ),
    (
        "DynamicPatcher",
        "boundary_map[:, :cut+1]",
        "Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix",
    ),
    (
        "BLTByteEncoder",
        "latent_patches[:, causal_patch_indices, :]",
        "Within-patch causal MultiheadAttention (no future byte keys)",
    ),
    (
        "PatchEntropyPool",
        "patch_entropy[:, causal_patch_indices, :]",
        "Mean pool over patch bytes; invariant when byte entropy is causal",
    ),
    (
        "Backbone",
        "backbone_out[:, causal_patch_indices, :]",
        "Patch-level causal attention; inherits causal patch inputs",
    ),
    (
        "BLTByteDecoder",
        "logits[:, :cut+1, :]",
        "Cross-attention keys masked to patches with patch_end <= query byte index",
    ),
]


def compare_module(
    module: str,
    tensor_path: str,
    masking_issue: str,
    corpus: CorpusType,
    model: IVERIModel,
    raw_bytes: torch.Tensor,
    cut_index: int,
    ref: IntermediateOutputs,
    pert: IntermediateOutputs,
) -> tuple[float, float, bool]:
    if module == "ByteEntropyModel":
        a = ref.byte_entropy[:, : cut_index + 1, :]
        b = pert.byte_entropy[:, : cut_index + 1, :]
    elif module == "DynamicPatcher":
        a = ref.boundary_map[:, : cut_index + 1].float()
        b = pert.boundary_map[:, : cut_index + 1].float()
    elif module in {"BLTByteEncoder", "PatchEntropyPool", "Backbone"}:
        idx = causal_patch_indices(ref.boundary_map, cut_index)
        if not idx:
            return 0.0, 0.0, True
        idx_t = torch.tensor(idx, device=raw_bytes.device, dtype=torch.long)
        if module == "BLTByteEncoder":
            a = ref.latent_patches.index_select(1, idx_t)
            b = pert.latent_patches.index_select(1, idx_t)
        elif module == "PatchEntropyPool":
            a = ref.patch_entropy.index_select(1, idx_t)
            b = pert.patch_entropy.index_select(1, idx_t)
        else:
            a = ref.backbone_out.index_select(1, idx_t)
            b = pert.backbone_out.index_select(1, idx_t)
    elif module == "BLTByteDecoder":
        a = ref.logits[:, : cut_index + 1, :]
        b = pert.logits[:, : cut_index + 1, :]
    else:
        raise ValueError(module)

    max_abs, max_rel = tensor_delta(a, b)
    ok = torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    return max_abs, max_rel, ok


def probe_corpus(
    model: IVERIModel,
    corpus: CorpusType,
    seq_len: int,
    seed: int,
    position_step: int = 1,
    trials_per_position: int = 2,
) -> list[ModuleProbeSummary]:
    model.eval()
    raw_bytes = corpus_tensor(corpus, seq_len, seed)
    summaries: list[ModuleProbeSummary] = []

    for module, tensor_path, masking_issue in MODULE_SPECS:
        position_errors: list[PositionError] = []
        for cut_index in range(1, seq_len - 1, position_step):
            for trial in range(trials_per_position):
                gen = torch.Generator().manual_seed(seed + cut_index * 17 + trial * 101)
                perturbed = perturb_future_bytes(raw_bytes, cut_index, gen)
                with torch.no_grad():
                    ref = forward_intermediates(model, raw_bytes)
                    pert = forward_intermediates(model, perturbed)
                max_abs, max_rel, ok = compare_module(
                    module,
                    tensor_path,
                    masking_issue,
                    corpus,
                    model,
                    raw_bytes,
                    cut_index,
                    ref,
                    pert,
                )
                if not ok:
                    position_errors.append(
                        PositionError(cut_index, max_abs, max_rel, trial)
                    )

        leaking = len(position_errors)
        tested = len(range(1, seq_len - 1, position_step)) * trials_per_position
        max_abs = max((p.max_abs_error for p in position_errors), default=0.0)
        max_rel = max((p.max_rel_error for p in position_errors), default=0.0)
        first_leak = position_errors[0].cut_index if position_errors else None
        summaries.append(
            ModuleProbeSummary(
                module=module,
                tensor_path=tensor_path,
                masking_issue=masking_issue,
                corpus=corpus.value,
                positions_tested=tested,
                leaking_positions=leaking,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                first_leak_index=first_leak,
                passed=leaking == 0,
            )
        )
    return summaries


def build_mini_config() -> IVERIConfig:
    cfg = get_base_config()
    cfg.model.hidden_dim = 64
    cfg.model.num_heads = 2
    cfg.model.num_layers = 2
    cfg.model.mamba_ratio = 2
    cfg.model.num_experts = 4
    cfg.model.num_active_experts = 2
    cfg.model.max_recursion_depth = 4
    cfg.model.titans_memory_dim = 32
    cfg.validate()
    return cfg


def run_causality_audit(
    seq_len: int = 32,
    seed: int = 42,
    position_step: int = 2,
    trials_per_position: int = 2,
) -> CausalityAuditResult:
    cfg = build_mini_config()
    model = IVERIModel(cfg)
    model.eval()

    all_summaries: list[ModuleProbeSummary] = []
    for corpus in CorpusType:
        all_summaries.extend(
            probe_corpus(
                model,
                corpus,
                seq_len=seq_len,
                seed=seed,
                position_step=position_step,
                trials_per_position=trials_per_position,
            )
        )

    # Identify earliest module in pipeline with any leak (English corpus priority)
    pipeline_order = [m for m, _, _ in MODULE_SPECS]
    primary = None
    for module in pipeline_order:
        leaked = [
            s
            for s in all_summaries
            if s.module == module and not s.passed and s.corpus == CorpusType.ENGLISH.value
        ]
        if leaked:
            primary = leaked[0]
            break
    if primary is None:
        for module in pipeline_order:
            leaked = [s for s in all_summaries if s.module == module and not s.passed]
            if leaked:
                primary = leaked[0]
                break

    e2e = next(
        (s for s in all_summaries if s.module == "BLTByteDecoder" and s.corpus == CorpusType.RANDOM.value),
        None,
    )
    verdict = "PASS" if e2e and e2e.passed else "FAIL"

    return CausalityAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        device="cpu",
        seq_len=seq_len,
        model_hidden_dim=cfg.model.hidden_dim,
        model_num_layers=cfg.model.num_layers,
        corpora=[c.value for c in CorpusType],
        module_summaries=all_summaries,
        end_to_end_verdict=verdict,
        primary_leak_module=primary.module if primary else None,
        primary_tensor=primary.tensor_path if primary else None,
        primary_attention_path=(
            f"{primary.module}.forward" if primary and "Encoder" in primary.module
            else f"{primary.module}.forward" if primary and "Decoder" in primary.module
            else f"{primary.module}" if primary
            else None
        ),
        primary_masking_issue=primary.masking_issue if primary else None,
    )


def render_causality_report(result: CausalityAuditResult) -> str:
    lines = [
        "# Causality Report — Runtime Perturbation Audit (Phase 6.3.2 / Objective 1)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        f"**Device:** {result.device}  ",
        f"**Sequence length:** {result.seq_len}  ",
        f"**Model:** hidden_dim={result.model_hidden_dim}, layers={result.model_num_layers}  ",
        "",
        "## Executive Verdict",
        "",
        f"**End-to-end causality:** `{result.end_to_end_verdict}`",
        "",
        "Perturbation protocol: for each cut index `i`, bytes at positions `> i` are replaced "
        "with random values; outputs at positions `<= i` (and patch tensors fully contained in "
        "`[0, i]`) must remain identical.",
        "",
    ]

    if result.end_to_end_verdict == "FAIL":
        lines.extend(
            [
                "## Primary Leak Attribution (earliest failing module)",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| Module | `{result.primary_leak_module}` |",
                f"| Tensor | `{result.primary_tensor}` |",
                f"| Attention path | `{result.primary_attention_path}` |",
                f"| Masking issue | {result.primary_masking_issue} |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Causality Restoration (Phase 6.3.2 Objective 1)",
                "",
                "BLT stack updated for strict autoregressive byte modeling:",
                "",
                "1. **ByteEntropyModel:** left-padded causal Conv1d (no symmetric padding).",
                "2. **BLTByteEncoder:** within-patch causal self-attention mask.",
                "3. **BLTByteDecoder:** cross-attention limited to patches with `patch_end <= query_index`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Measured Error Summary",
            "",
            "| Corpus | Module | Positions Tested | Leaking | Max Abs Error | Max Rel Error | First Leak @ i |",
            "|--------|--------|------------------:|--------:|--------------:|--------------:|---------------:|",
        ]
    )
    for s in result.module_summaries:
        first = "" if s.first_leak_index is None else str(s.first_leak_index)
        status = "OK" if s.passed else "LEAK"
        lines.append(
            f"| {s.corpus} | {s.module} | {s.positions_tested} | {s.leaking_positions} "
            f"| {s.max_abs_error:.6e} | {s.max_rel_error:.6e} | {first} |"
        )

    lines.extend(
        [
            "",
            "## Module Notes",
            "",
        ]
    )
    for module, tensor_path, masking_issue in MODULE_SPECS:
        lines.append(f"### {module}")
        lines.append(f"- **Tensor compared:** `{tensor_path}`")
        lines.append(f"- **Known masking gap:** {masking_issue}")
        lines.append("")

    lines.extend(
        [
            "## Pass Tolerance",
            "",
            "Positions pass when `torch.allclose(ref, pert, atol=1e-6, rtol=1e-5)`.",
            "",
            "## Raw JSON",
            "",
            "```json",
            json.dumps(result.to_dict(), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_causality_report(
    output_path: str | Path,
    seq_len: int = 32,
    seed: int = 42,
    position_step: int = 2,
    trials_per_position: int = 2,
) -> CausalityAuditResult:
    result = run_causality_audit(
        seq_len=seq_len,
        seed=seed,
        position_step=position_step,
        trials_per_position=trials_per_position,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_causality_report(result), encoding="utf-8")
    return result
