# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Physical ablation verification audit (Phase 6.3.1F / 6.3.2 OBJ4).

Proves that ``ModelConfig`` boolean ablation flags remove components from the
forward path, that each ablation yields a distinct architecture fingerprint,
and detects unused flags, dead configuration, and silent fallback paths.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig, ModelConfig, apply_ablation_overrides, get_base_config
from model.backbone import Backbone
from model.iveri_core import IVERIModel
from model.moe.router import SparseMoERouter
from research.campaign_runner import ABLATION_CONFIG_OVERRIDES


@dataclass
class AblationProbeResult:
    ablation_key: str
    flag_field: str
    flag_value: bool
    component_absent: bool
    parameter_delta: int
    output_diff_l1: float
    proof: str


@dataclass
class ArchitectureFingerprint:
    label: str
    flags: dict[str, bool]
    param_count: int
    has_titans: bool
    has_entropy_model: bool
    has_moe_router: bool
    has_dense_ffn: bool
    mor_active: bool
    output_checksum: float

    def signature(self) -> tuple[Any, ...]:
        return (
            self.param_count,
            self.has_titans,
            self.has_entropy_model,
            self.has_moe_router,
            self.has_dense_ffn,
            self.mor_active,
            round(self.output_checksum, 8),
        )


@dataclass
class AntipatternFinding:
    category: str
    severity: str
    detail: str


@dataclass
class AblationAuditResult:
    protocol_version: str = "Phase-6.3.1F"
    timestamp_utc: str = ""
    device: str = "cpu"
    production_verdict: str = "UNKNOWN"
    probes: list[AblationProbeResult] = field(default_factory=list)
    campaign_overrides_applied: bool = False
    fingerprints: list[ArchitectureFingerprint] = field(default_factory=list)
    pairwise_distinct: bool = False
    antipatterns: list[AntipatternFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _forward_bytes(model: IVERIModel, batch: int = 2, seq: int = 16) -> torch.Tensor:
    raw = torch.randint(0, 256, (batch, seq))
    with torch.no_grad():
        out = model(raw, return_dict=False)
    assert isinstance(out, torch.Tensor)
    return out


class _CallCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self._originals: dict[str, Callable[..., Any]] = {}

    def wrap(self, obj: object, method: str, label: str | None = None) -> None:
        key = label or method
        original = getattr(obj, method)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.counts[key] += 1
            return original(*args, **kwargs)

        self._originals[key] = original
        setattr(obj, method, wrapped)

    def restore(self, obj: object, method: str, label: str | None = None) -> None:
        key = label or method
        if key in self._originals:
            setattr(obj, method, self._originals[key])


def probe_no_titans(cfg: IVERIConfig) -> AblationProbeResult:
    full_cfg = build_mini_config()
    ablated_cfg = build_mini_config()
    ablated_cfg.model.use_titans = False

    full = IVERIModel(full_cfg)
    ablated = IVERIModel(ablated_cfg)
    full_out = _forward_bytes(full)
    abl_out = _forward_bytes(ablated)

    counter = _CallCounter()
    if ablated.backbone.titans is not None:
        counter.wrap(ablated.backbone.titans, "forward_with_injection")
    _ = _forward_bytes(ablated)
    titans_calls = counter.counts.get("forward_with_injection", 0)

    return AblationProbeResult(
        ablation_key="no_titans",
        flag_field="use_titans",
        flag_value=False,
        component_absent=ablated.backbone.titans is None and titans_calls == 0,
        parameter_delta=_count_params(full) - _count_params(ablated),
        output_diff_l1=float((full_out - abl_out).abs().mean().item()),
        proof="Backbone skips TitansMemory construction and forward_with_injection when use_titans=False.",
    )


def probe_no_blt(cfg: IVERIConfig) -> AblationProbeResult:
    full = IVERIModel(build_mini_config())
    ablated_cfg = build_mini_config()
    ablated_cfg.model.use_blt = False
    ablated = IVERIModel(ablated_cfg)

    counter = _CallCounter()
    if hasattr(ablated, "entropy_model"):
        counter.wrap(ablated.entropy_model, "forward")
    _ = _forward_bytes(ablated)
    entropy_calls = counter.counts.get("forward", 0)

    full_out = _forward_bytes(full)
    abl_out = _forward_bytes(ablated)

    return AblationProbeResult(
        ablation_key="no_blt",
        flag_field="use_blt",
        flag_value=False,
        component_absent=not hasattr(ablated, "entropy_model") and entropy_calls == 0,
        parameter_delta=_count_params(full) - _count_params(ablated),
        output_diff_l1=float((full_out - abl_out).abs().mean().item()),
        proof="IVERIModel uses byte_embed + bypass_lm_head instead of BLT stack when use_blt=False.",
    )


def probe_no_mor(cfg: IVERIConfig) -> AblationProbeResult:
    full_cfg = build_mini_config()
    ablated_cfg = build_mini_config()
    ablated_cfg.model.use_mor = False

    full = Backbone(full_cfg)
    ablated = Backbone(ablated_cfg)
    x = torch.randn(2, 6, full_cfg.model.hidden_dim)
    entropy = torch.rand(2, 6, 1)

    counter = _CallCounter()
    for block in ablated.blocks:
        counter.wrap(block.recursion_engine, "forward", label="recursion_engine.forward")
    with torch.no_grad():
        _ = ablated(x, entropy=entropy)
    recursion_calls = counter.counts.get("recursion_engine.forward", 0)

    with torch.no_grad():
        full_out = full(x, entropy=entropy)
        abl_out = ablated(x, entropy=entropy)

    return AblationProbeResult(
        ablation_key="no_mor",
        flag_field="use_mor",
        flag_value=False,
        component_absent=recursion_calls == 0,
        parameter_delta=_count_params(full) - _count_params(ablated),
        output_diff_l1=float((full_out - abl_out).abs().mean().item()),
        proof="BackboneBlock calls sub_block once without RecursionEngine when use_mor=False.",
    )


def probe_no_moe(cfg: IVERIConfig) -> AblationProbeResult:
    full_cfg = build_mini_config()
    ablated_cfg = build_mini_config()
    ablated_cfg.model.use_moe = False

    full = Backbone(full_cfg)
    ablated = Backbone(ablated_cfg)
    x = torch.randn(2, 6, full_cfg.model.hidden_dim)
    entropy = torch.rand(2, 6, 1)

    moe_present_full = any(block.sub_block.use_moe for block in full.blocks)
    moe_present_ablated = any(block.sub_block.use_moe for block in ablated.blocks)

    with torch.no_grad():
        full_out = full(x, entropy=entropy)
        abl_out = ablated(x, entropy=entropy)

    return AblationProbeResult(
        ablation_key="no_moe",
        flag_field="use_moe",
        flag_value=False,
        component_absent=moe_present_full and not moe_present_ablated,
        parameter_delta=_count_params(full) - _count_params(ablated),
        output_diff_l1=float((full_out - abl_out).abs().mean().item()),
        proof="BackboneSubBlock uses dense SwiGLU FFN instead of SparseMoERouter when use_moe=False.",
    )


def probe_no_entropy_routing(cfg: IVERIConfig) -> AblationProbeResult:
    router = SparseMoERouter(build_mini_config())
    router.eval()
    router.noise_enabled = False

    hidden = torch.randn(2, 6, router.hidden_dim)
    ent_a = torch.zeros(2, 6, 1)
    ent_b = torch.ones(2, 6, 1)

    with torch.no_grad():
        logits_a, _, _, _, _ = router._gating_logits(hidden, entropy=ent_a)
        logits_b, _, _, _, _ = router._gating_logits(hidden, entropy=ent_b)
    full_diff = float((logits_a - logits_b).abs().max().item())

    ablated_cfg = build_mini_config()
    ablated_cfg.model.use_entropy_routing = False
    router_ablated = SparseMoERouter(ablated_cfg)
    router_ablated.eval()
    router_ablated.noise_enabled = False

    with torch.no_grad():
        logits_a2, _, _, _, _ = router_ablated._gating_logits(hidden, entropy=ent_a)
        logits_b2, _, _, _, _ = router_ablated._gating_logits(hidden, entropy=ent_b)
    ablated_diff = float((logits_a2 - logits_b2).abs().max().item())

    return AblationProbeResult(
        ablation_key="no_entropy_routing",
        flag_field="use_entropy_routing",
        flag_value=False,
        component_absent=full_diff > 0.0 and ablated_diff == 0.0,
        parameter_delta=0,
        output_diff_l1=ablated_diff,
        proof=(
            "SparseMoERouter ignores entropy bias when use_entropy_routing=False "
            f"(full_diff={full_diff:.3e}, ablated_diff={ablated_diff:.3e})."
        ),
    )


def _ablation_flag_fields() -> list[str]:
    return [f for f in ModelConfig.__dataclass_fields__ if f.startswith("use_")]


def _config_for_ablation(ablation_key: str) -> IVERIConfig:
    cfg = build_mini_config()
    overrides = ABLATION_CONFIG_OVERRIDES.get(ablation_key, {})
    apply_ablation_overrides(cfg, overrides)
    return cfg


def fingerprint_architecture(label: str, cfg: IVERIConfig) -> ArchitectureFingerprint:
    """Structural + output fingerprint for pairwise architecture comparison."""
    torch.manual_seed(42)
    model = IVERIModel(cfg)
    backbone = model.backbone
    block = backbone.blocks[0].sub_block

    raw = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out = model(raw, return_dict=False)
    assert isinstance(out, torch.Tensor)

    flags = {f: bool(getattr(cfg.model, f)) for f in _ablation_flag_fields()}
    return ArchitectureFingerprint(
        label=label,
        flags=flags,
        param_count=_count_params(model),
        has_titans=backbone.titans is not None,
        has_entropy_model=hasattr(model, "entropy_model"),
        has_moe_router=hasattr(block, "moe_router") and block.use_moe,
        has_dense_ffn=hasattr(block, "dense_ffn") and not block.use_moe,
        mor_active=cfg.model.use_mor,
        output_checksum=float(out.abs().sum().item()),
    )


def collect_architecture_fingerprints() -> list[ArchitectureFingerprint]:
    keys = ["none"] + [k for k in ABLATION_CONFIG_OVERRIDES if k != "none"]
    return [fingerprint_architecture(k, _config_for_ablation(k)) for k in keys]


def verify_pairwise_distinct(fingerprints: list[ArchitectureFingerprint]) -> tuple[bool, list[str]]:
    collisions: list[str] = []
    for i, a in enumerate(fingerprints):
        for b in fingerprints[i + 1 :]:
            if a.signature() == b.signature():
                collisions.append(f"{a.label} ≡ {b.label} (identical architecture signature)")
    return len(collisions) == 0, collisions


def detect_antipatterns(
    probes: list[AblationProbeResult],
    fingerprints: list[ArchitectureFingerprint],
) -> list[AntipatternFinding]:
    findings: list[AntipatternFinding] = []

    campaign_fields = {
        field for overrides in ABLATION_CONFIG_OVERRIDES.values() for field in overrides
    }
    for field in _ablation_flag_fields():
        if field not in campaign_fields:
            findings.append(
                AntipatternFinding(
                    "unused_flag",
                    "HIGH",
                    f"ModelConfig.{field} is not referenced by any ABLATION_CONFIG_OVERRIDES entry.",
                )
            )

    for p in probes:
        if not p.component_absent and p.ablation_key != "no_entropy_routing":
            findings.append(
                AntipatternFinding(
                    "silent_fallback",
                    "CRITICAL",
                    f"{p.ablation_key}: flag {p.flag_field}=False but component still active on forward path.",
                )
            )
        if p.parameter_delta == 0 and p.output_diff_l1 == 0.0 and p.ablation_key not in {
            "no_entropy_routing",
            "no_mor",
        }:
            findings.append(
                AntipatternFinding(
                    "dead_configuration",
                    "HIGH",
                    f"{p.ablation_key}: toggling {p.flag_field} changes neither parameters nor outputs.",
                )
            )

    baseline = next((f for f in fingerprints if f.label == "none"), None)
    for fp in fingerprints:
        if fp.label == "none":
            continue
        if baseline and fp.signature() == baseline.signature():
            findings.append(
                AntipatternFinding(
                    "dead_configuration",
                    "CRITICAL",
                    f"Ablation '{fp.label}' produces identical architecture signature to baseline.",
                )
            )

    return findings


def verify_campaign_override_application() -> bool:
    cfg = get_base_config()
    apply_ablation_overrides(cfg, {"use_titans": False})
    return cfg.model.use_titans is False


def run_ablation_audit() -> AblationAuditResult:
    cfg = build_mini_config()
    probes = [
        probe_no_titans(cfg),
        probe_no_blt(cfg),
        probe_no_mor(cfg),
        probe_no_moe(cfg),
        probe_no_entropy_routing(cfg),
    ]
    fingerprints = collect_architecture_fingerprints()
    distinct, _ = verify_pairwise_distinct(fingerprints)
    antipatterns = detect_antipatterns(probes, fingerprints)
    physical_ok = all(p.component_absent for p in probes) and verify_campaign_override_application()
    architecture_ok = distinct and not any(a.severity == "CRITICAL" for a in antipatterns)
    all_pass = physical_ok and architecture_ok
    return AblationAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        device="cpu",
        production_verdict="PASS" if all_pass else "FAIL",
        probes=probes,
        campaign_overrides_applied=verify_campaign_override_application(),
        fingerprints=fingerprints,
        pairwise_distinct=distinct,
        antipatterns=antipatterns,
    )


def render_ablation_verification_report(result: AblationAuditResult) -> str:
    lines = [
        "# Ablation Verification Report (Phase 6.3.1F)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        f"**Device:** {result.device}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Physical ablation framework:** `{result.production_verdict}`",
        "",
        f"**Campaign overrides apply to ModelConfig:** `{result.campaign_overrides_applied}`",
        "",
        f"**Pairwise distinct architectures:** `{result.pairwise_distinct}`",
        "",
        "## Ablation Probes",
        "",
        "| Ablation | Flag | Absent from forward path | Param Δ | Output Δ (L1) |",
        "|----------|------|:------------------------:|--------:|----------------:|",
    ]
    for p in result.probes:
        lines.append(
            f"| {p.ablation_key} | `{p.flag_field}={p.flag_value}` | "
            f"{p.component_absent} | {p.parameter_delta} | {p.output_diff_l1:.4e} |"
        )

    lines.extend(["", "## Architecture Fingerprints", ""])
    lines.append(
        "| Config | Params | Titans | BLT | MoE router | Dense FFN | MoR | Output Σ |"
    )
    lines.append("|--------|-------:|:------:|:---:|:----------:|:---------:|:---:|---------:|")
    for fp in result.fingerprints:
        lines.append(
            f"| {fp.label} | {fp.param_count} | {fp.has_titans} | {fp.has_entropy_model} | "
            f"{fp.has_moe_router} | {fp.has_dense_ffn} | {fp.mor_active} | {fp.output_checksum:.4e} |"
        )

    lines.extend(["", "## Pairwise Distinctness", ""])
    distinct, collisions = verify_pairwise_distinct(result.fingerprints)
    if distinct:
        lines.append("All baseline + ablation configurations produce unique architecture signatures.")
    else:
        for c in collisions:
            lines.append(f"- **COLLISION:** {c}")

    lines.extend(["", "## Antipattern Detection", ""])
    if not result.antipatterns:
        lines.append("No unused flags, dead configuration, or silent fallback detected.")
    else:
        lines.append("| Category | Severity | Detail |")
        lines.append("|----------|----------|--------|")
        for a in result.antipatterns:
            lines.append(f"| {a.category} | {a.severity} | {a.detail} |")

    lines.extend(["", "## Proof Statements", ""])
    for i, p in enumerate(result.probes, 1):
        lines.append(f"{i}. **{p.ablation_key}:** {p.proof}")

    lines.extend(
        [
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


def write_ablation_verification_report(output_path: str | Path) -> AblationAuditResult:
    result = run_ablation_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ablation_verification_report(result), encoding="utf-8")
    return result
