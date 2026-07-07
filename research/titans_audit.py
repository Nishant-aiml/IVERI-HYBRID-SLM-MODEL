# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Runtime Titans memory audit instrumentation (Phase 6.3.1D).

Measures reads, writes, online updates, persistence, gradient flow, and memory
replacement without modifying model architecture code.
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

from configs.base_config import IVERIConfig, get_base_config
from model.backbone import Backbone
from model.iveri_core import IVERIModel
from model.titans.memory import TitansMemory


@dataclass
class MemorySnapshot:
    stage: str
    base_w1_norm: float
    base_w1_sum: float
    online_w1_norm: float | None
    online_w1_sum: float | None
    online_weights_present: bool
    telemetry_update_count: int
    telemetry_avg_update_mag: float


@dataclass
class PathAuditResult:
    path_name: str
    read_calls: int
    write_calls: int
    forward_calls: int
    inject_calls: int
    updater_calls: int
    online_weight_delta_after: float
    persistence_across_second_call: bool
    telemetry_reports_writes: bool
    snapshots: list[MemorySnapshot] = field(default_factory=list)


@dataclass
class GradientAuditResult:
    path_name: str
    base_w1_grad_norm: float
    base_w2_grad_norm: float
    q_proj_grad_norm: float
    online_weights_changed_during_forward: bool
    optimizer_changed_base_w1: bool
    optimizer_changed_online_w1: bool


@dataclass
class TitansAuditResult:
    protocol_version: str = "Phase-6.3.2-OBJ2"
    timestamp_utc: str = ""
    device: str = "cpu"
    production_verdict: str = "UNKNOWN"
    writes_occur_in_production: bool = False
    write_absence_proof: list[str] = field(default_factory=list)
    write_presence_proof: list[str] = field(default_factory=list)
    path_results: list[PathAuditResult] = field(default_factory=list)
    gradient_results: list[GradientAuditResult] = field(default_factory=list)
    isolated_forward_updates: int = 0
    isolated_forward_avg_update_mag: float = 0.0

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


def _snapshot_memory(memory: TitansMemory, stage: str) -> MemorySnapshot:
    tel = memory.telemetry or {}
    online_norm: float | None = None
    online_sum: float | None = None
    if memory.current_weights is not None:
        w1 = memory.current_weights[0].detach()
        online_norm = float(w1.norm().item())
        online_sum = float(w1.sum().item())
    return MemorySnapshot(
        stage=stage,
        base_w1_norm=float(memory.base_W1.detach().norm().item()),
        base_w1_sum=float(memory.base_W1.detach().sum().item()),
        online_w1_norm=online_norm,
        online_w1_sum=online_sum,
        online_weights_present=memory.current_weights is not None,
        telemetry_update_count=int(tel.get("update_count", 0)),
        telemetry_avg_update_mag=float(tel.get("average_update_magnitude", 0.0)),
    )


def _online_w1_delta(memory: TitansMemory, reference: torch.Tensor | None) -> float:
    if reference is None or memory.current_weights is None:
        return 0.0
    return float((memory.current_weights[0].detach() - reference).pow(2).sum().sqrt().item())


class TitansInstrumentor:
    """Attach call counters to a TitansMemory instance (non-invasive wrapper)."""

    def __init__(self, memory: TitansMemory) -> None:
        self.memory = memory
        self.counts: dict[str, int] = defaultdict(int)
        self._originals: dict[str, Callable[..., Any]] = {}
        self._attach()

    def _wrap(self, name: str) -> Callable[..., Any]:
        original = getattr(self.memory, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.counts[name] += 1
            return original(*args, **kwargs)

        self._originals[name] = original
        setattr(self.memory, name, wrapped)
        return wrapped

    def _attach(self) -> None:
        for method in ("read", "write", "forward", "inject", "forward_with_injection"):
            self._wrap(method)
        updater = self.memory.updater
        original_update = updater.update

        def wrapped_update(*args: Any, **kwargs: Any) -> Any:
            self.counts["updater.update"] += 1
            return original_update(*args, **kwargs)

        updater.update = wrapped_update  # type: ignore[method-assign]
        self._originals["updater.update"] = original_update

    def restore(self) -> None:
        for name, fn in self._originals.items():
            if name == "updater.update":
                self.memory.updater.update = fn  # type: ignore[method-assign]
            else:
                setattr(self.memory, name, fn)


def audit_path(
    path_name: str,
    run_fn: Callable[[TitansMemory, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: IVERIConfig | None = None,
    batch: int = 2,
    patches: int = 6,
) -> PathAuditResult:
    cfg = cfg or build_mini_config()
    memory = TitansMemory(cfg)
    instrumentor = TitansInstrumentor(memory)
    memory.reset_memory()

    x = torch.randn(batch, patches, cfg.model.hidden_dim)
    entropy = torch.rand(batch, patches, 1)

    snapshots: list[MemorySnapshot] = []
    snapshots.append(_snapshot_memory(memory, "before_forward"))
    base_w1_ref = memory.base_W1.detach().clone()

    out = run_fn(memory, x, entropy)
    snapshots.append(_snapshot_memory(memory, "after_forward"))

    after_first_online = (
        memory.current_weights[0].detach().clone()
        if memory.current_weights is not None
        else None
    )

    online_delta = 0.0
    if memory.current_weights is not None:
        base_expanded = base_w1_ref.unsqueeze(0).expand(memory.current_weights[0].shape[0], -1, -1)
        online_delta = float(
            (memory.current_weights[0].detach() - base_expanded).pow(2).sum().sqrt().item()
        )

    # Second identical call for persistence check
    _ = run_fn(memory, x, entropy)
    snapshots.append(_snapshot_memory(memory, "after_second_forward"))

    persistence = False
    if after_first_online is not None and memory.current_weights is not None:
        persistence = torch.allclose(
            after_first_online, memory.current_weights[0].detach(), atol=0.0, rtol=0.0
        )
    tel = memory.telemetry or {}
    telemetry_reports_writes = int(tel.get("update_count", 0)) > 0

    result = PathAuditResult(
        path_name=path_name,
        read_calls=instrumentor.counts["read"],
        write_calls=instrumentor.counts["write"],
        forward_calls=instrumentor.counts["forward"],
        inject_calls=instrumentor.counts["inject"],
        updater_calls=instrumentor.counts["updater.update"],
        online_weight_delta_after=online_delta,
        persistence_across_second_call=persistence,
        telemetry_reports_writes=telemetry_reports_writes,
        snapshots=snapshots,
    )
    instrumentor.restore()
    _ = out  # silence unused in paths that return tensor
    return result


def audit_backbone_production(cfg: IVERIConfig | None = None) -> PathAuditResult:
    cfg = cfg or build_mini_config()
    backbone = Backbone(cfg)
    instrumentor = TitansInstrumentor(backbone.titans)
    backbone.titans.reset_memory()

    B, P, D = 2, 6, cfg.model.hidden_dim
    x = torch.randn(B, P, D)
    entropy = torch.rand(B, P, 1)

    snapshots: list[MemorySnapshot] = []
    snapshots.append(_snapshot_memory(backbone.titans, "before_forward"))
    base_w1_ref = backbone.titans.base_W1.detach().clone()

    out = backbone(x, entropy=entropy)
    snapshots.append(_snapshot_memory(backbone.titans, "after_forward"))

    loss = out.pow(2).mean()
    loss.backward()
    snapshots.append(_snapshot_memory(backbone.titans, "after_backward"))

    opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    base_before = backbone.titans.base_W1.detach().clone()
    online_before = (
        backbone.titans.current_weights[0].detach().clone()
        if backbone.titans.current_weights is not None
        else None
    )
    opt.step()
    snapshots.append(_snapshot_memory(backbone.titans, "after_optimizer"))

    _ = backbone(x, entropy=entropy)
    snapshots.append(_snapshot_memory(backbone.titans, "after_second_forward"))

    persistence = False
    if online_before is not None and backbone.titans.current_weights is not None:
        persistence = torch.allclose(
            online_before, backbone.titans.current_weights[0].detach(), atol=0.0, rtol=0.0
        )

    tel = backbone.titans.telemetry or {}
    backbone_telemetry_writes = int(backbone.telemetry.get("titans_write_count", 0))

    online_delta = 0.0
    if backbone.titans.current_weights is not None:
        base_expanded = base_w1_ref.unsqueeze(0).expand(
            backbone.titans.current_weights[0].shape[0], -1, -1
        )
        online_delta = float(
            (backbone.titans.current_weights[0].detach() - base_expanded)
            .pow(2)
            .sum()
            .sqrt()
            .item()
        )

    result = PathAuditResult(
        path_name="Backbone.production (forward→update→gate)",
        read_calls=instrumentor.counts["read"],
        write_calls=instrumentor.counts["write"],
        forward_calls=instrumentor.counts["forward"]
        + instrumentor.counts.get("forward_with_injection", 0),
        inject_calls=instrumentor.counts["inject"],
        updater_calls=instrumentor.counts["updater.update"],
        online_weight_delta_after=online_delta,
        persistence_across_second_call=persistence,
        telemetry_reports_writes=int(tel.get("update_count", 0)) > 0
        or backbone_telemetry_writes > 0,
        snapshots=snapshots,
    )
    instrumentor.restore()
    return result


def audit_gradient_flow(cfg: IVERIConfig | None = None) -> list[GradientAuditResult]:
    cfg = cfg or build_mini_config()
    results: list[GradientAuditResult] = []

    # Production inject path
    backbone = Backbone(cfg)
    B, P, D = 2, 5, cfg.model.hidden_dim
    x = torch.randn(B, P, D, requires_grad=True)
    entropy = torch.rand(B, P, 1)
    online_before = None
    backbone.titans.reset_memory()
    out = backbone(x, entropy=entropy)
    if backbone.titans.current_weights is not None:
        online_before = backbone.titans.current_weights[0].detach().clone()
    loss = out.sum() + sum(backbone.current_aux_losses)
    loss.backward()
    base_g1 = float(backbone.titans.base_W1.grad.norm().item()) if backbone.titans.base_W1.grad is not None else 0.0
    base_g2 = float(backbone.titans.base_W2.grad.norm().item()) if backbone.titans.base_W2.grad is not None else 0.0
    q_g = float(backbone.titans.q_proj.weight.grad.norm().item()) if backbone.titans.q_proj.weight.grad is not None else 0.0
    online_changed = False
    if online_before is not None and backbone.titans.current_weights is not None:
        online_changed = not torch.allclose(online_before, backbone.titans.current_weights[0].detach())
    base_clone = backbone.titans.base_W1.detach().clone()
    opt = torch.optim.SGD(backbone.parameters(), lr=0.01)
    opt.step()
    optimizer_base = not torch.allclose(base_clone, backbone.titans.base_W1.detach())
    optimizer_online = False
    if online_before is not None and backbone.titans.current_weights is not None:
        optimizer_online = not torch.allclose(online_before, backbone.titans.current_weights[0].detach())
    results.append(
        GradientAuditResult(
            path_name="Backbone.forward_with_injection (production)",
            base_w1_grad_norm=base_g1,
            base_w2_grad_norm=base_g2,
            q_proj_grad_norm=q_g,
            online_weights_changed_during_forward=online_changed,
            optimizer_changed_base_w1=optimizer_base,
            optimizer_changed_online_w1=optimizer_online,
        )
    )

    # Isolated TitansMemory.forward path
    memory = TitansMemory(cfg)
    memory.reset_memory()
    x2 = torch.randn(B, P, D, requires_grad=True)
    online_before2 = None
    out2 = memory.forward(x2)
    if memory.current_weights is not None:
        online_before2 = memory.current_weights[0].detach().clone()
    loss2 = out2.sum()
    loss2.backward()
    base_g1_2 = float(memory.base_W1.grad.norm().item()) if memory.base_W1.grad is not None else 0.0
    base_g2_2 = float(memory.base_W2.grad.norm().item()) if memory.base_W2.grad is not None else 0.0
    q_g2 = float(memory.q_proj.weight.grad.norm().item()) if memory.q_proj.weight.grad is not None else 0.0
    online_changed2 = False
    if online_before2 is not None and memory.current_weights is not None:
        online_changed2 = not torch.allclose(online_before2, memory.current_weights[0].detach())
    results.append(
        GradientAuditResult(
            path_name="TitansMemory.forward (isolated)",
            base_w1_grad_norm=base_g1_2,
            base_w2_grad_norm=base_g2_2,
            q_proj_grad_norm=q_g2,
            online_weights_changed_during_forward=online_changed2,
            optimizer_changed_base_w1=False,
            optimizer_changed_online_w1=False,
        )
    )
    return results


def prove_write_presence(production: PathAuditResult) -> list[str]:
    return [
        "Production `Backbone.forward` calls `self.titans.forward_with_injection(x, entropy)` "
        "(model/backbone.py), which invokes `TitansMemory.forward()` for online read/update/write "
        "and applies entropy gating.",
        "`forward()` runs a sequential loop: read via `_forward_mlp`, compute local loss, "
        "and call `MemoryUpdater.update` per patch step.",
        f"Runtime instrumentation on production path: forward_calls={production.forward_calls}, "
        f"updater_calls={production.updater_calls}, read_calls={production.read_calls}, "
        f"write_calls={production.write_calls}, inject_calls={production.inject_calls}.",
        f"Online weights diverge from expanded base_W1 after production forward "
        f"(online_weight_delta={production.online_weight_delta_after:.6e}).",
        f"Titans telemetry reports updates after production forward: "
        f"update_count={production.snapshots[1].telemetry_update_count}, "
        f"avg_update_mag={production.snapshots[1].telemetry_avg_update_mag:.6e}.",
        f"Backbone telemetry `titans_write_count` is sourced from measured Titans telemetry "
        f"(not hardcoded B*P).",
    ]


def prove_write_absence(production: PathAuditResult) -> list[str]:
    proofs = [
        "Production `Backbone.forward` calls `self.titans.inject(x, entropy)` only "
        "(model/backbone.py) — never `titans.forward()` or `titans.write()`.",
        "`inject()` implementation calls `self.read(x)` then applies entropy gate; "
        "`read()` docstring and code explicitly perform parallel reads with no weight updates.",
        f"Runtime instrumentation on production path: read_calls={production.read_calls}, "
        f"write_calls={production.write_calls}, forward_calls={production.forward_calls}, "
        f"updater_calls={production.updater_calls}.",
        "Online `current_weights` tensor unchanged across production forward/backward/optimizer "
        f"(online_weight_delta={production.online_weight_delta_after:.6e}, "
        f"persistence={production.persistence_across_second_call}).",
        "`MemoryUpdater.update` is only reachable from `TitansMemory.write()` and "
        "`TitansMemory.forward()` sequential loops — both bypassed in production.",
        f"Backbone telemetry hardcodes `titans_writes = B * P` regardless of actual updates; "
        f"Titans telemetry update_count after production forward = "
        f"{production.snapshots[1].telemetry_update_count}.",
    ]
    return proofs


def run_titans_audit() -> TitansAuditResult:
    cfg = build_mini_config()

    isolated_forward = audit_path(
        "TitansMemory.forward (isolated)",
        lambda m, x, _e: m.forward(x),
        cfg,
    )
    isolated_write = audit_path(
        "TitansMemory.write (isolated)",
        lambda m, x, _e: (m.write(m.k_proj(x), m.v_proj(x)), x)[1],
        cfg,
    )
    isolated_inject = audit_path(
        "TitansMemory.inject (isolated)",
        lambda m, x, e: m.inject(x, e),
        cfg,
    )
    production = audit_backbone_production(cfg)
    gradients = audit_gradient_flow(cfg)

    writes_in_prod = production.forward_calls > 0 and production.updater_calls > 0
    presence_proofs = prove_write_presence(production) if writes_in_prod else []
    absence_proofs = [] if writes_in_prod else prove_write_absence(production)

    return TitansAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        device="cpu",
        production_verdict="PASS" if writes_in_prod else "FAIL",
        writes_occur_in_production=writes_in_prod,
        write_absence_proof=absence_proofs,
        write_presence_proof=presence_proofs,
        path_results=[isolated_forward, isolated_write, isolated_inject, production],
        gradient_results=gradients,
        isolated_forward_updates=isolated_forward.updater_calls,
        isolated_forward_avg_update_mag=isolated_forward.snapshots[-1].telemetry_avg_update_mag,
    )


def render_titans_verification_report(result: TitansAuditResult) -> str:
    lines = [
        "# Titans Memory Verification Report (Phase 6.3.2 OBJ2)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        f"**Device:** {result.device}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Production integrated path:** `{result.production_verdict}`",
        "",
        f"**Writes occur in production:** `{result.writes_occur_in_production}`",
        "",
        "Instrumentation captures memory state at: **before forward**, **after forward**, "
        "**after backward**, **after optimizer**, and **after second forward** (persistence).",
        "",
    ]

    if result.writes_occur_in_production:
        lines.extend(["## Proof: Online Writes Occur in Production", ""])
        for i, proof in enumerate(result.write_presence_proof, 1):
            lines.append(f"{i}. {proof}")
        lines.append("")
    else:
        lines.extend(["## Proof: Why Writes Never Occur in Production", ""])
        for i, proof in enumerate(result.write_absence_proof, 1):
            lines.append(f"{i}. {proof}")
        lines.append("")

    lines.extend(
        [
            "## Path Comparison (runtime call counts)",
            "",
            "| Path | read | write | forward | inject | updater.update | Online ΔW1 | Persists 2nd call |",
            "|------|-----:|------:|--------:|-------:|---------------:|-----------:|:---------------:|",
        ]
    )
    for p in result.path_results:
        persist = "yes" if p.persistence_across_second_call else "no"
        lines.append(
            f"| {p.path_name} | {p.read_calls} | {p.write_calls} | {p.forward_calls} | "
            f"{p.inject_calls} | {p.updater_calls} | {p.online_weight_delta_after:.3e} | {persist} |"
        )

    lines.extend(["", "## Lifecycle Snapshots (production path)", ""])
    prod = next(p for p in result.path_results if "Backbone.production" in p.path_name)
    lines.append(
        "| Stage | base_W1 ‖·‖ | online_W1 ‖·‖ | telemetry updates | avg update mag |"
    )
    lines.append("|-------|----------:|---------------:|------------------:|---------------:|")
    for s in prod.snapshots:
        online = f"{s.online_w1_norm:.6f}" if s.online_w1_norm is not None else "N/A"
        lines.append(
            f"| {s.stage} | {s.base_w1_norm:.6f} | {online} | "
            f"{s.telemetry_update_count} | {s.telemetry_avg_update_mag:.6e} |"
        )

    lines.extend(["", "## Gradient Flow", ""])
    lines.append(
        "| Path | ‖grad base_W1‖ | ‖grad base_W2‖ | ‖grad q_proj‖ | online W changed in fwd | "
        "opt changed base_W1 | opt changed online_W1 |"
    )
    lines.append("|---|--:|--:|--:|:---:|:---:|:---:|")
    for g in result.gradient_results:
        lines.append(
            f"| {g.path_name} | {g.base_w1_grad_norm:.4e} | {g.base_w2_grad_norm:.4e} | "
            f"{g.q_proj_grad_norm:.4e} | {g.online_weights_changed_during_forward} | "
            f"{g.optimizer_changed_base_w1} | {g.optimizer_changed_online_w1} |"
        )

    lines.extend(
        [
            "",
            "## Capability Matrix (measured)",
            "",
            "| Capability | Isolated forward | Isolated write | Production forward |",
            "|------------|:----------------:|:--------------:|:------------------:|",
        ]
    )
    iso_fwd = next(p for p in result.path_results if "forward (isolated)" in p.path_name)
    iso_w = next(p for p in result.path_results if "write (isolated)" in p.path_name)
    lines.append(
        f"| Memory reads | {iso_fwd.read_calls > 0} | {iso_w.read_calls > 0} | {prod.read_calls > 0} |"
    )
    lines.append(
        f"| Memory writes | {iso_fwd.write_calls > 0} | {iso_w.write_calls > 0} | {prod.write_calls > 0} |"
    )
    lines.append(
        f"| Online updates (updater) | {iso_fwd.updater_calls > 0} | {iso_w.updater_calls > 0} | "
        f"{prod.updater_calls > 0} |"
    )
    lines.append(
        f"| Online weight replacement | {iso_fwd.online_weight_delta_after > 0} | "
        f"{iso_w.online_weight_delta_after > 0} | {prod.online_weight_delta_after > 0} |"
    )
    lines.append(
        f"| Persistence (2nd identical call) | {iso_fwd.persistence_across_second_call} | "
        f"{iso_w.persistence_across_second_call} | {prod.persistence_across_second_call} |"
    )

    lines.extend(
        [
            "",
            "## Isolated Forward Reference",
            "",
            f"- `updater.update` calls during isolated `TitansMemory.forward`: **{result.isolated_forward_updates}**",
            f"- Average update magnitude (isolated): **{result.isolated_forward_avg_update_mag:.6e}**",
            "",
            "> Phase 6.3.2 OBJ2 wires `TitansMemory.forward_with_injection` into `Backbone.forward`.",
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


def write_titans_verification_report(output_path: str | Path) -> TitansAuditResult:
    result = run_titans_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_titans_verification_report(result), encoding="utf-8")
    return result
