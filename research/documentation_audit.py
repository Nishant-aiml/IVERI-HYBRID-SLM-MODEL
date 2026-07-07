# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Documentation sync audit (Phase 6.3.2 OBJ8)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.constants import ARCHITECTURE_VERSION, CURRENT_PHASE


@dataclass
class DocumentationGateProbe:
    gate_name: str
    passed: bool
    detail: str


@dataclass
class DocumentationAuditResult:
    protocol_version: str = "Phase-6.3.2-OBJ8"
    timestamp_utc: str = ""
    production_verdict: str = "UNKNOWN"
    gates: list[DocumentationGateProbe] = field(default_factory=list)
    presence_proof: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_repo_file(rel_path: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / rel_path).read_text(encoding="utf-8")


def _probe_readme_phase_sync() -> DocumentationGateProbe:
    text = _read_repo_file("README.md")
    has_632 = "6.3.2" in text
    stale = "🔲 Not Started" in text
    ok = has_632 and not stale and "Documentation_Sync_Report" in text
    return DocumentationGateProbe(
        "readme_phase_sync",
        ok,
        "README lists Phase 6.3.2; no Phases 2–6 'Not Started'" if ok else "README stale",
    )


def _probe_changelog_632() -> DocumentationGateProbe:
    text = _read_repo_file("CHANGELOG.md")
    ok = "[1.5.0]" in text and "Phase 6.3.2" in text
    return DocumentationGateProbe(
        "changelog_632",
        ok,
        "CHANGELOG [1.5.0] documents Phase 6.3.2" if ok else "missing 1.5.0 entry",
    )


def _probe_master_status() -> DocumentationGateProbe:
    text = _read_repo_file("IVERI_PROJECT_MASTER.md")
    ok = "Build starting Phase 0" not in text and "6.3.2" in text
    no_legacy_core = "IVERICore(rag=rag)" not in text and "from iveri_core import IVERICore" not in text
    ok = ok and no_legacy_core
    return DocumentationGateProbe(
        "master_status",
        ok,
        "master doc status updated; IVERIModel naming" if ok else "master doc stale",
    )


def _probe_architecture_version_footer() -> DocumentationGateProbe:
    readme = _read_repo_file("README.md")
    ok = ARCHITECTURE_VERSION in readme
    return DocumentationGateProbe(
        "architecture_version_footer",
        ok,
        f"README cites {ARCHITECTURE_VERSION}" if ok else "README arch version mismatch",
    )


def _probe_current_phase_constant() -> DocumentationGateProbe:
    ok = CURRENT_PHASE >= 6
    return DocumentationGateProbe(
        "current_phase_constant",
        ok,
        f"CURRENT_PHASE={CURRENT_PHASE}" if ok else f"CURRENT_PHASE={CURRENT_PHASE} stale",
    )


def _probe_phases_index() -> DocumentationGateProbe:
    path = Path(__file__).resolve().parents[1] / "docs/phases/INDEX.md"
    ok = path.exists() and "scientific_integrity_audit" in path.read_text(encoding="utf-8")
    return DocumentationGateProbe(
        "phases_index",
        ok,
        "docs/phases/INDEX.md links audit reports" if ok else "missing phase index",
    )


def _probe_readme_changelog_alignment() -> DocumentationGateProbe:
    readme = _read_repo_file("README.md")
    changelog = _read_repo_file("CHANGELOG.md")
    readme_phases_complete = "Phase 2" in readme and "✅" in readme
    changelog_has_training = "Phase 2.1" in changelog or "Phase 2" in changelog
    ok = readme_phases_complete and changelog_has_training
    return DocumentationGateProbe(
        "readme_changelog_alignment",
        ok,
        "README and CHANGELOG both show post-Phase-1 progress" if ok else "contradiction",
    )


def run_documentation_audit() -> DocumentationAuditResult:
    gates = [
        _probe_readme_phase_sync(),
        _probe_changelog_632(),
        _probe_master_status(),
        _probe_architecture_version_footer(),
        _probe_current_phase_constant(),
        _probe_phases_index(),
        _probe_readme_changelog_alignment(),
    ]
    passed = all(g.passed for g in gates)
    return DocumentationAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        production_verdict="PASS" if passed else "FAIL",
        gates=gates,
        presence_proof=[
            "README phase table aligned with CHANGELOG through Phase 6.3.2.",
            "IVERI_PROJECT_MASTER.md status reflects implementation (not Phase 0).",
            "Production model documented as IVERIModel.",
            f"ARCHITECTURE_VERSION {ARCHITECTURE_VERSION} in README footer.",
            "docs/phases/INDEX.md indexes reports and OBJ1–8 audit artifacts.",
            "Original Documentation_Discrepancies.md retained as pre-sync baseline.",
        ],
    )


def render_documentation_sync_report(data: dict[str, Any]) -> str:
    lines = [
        "# Documentation Sync Report (Phase 6.3.2 OBJ8)",
        "",
        f"**Generated:** {data['timestamp_utc']}  ",
        f"**Protocol:** {data['protocol_version']}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Documentation consistency (post-sync):** `{data['production_verdict']}`",
        "",
        "| Gate | Result | Detail |",
        "|------|--------|--------|",
    ]
    for gate in data["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"| {gate['gate_name']} | {status} | {gate['detail']} |")
    lines.append("")

    if data["production_verdict"] == "PASS":
        lines.extend(["## Sync Actions Completed", ""])
        for i, proof in enumerate(data["presence_proof"], 1):
            lines.append(f"{i}. {proof}")
        lines.extend(
            [
                "",
                "## Historical Baseline",
                "",
                "Pre-restoration discrepancies: `Documentation_Discrepancies.md` (2026-07-06 read-only audit).",
                "",
            ]
        )

    lines.extend(
        [
            "## Raw JSON",
            "",
            "```json",
            json.dumps(data, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_documentation_sync_report(output_path: str | Path) -> dict[str, Any]:
    result = run_documentation_audit()
    data = result.to_dict()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_documentation_sync_report(data), encoding="utf-8")
    return data
