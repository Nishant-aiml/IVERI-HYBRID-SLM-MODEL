# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Documentation discrepancies audit (Phase 6.3.1H).

Compares README, implementation plans, walkthroughs, task files, architecture
docs, publication artifacts, and replay docs against verified implementation
evidence. Does not modify scientific claims in publication artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "reports" / "scientific_integrity_audit"

DiscrepancyStatus = Literal["VERIFIED", "TODO", "PENDING", "CONTRADICTION"]


@dataclass
class DiscrepancyItem:
    source: str
    claim: str
    status: DiscrepancyStatus
    evidence: str
    severity: str = "MEDIUM"


@dataclass
class DocumentationDiscrepanciesResult:
    protocol_version: str = "Phase-6.3.1H"
    timestamp_utc: str = ""
    production_verdict: str = "UNKNOWN"
    items: list[DiscrepancyItem] = field(default_factory=list)
    documents_scanned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(rel: str) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _audit_report_pass(name: str) -> bool | None:
    path = AUDIT_DIR / name
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if "**PASS**" in text or "`PASS`" in text:
        return True
    if "**FAIL**" in text or "`FAIL`" in text:
        return False
    return None


def _glob_exists(pattern: str) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in REPO_ROOT.glob(pattern)]


def run_documentation_discrepancies_audit() -> DocumentationDiscrepanciesResult:
    items: list[DiscrepancyItem] = []
    scanned: list[str] = []

    doc_targets = [
        "README.md",
        "CHANGELOG.md",
        "IVERI_PROJECT_MASTER.md",
        "docs/phases/INDEX.md",
        "docs/phases/phase_0_plan.md",
        "docs/architecture/overview.md",
        "docs/architecture/README.md",
        "docs/research/Research_Methodology.md",
        "docs/research/Reproducibility_Guide.md",
        "docs/migrations/PHASE_6_3_2_OBJ1_CAUSALITY.md",
        "docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md",
    ]
    for rel in doc_targets:
        if (REPO_ROOT / rel).exists():
            scanned.append(rel)

    readme = _read("README.md")
    changelog = _read("CHANGELOG.md")
    master = _read("IVERI_PROJECT_MASTER.md")
    methodology = _read("docs/research/Research_Methodology.md")

    # README phase status (post OBJ8 sync)
    if "6.3.2" in readme and "🔲 Not Started" not in readme:
        items.append(
            DiscrepancyItem(
                "README.md",
                "Phase roadmap reflects completed work through 6.3.2",
                "VERIFIED",
                "README lists 6.3.2; no stale 'Not Started' markers for Phases 2–6",
                "LOW",
            )
        )
    elif "🔲 Not Started" in readme:
        items.append(
            DiscrepancyItem(
                "README.md",
                "Phases 2–6 marked Not Started",
                "CONTRADICTION",
                "CHANGELOG and campaign artifacts show Phases 2–6.3 implemented",
                "CRITICAL",
            )
        )

    # CHANGELOG vs README
    if "[1.5.0]" in changelog and "Phase 6.3.2" in changelog:
        items.append(
            DiscrepancyItem(
                "CHANGELOG.md",
                "Phase 6.3.2 scientific integrity restoration documented",
                "VERIFIED",
                "CHANGELOG [1.5.0] entry present",
                "LOW",
            )
        )

    # Master doc status
    if "Build starting Phase 0" in master:
        items.append(
            DiscrepancyItem(
                "IVERI_PROJECT_MASTER.md",
                'Status still "Build starting Phase 0"',
                "CONTRADICTION",
                "reports/phase_6_3/ and Phase 6.3.2 audits exist",
                "CRITICAL",
            )
        )
    elif "6.3.2" in master:
        items.append(
            DiscrepancyItem(
                "IVERI_PROJECT_MASTER.md",
                "Master document references Phase 6.3.2",
                "VERIFIED",
                "Status section updated in OBJ8 sync",
                "LOW",
            )
        )

    # Titans — verify against runtime audit
    titans_pass = _audit_report_pass("Titans_Verification.md")
    if titans_pass is True:
        items.append(
            DiscrepancyItem(
                "IVERI_PROJECT_MASTER.md / architecture docs",
                "Titans online memory writes on production forward path",
                "VERIFIED",
                "Titans_Verification.md PASS (Phase 6.3.2 OBJ2)",
                "LOW",
            )
        )
    elif titans_pass is False:
        items.append(
            DiscrepancyItem(
                "IVERI_PROJECT_MASTER.md",
                "Titans online memory active",
                "CONTRADICTION",
                "Titans_Verification.md FAIL",
                "CRITICAL",
            )
        )
    else:
        items.append(
            DiscrepancyItem(
                "IVERI_PROJECT_MASTER.md",
                "Titans online memory claims",
                "PENDING",
                "Titans_Verification.md not found or inconclusive",
                "HIGH",
            )
        )

    # Entropy MoE
    entropy_pass = _audit_report_pass("Entropy_Routing_Report.md")
    if entropy_pass is True:
        items.append(
            DiscrepancyItem(
                "docs/architecture/moe_routing.md",
                "Entropy-conditioned MoE routing implemented",
                "VERIFIED",
                "Entropy_Routing_Report.md PASS (Phase 6.3.2 OBJ3)",
                "LOW",
            )
        )
    else:
        items.append(
            DiscrepancyItem(
                "docs/architecture/moe_routing.md",
                "Entropy routing behavior",
                "TODO",
                "Re-run Entropy_MoE_Audit.md after campaign evidence",
                "MEDIUM",
            )
        )

    # Ablation verification
    ablation_pass = _audit_report_pass("Ablation_Verification.md")
    if ablation_pass is True:
        items.append(
            DiscrepancyItem(
                "docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md",
                "Physical ablations (no Titans/BLT/MoR/MoE/entropy routing) distinct",
                "VERIFIED",
                "Ablation_Verification.md PASS (Phase 6.3.1F)",
                "LOW",
            )
        )

    # Statistics methodology count
    method_count_readme = len(re.findall(r"Shapiro|Wilcoxon|Cohen|Cliff|Holm|bootstrap", methodology, re.I))
    stats_pass = _audit_report_pass("Statistics_Consistency_Report.md")
    if stats_pass is True:
        items.append(
            DiscrepancyItem(
                "docs/research/Research_Methodology.md",
                "Seven statistical methods via single pipeline",
                "VERIFIED",
                "Statistics_Consistency_Report.md PASS",
                "LOW",
            )
        )
    elif method_count_readme < 5:
        items.append(
            DiscrepancyItem(
                "docs/research/Research_Methodology.md",
                "Lists 4 methods; certificates claim 7",
                "TODO",
                "Update methodology doc after Statistics_Consistency_Report PASS",
                "MEDIUM",
            )
        )

    # Missing phase plans
    phase_plans = _glob_exists("docs/phases/phase_*_plan.md")
    if len(phase_plans) <= 1:
        items.append(
            DiscrepancyItem(
                "docs/phases/",
                "Phase 1–6.3 implementation plans",
                "PENDING",
                f"Only {len(phase_plans)} phase plan file(s) on disk",
                "MEDIUM",
            )
        )

    # Walkthroughs / task files
    walkthroughs = _glob_exists("**/*walkthrough*")
    task_files = _glob_exists("**/task.md")
    if not walkthroughs:
        items.append(
            DiscrepancyItem(
                "repository root",
                "Project walkthrough documents",
                "PENDING",
                "No walkthrough files found",
                "LOW",
            )
        )
    if not task_files:
        items.append(
            DiscrepancyItem(
                "repository root",
                "task.md task tracking files",
                "PENDING",
                "No task.md files found",
                "LOW",
            )
        )

    # Publication vs evidence index
    hyp_report = _read("reports/phase_6_3/Hypothesis_Report.md")
    cert = _read("reports/phase_6_3/Phase_6_3_Certificate.md")
    if hyp_report and "PENDING" in hyp_report and cert and "SUPPORTED" in cert:
        items.append(
            DiscrepancyItem(
                "reports/phase_6_3/",
                "Hypothesis_Report PENDING vs Certificate SUPPORTED",
                "CONTRADICTION",
                "Publication artifacts disagree; do not alter scientific claims without campaign",
                "HIGH",
            )
        )

    # Replay integrity
    replay_pass = _audit_report_pass("Replay_Integrity_Report.md")
    if replay_pass is True:
        items.append(
            DiscrepancyItem(
                "docs/research/Reproducibility_Guide.md",
                "Replay lineage and artifact checksums",
                "VERIFIED",
                "Replay_Integrity_Report.md PASS (Phase 6.3.2 OBJ6)",
                "LOW",
            )
        )

    # Phase numbering in campaign
    campaign = _read("research/campaign_runner.py")
    if "Phase 5.0" in campaign or "Phase5" in campaign:
        items.append(
            DiscrepancyItem(
                "research/campaign_runner.py",
                "Experiment IDs label Phase 5.0 while output dir is phase_6_3",
                "TODO",
                "Harmonize phase labels in metadata without changing metrics",
                "MEDIUM",
            )
        )

    contradictions = sum(1 for i in items if i.status == "CONTRADICTION")
    verdict = "PASS" if contradictions == 0 else "FAIL"
    return DocumentationDiscrepanciesResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        production_verdict=verdict,
        items=items,
        documents_scanned=scanned,
    )


def render_documentation_discrepancies_report(result: DocumentationDiscrepanciesResult) -> str:
    lines = [
        "# Documentation Discrepancies Report (Phase 6.3.1H)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        f"**Mode:** Audit-only — scientific claims in publication artifacts are not modified",
        "",
        "## Executive Verdict",
        "",
        f"**Documentation consistency:** `{result.production_verdict}`",
        "",
        f"**Documents scanned:** {len(result.documents_scanned)}",
        "",
        "## Discrepancy Register",
        "",
        "| Source | Claim | Status | Severity | Evidence |",
        "|--------|-------|--------|----------|----------|",
    ]
    for item in result.items:
        claim = item.claim.replace("|", "\\|")
        evidence = item.evidence.replace("|", "\\|")
        lines.append(
            f"| `{item.source}` | {claim} | **{item.status}** | {item.severity} | {evidence} |"
        )

    status_counts: dict[str, int] = {}
    for item in result.items:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1

    lines.extend(["", "## Status Summary", ""])
    for status in ("VERIFIED", "TODO", "PENDING", "CONTRADICTION"):
        if status in status_counts:
            lines.append(f"- **{status}:** {status_counts[status]}")

    lines.extend(
        [
            "",
            "## Documents Scanned",
            "",
        ]
    )
    for doc in result.documents_scanned:
        lines.append(f"- `{doc}`")

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


def write_documentation_discrepancies_report(output_path: str | Path) -> DocumentationDiscrepanciesResult:
    result = run_documentation_discrepancies_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_documentation_discrepancies_report(result), encoding="utf-8")
    return result
