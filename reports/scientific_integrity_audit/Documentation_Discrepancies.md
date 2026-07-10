# Documentation Discrepancies Report (Phase 6.3.1H)

**Generated:** 2026-07-09T03:01:31Z  
**Protocol:** Phase-6.3.1H  
**Mode:** Audit-only — scientific claims in publication artifacts are not modified

## Executive Verdict

**Documentation consistency:** `PASS`

**Documents scanned:** 11

## Discrepancy Register

| Source | Claim | Status | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `README.md` | Phase roadmap reflects completed work through 6.3.2 | **VERIFIED** | LOW | README lists 6.3.2; no stale 'Not Started' markers for Phases 2–6 |
| `CHANGELOG.md` | Phase 6.3.2 scientific integrity restoration documented | **VERIFIED** | LOW | CHANGELOG [1.5.0] entry present |
| `IVERI_PROJECT_MASTER.md` | Master document references Phase 6.3.2 | **VERIFIED** | LOW | Status section updated in OBJ8 sync |
| `IVERI_PROJECT_MASTER.md / architecture docs` | Titans online memory writes on production forward path | **VERIFIED** | LOW | Titans_Verification.md PASS (Phase 6.3.2 OBJ2) |
| `docs/architecture/moe_routing.md` | Entropy-conditioned MoE routing implemented | **VERIFIED** | LOW | Entropy_Routing_Report.md PASS (Phase 6.3.2 OBJ3) |
| `docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md` | Physical ablations (no Titans/BLT/MoR/MoE/entropy routing) distinct | **VERIFIED** | LOW | Ablation_Verification.md PASS (Phase 6.3.1F) |
| `docs/research/Research_Methodology.md` | Seven statistical methods via single pipeline | **VERIFIED** | LOW | Statistics_Consistency_Report.md PASS |
| `docs/phases/` | Phase 1–6.3 implementation plans | **PENDING** | MEDIUM | Only 1 phase plan file(s) on disk |
| `repository root` | Project walkthrough documents | **PENDING** | LOW | No walkthrough files found |
| `docs/research/Reproducibility_Guide.md` | Replay lineage and artifact checksums | **VERIFIED** | LOW | Replay_Integrity_Report.md PASS (Phase 6.3.2 OBJ6) |
| `research/campaign_runner.py` | Experiment IDs label Phase 5.0 while output dir is phase_6_3 | **TODO** | MEDIUM | Harmonize phase labels in metadata without changing metrics |

## Status Summary

- **VERIFIED:** 8
- **TODO:** 1
- **PENDING:** 2

## Documents Scanned

- `README.md`
- `CHANGELOG.md`
- `IVERI_PROJECT_MASTER.md`
- `docs/phases/INDEX.md`
- `docs/phases/phase_0_plan.md`
- `docs/architecture/overview.md`
- `docs/architecture/README.md`
- `docs/research/Research_Methodology.md`
- `docs/research/Reproducibility_Guide.md`
- `docs/migrations/PHASE_6_3_2_OBJ1_CAUSALITY.md`
- `docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md`

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.1H",
  "timestamp_utc": "2026-07-09T03:01:31Z",
  "production_verdict": "PASS",
  "items": [
    {
      "source": "README.md",
      "claim": "Phase roadmap reflects completed work through 6.3.2",
      "status": "VERIFIED",
      "evidence": "README lists 6.3.2; no stale 'Not Started' markers for Phases 2\u20136",
      "severity": "LOW"
    },
    {
      "source": "CHANGELOG.md",
      "claim": "Phase 6.3.2 scientific integrity restoration documented",
      "status": "VERIFIED",
      "evidence": "CHANGELOG [1.5.0] entry present",
      "severity": "LOW"
    },
    {
      "source": "IVERI_PROJECT_MASTER.md",
      "claim": "Master document references Phase 6.3.2",
      "status": "VERIFIED",
      "evidence": "Status section updated in OBJ8 sync",
      "severity": "LOW"
    },
    {
      "source": "IVERI_PROJECT_MASTER.md / architecture docs",
      "claim": "Titans online memory writes on production forward path",
      "status": "VERIFIED",
      "evidence": "Titans_Verification.md PASS (Phase 6.3.2 OBJ2)",
      "severity": "LOW"
    },
    {
      "source": "docs/architecture/moe_routing.md",
      "claim": "Entropy-conditioned MoE routing implemented",
      "status": "VERIFIED",
      "evidence": "Entropy_Routing_Report.md PASS (Phase 6.3.2 OBJ3)",
      "severity": "LOW"
    },
    {
      "source": "docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md",
      "claim": "Physical ablations (no Titans/BLT/MoR/MoE/entropy routing) distinct",
      "status": "VERIFIED",
      "evidence": "Ablation_Verification.md PASS (Phase 6.3.1F)",
      "severity": "LOW"
    },
    {
      "source": "docs/research/Research_Methodology.md",
      "claim": "Seven statistical methods via single pipeline",
      "status": "VERIFIED",
      "evidence": "Statistics_Consistency_Report.md PASS",
      "severity": "LOW"
    },
    {
      "source": "docs/phases/",
      "claim": "Phase 1\u20136.3 implementation plans",
      "status": "PENDING",
      "evidence": "Only 1 phase plan file(s) on disk",
      "severity": "MEDIUM"
    },
    {
      "source": "repository root",
      "claim": "Project walkthrough documents",
      "status": "PENDING",
      "evidence": "No walkthrough files found",
      "severity": "LOW"
    },
    {
      "source": "docs/research/Reproducibility_Guide.md",
      "claim": "Replay lineage and artifact checksums",
      "status": "VERIFIED",
      "evidence": "Replay_Integrity_Report.md PASS (Phase 6.3.2 OBJ6)",
      "severity": "LOW"
    },
    {
      "source": "research/campaign_runner.py",
      "claim": "Experiment IDs label Phase 5.0 while output dir is phase_6_3",
      "status": "TODO",
      "evidence": "Harmonize phase labels in metadata without changing metrics",
      "severity": "MEDIUM"
    }
  ],
  "documents_scanned": [
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
    "docs/migrations/PHASE_6_3_2_OBJ4_ABLATION.md"
  ]
}
```
