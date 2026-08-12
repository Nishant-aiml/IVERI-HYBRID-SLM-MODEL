# IVERI CORE — Architecture Decision Log (06_DECISION_LOG.md)

## Architecture Decision Board Scoring Matrix

Each component is quantitatively evaluated across 6 primary axes (0–10 scale):
1. **Scientific Value (25%)**
2. **Engineering Quality (20%)**
3. **Patent Value (20%)**
4. **Production Readiness (15%)**
5. **Maintenance Overhead (-10%)**
6. **Training Cost (-10%)**

---

## Logged Decisions

### DECISION-001: Gradient Checkpointing for Short Sequences
- **Date:** 2026-07-26
- **Evidence Source:** `latency_breakdown.json` (Phase 3 profiling)
- **Finding:** Grad checkpointing added 32% throughput overhead (154 tok/s vs 203 tok/s) at seq_len=64 while VRAM was only 1.7GB / 4.0GB.
- **Decision:** **MODIFY** → Disable gradient checkpointing for seq_len < 128 (creates IVERI v0.91).
- **Status:** **IMPLEMENTED & VERIFIED** (v0.91 achieved 172.5 tok/s vs 137.0 tok/s in v0.90, +25.9% speedup).

### DECISION-002: Failure Analysis & Initial Gradient Explosions (Phase 5)
- **Date:** 2026-07-26
- **Evidence Source:** `tier1_ablations_and_failures.json` (Phase 5 failure detection)
- **Finding:** Early steps (steps 1–3) suffered `GRADIENT_EXPLOSION` (`val: Infinity`) across IVERI Full (5 events) and IVERI No Titans (7 events). Dense Baseline suffered 0 events (`mean_grad_norm` = 1.52). IVERI No MoR suffered only 1 event.
- **Root Cause:** Complex routing & memory parameter updates without initial learning rate warmup cause large early loss gradient steps before AdamW moment estimates stabilize.
- **Decision:** **MODIFY** → Introduce 100-step linear LR warmup in trainer configuration (**IVERI v0.91A**).
- **Status:** **APPROVED FOR IMPLEMENTATION**

### DECISION-003: Mixture of Recursions (MoR) Latency vs Loss Benefit (Phase 6)
- **Date:** 2026-07-26
- **Evidence Source:** `phase_5_6_analysis_report.json` (Phase 6 Tier 1 ablations)
- **Finding:** Disabling MoR doubles throughput (172.5 tok/s → 359.2 tok/s, **2.08× speedup**) and reduces peak VRAM by 49% (828.4 MB → 422.7 MB), while final loss difference is statistically inconclusive (3.1425 vs 3.1365, p = 0.4611).
- **Decision:** **REDESIGN** → Implement conditional depth recursion: disable MoR at short sequence lengths (seq_len < 128) where sequence complexity does not justify deep recursive passes (**IVERI v0.92**).
- **Status:** **APPROVED FOR IMPLEMENTATION**

---

## Quantitative Component Maturity Table (Updated July 26, 2026)

| Component | Scientific | Engineering | Patent | Production | Keep? | Version |
|-----------|------------|-------------|--------|------------|-------|---------|
| Entropy Router | 8.0 | 9.0 | 9.5 | 7.0 | KEEP | v0.91 |
| Titans Memory | 7.5 | 8.0 | 9.0 | 6.0 | KEEP | v0.91 |
| MoR Recursion | 6.0 | 7.5 | 8.5 | 5.0 | REDESIGN | v0.92 (Conditional) |
| BLT Patcher | 8.5 | 8.5 | 8.0 | 6.5 | KEEP | v0.91 |
