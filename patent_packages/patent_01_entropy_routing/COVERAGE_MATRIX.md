# Coverage Matrix: patent_01_entropy_routing
**Patent Title:** *System and Method for Entropy-Guided Dynamic Expert Allocation in Neural Networks*

| Requirement / Claim | Technical Effect | Verification Artifact | Status |
|:---|:---|:---|:---:|
| **Claim 1 (Independent)** | Dynamic entropy threshold expert routing | `proof/VERIFY_PATENT.py` | ✅ Enablement Validated |
| **Claim 2 (Dependent)** | Adaptive token-level capacity scaling | `proof/VERIFY_PATENT.py` | ✅ Enablement Validated |
| **Technical Effect Proof** | 22-35% FLOP reduction vs static MoE | `technical_evidence/routing_stats.csv` | 🟡 In Progress (Stage B) |
| **Prior Art Comparison** | Distinct vs Switch Transformer | `NOVELTY_MATRIX.md` | ✅ Validated |
| **Industrial Applicability** | Runs on 4GB VRAM RTX 3050 | `EVIDENCE_LEDGER.json` | ✅ Validated |
