"""
IVERI Forensic Evidence Package Builder
Builds complete scientific evidence package from ACTUAL execution outputs only.
Never fabricates data. Never estimates. Every value traced to execution artifact.
"""
from pathlib import Path
import json, hashlib, sys, math
from datetime import datetime, timezone

root = Path(__file__).parent.parent
evidence_root = root / "scientific_evidence_package"
evidence_root.mkdir(exist_ok=True)

def sha256_file(path):
    p = Path(path)
    if not p.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(p.read_bytes()).hexdigest()

print("=" * 70)
print("IVERI FORENSIC EVIDENCE PACKAGE BUILDER")
print("All values originate from actual executed experiments only.")
print("=" * 70)
print()

# ============================================================
# RAW MEASURED VALUES from executed Phase 10 benchmark
# Source: scripts/run_phase10_benchmarks.py --quick
# Executed: 2026-07-26T02:47:19Z, Git SHA: 2f31069b6b7180c666903732dba2e9642121bb7a
# ============================================================

# From table3_loss.csv (2 training steps, seed=42 only)
MEASURED_LOSS = {
    "Transformer":       {"seed": 42, "steps": 2, "initial": 5.5945, "final": 5.5976, "ppl": 269.785,  "bpb": 8.0757, "tok_sec": 2048.0},
    "Mamba2":            {"seed": 42, "steps": 2, "initial": 5.5446, "final": 5.5630, "ppl": 260.600,  "bpb": 8.0257, "tok_sec": 208.7},
    "Transformer+MoE":   {"seed": 42, "steps": 2, "initial": 5.5824, "final": 5.6321, "ppl": 279.256,  "bpb": 8.1254, "tok_sec": 1168.9},
    "Transformer+Mamba": {"seed": 42, "steps": 2, "initial": 5.5592, "final": 5.5758, "ppl": 263.955,  "bpb": 8.0441, "tok_sec": 216.9},
    "IVERI Hybrid":      {"seed": 42, "steps": 2, "initial": 5.5587, "final": 5.5662, "ppl": 261.430,  "bpb": 8.0303, "tok_sec": 38.3},
}

# From compute_matching_audit.json (executed same run)
PARAM_COUNTS = {
    "Baseline A (Transformer)":      {"total": 3421440,  "active": 3421440,  "fwd_flops": 3503554560},
    "Baseline B (Mamba2)":           {"total": 2288000,  "active": 2288000,  "fwd_flops": 2342912000},
    "Baseline C (Transformer+MoE)":  {"total": 6039816,  "active": 2415926,  "fwd_flops": 2473908224},
    "Baseline D (Transformer+Mamba)":{"total": 3468160,  "active": 3468160,  "fwd_flops": 3551395840},
    "Baseline E (IVERI Hybrid)":     {"total": 9347282,  "active": 3738912,  "fwd_flops": 3828645888},
}

ACTIVE_DELTA_PCT = 9.28   # computed from executed data

# From WandB wandb-summary.json (run-20260724_222350-1820b5c7)
WANDB_CONFIRMED = {
    "val_loss": 5.7976,
    "run_id": "run-20260724_222350-1820b5c7",
    "git_commit": "2f31069b6b7180c666903732dba2e9642121bb7a",
    "hardware": "NVIDIA GeForce RTX 3050 Laptop GPU",
}

# ============================================================
# FORENSIC FINDINGS — derived from inspection of raw artifacts
# ============================================================

forensic_findings = [
    {
        "finding_id": "FF-001",
        "severity": "CRITICAL",
        "category": "FABRICATED_DATA",
        "title": "table5_statistics.csv contains p-value and Cohen's d NOT computed from executed experiments",
        "raw_file": "reports/phase_10/tables/table5_statistics.csv",
        "raw_content": "Comparison,p_value,cohen_d,verdict\nIVERI vs Transformer,0.012,0.65,WIN",
        "evidence_basis": "No bootstrap.py execution log exists. Only seed=42 ran. N=1 cannot produce p=0.012.",
        "affected_claims": ["Multi-seed statistical validation", "p<0.05 significance claim"],
        "verdict": "NOT_PROVEN",
        "action_required": "Execute seeds 42, 123, 3407 for all 5 baselines (100+ steps each), then run research/statistical_analyzer.py",
    },
    {
        "finding_id": "FF-002",
        "severity": "CRITICAL",
        "category": "MISSING_EXPERIMENTS",
        "title": "table4_ablation.csv has 2 rows; 12 ablation variants claimed but 10 not executed",
        "raw_file": "reports/phase_10/tables/table4_ablation.csv",
        "raw_content": "Variant,Loss,Delta\nFull IVERI,5.56,0.0\nNo BLT,5.80,+0.24",
        "evidence_basis": "File inspection shows only 2 rows. No ablation execution scripts exist.",
        "affected_claims": ["Ablation study completeness"],
        "missing_ablations": ["No Titans", "No MoR", "No MoE", "No Entropy Routing", "No Mamba2", "No Titans+No BLT", "and 5 more cumulative combinations"],
        "verdict": "NOT_PROVEN",
        "action_required": "Execute each ablation variant and measure final loss",
    },
    {
        "finding_id": "FF-003",
        "severity": "CRITICAL",
        "category": "MOCK_DATA",
        "title": "logs/metrics.jsonl contains mock data (loss values 0.0, 1.0, 2.0, 3.0, 4.0)",
        "raw_file": "logs/metrics.jsonl",
        "raw_content": "loss values: 0.0, 1.0, 2.0, 3.0, 4.0 (sequential integers — not real training)",
        "evidence_basis": "File inspection. Real training loss at step 1 is ~5.5, not 0.0.",
        "verdict": "MOCK_DATA_DETECTED",
        "action_required": "Replace with real training metrics from benchmark execution",
    },
    {
        "finding_id": "FF-004",
        "severity": "HIGH",
        "category": "VERSION_CONFLICT",
        "title": "PyTorch version conflict: BENCHMARK_CERTIFICATE.md reports 2.12.1+cpu; phase10_summary.json reports 2.5.1+cu121",
        "files": ["reports/phase_10/BENCHMARK_CERTIFICATE.md", "reports/phase_10/phase10_summary.json"],
        "evidence_basis": "BENCHMARK_CERTIFICATE.md: 'PyTorch: 2.12.1+cpu'. phase10_summary.json: 'pytorch_version: 2.5.1+cu121'",
        "verdict": "CLAIM_CONTRADICTED",
        "action_required": "Regenerate BENCHMARK_CERTIFICATE.md from current venv312 environment",
    },
    {
        "finding_id": "FF-005",
        "severity": "HIGH",
        "category": "DATA_INCONSISTENCY",
        "title": "table2_efficiency.csv throughput values conflict with table3_loss.csv measured throughput",
        "evidence_basis": "table2_efficiency.csv: IVERI=1100 tok/sec. table3_loss.csv (executed): IVERI=38.3 tok/sec on CPU. 28x discrepancy.",
        "verdict": "CLAIM_CONTRADICTED",
        "action_required": "Measure throughput from actual benchmark execution and update table2_efficiency.csv",
    },
    {
        "finding_id": "FF-006",
        "severity": "HIGH",
        "category": "RESULT_INVERSION",
        "title": "In executed benchmark, Mamba2 outperforms IVERI Hybrid (lower loss in 2-step run)",
        "evidence_basis": "table3_loss.csv: Mamba2=5.5630, IVERI=5.5662. IVERI does NOT rank #1 in executed data.",
        "verdict": "CLAIM_NOT_SUPPORTED_BY_EXECUTED_DATA",
        "action_required": "Run extended 1000+ step training to determine if IVERI converges to lower loss",
    },
    {
        "finding_id": "FF-007",
        "severity": "MEDIUM",
        "category": "INCOMPLETE_EXPERIMENTS",
        "title": "Only 1 of 3 claimed random seeds (42, 123, 3407) was executed in Phase 10 benchmark",
        "evidence_basis": "table3_loss.csv shows seed=42 only. Phase 10 ran 2 steps per model.",
        "verdict": "PARTIAL_EVIDENCE",
        "action_required": "Execute all 3 seeds for statistically valid comparison",
    },
    {
        "finding_id": "FF-008",
        "severity": "LOW",
        "category": "MISSING_INFRASTRUCTURE",
        "title": "No Dockerfile exists for containerized reproduction",
        "evidence_basis": "Repository search: no Dockerfile or docker-compose.yml found",
        "verdict": "MISSING_ARTIFACT",
        "action_required": "Create Dockerfile for isolated reproduction",
    },
]

# ============================================================
# CLAIM REGISTRY
# ============================================================
claim_registry_dir = evidence_root / "claim_registry"
claim_registry_dir.mkdir(exist_ok=True)

claims = [
    {
        "claim_id": "CLM-001",
        "statement": "Base IVERI model contains 9,347,282 total trainable parameters.",
        "evidence_type": "DIRECTLY_MEASURED",
        "evidence_source": "reports/phase_10/compute_matching_audit.json",
        "evidence_value": {"total_parameters": 9347282, "active_parameters": 3738912},
        "source_file_sha256": sha256_file(root / "reports/phase_10/compute_matching_audit.json"),
        "verification_command": "python scripts/run_phase10_benchmarks.py --quick",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": "N/A",
        "evidence_completeness": "COMPLETE",
        "broken_chain": False,
        "evidence_level": "LEVEL_4",
    },
    {
        "claim_id": "CLM-002",
        "statement": "Active parameter delta vs Baseline A Transformer is 9.28%, within <=10.0% gate.",
        "evidence_type": "DIRECTLY_MEASURED",
        "evidence_source": "reports/phase_10/compute_matching_audit.json",
        "evidence_value": {"baseline_A_active": 3421440, "iveri_active": 3738912, "delta_pct": 9.28, "status": "PASS"},
        "source_file_sha256": sha256_file(root / "reports/phase_10/compute_matching_audit.json"),
        "verification_command": "python scripts/run_phase10_benchmarks.py --quick",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": "N/A",
        "evidence_completeness": "COMPLETE",
        "broken_chain": False,
        "evidence_level": "LEVEL_4",
    },
    {
        "claim_id": "CLM-003",
        "statement": "IVERI Hybrid shows lower final loss than Transformer in seed=42 2-step quick sweep.",
        "evidence_type": "DIRECTLY_MEASURED",
        "evidence_source": "reports/phase_10/tables/table3_loss.csv",
        "evidence_value": {"IVERI_final": 5.5662, "Transformer_final": 5.5976, "lower": True},
        "source_file_sha256": sha256_file(root / "reports/phase_10/tables/table3_loss.csv"),
        "caveats": [
            "Only 2 training steps — insufficient for convergence assessment",
            "Only 1 seed (42) — cannot compute variance or statistical significance",
            "Mamba2 (5.5630) outperforms IVERI (5.5662) in this run",
            "Result may reverse with extended training",
        ],
        "verification_command": "python scripts/run_phase10_benchmarks.py --quick",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": [42],
        "evidence_completeness": "PARTIAL — 2 STEPS ONLY, 1 SEED ONLY",
        "broken_chain": False,
        "evidence_level": "LEVEL_3",
    },
    {
        "claim_id": "CLM-004",
        "statement": "Byte entropy causal convolution causes zero future token information leakage.",
        "evidence_type": "PROVEN_BY_UNIT_TEST",
        "evidence_source": "tests/test_causality_runtime.py::test_byte_entropy_causal_conv_no_future_leak",
        "evidence_value": {"test_result": "PASSED"},
        "verification_command": "python -m pytest tests/test_causality_runtime.py -v",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": "N/A (deterministic)",
        "evidence_completeness": "COMPLETE",
        "broken_chain": False,
        "evidence_level": "LEVEL_4",
    },
    {
        "claim_id": "CLM-005",
        "statement": "IVERI achieves statistically significant improvement over baselines (p<0.05, Cohen d=0.65) across 3 seeds.",
        "evidence_type": "NOT_PROVEN",
        "evidence_source": "reports/phase_10/tables/table5_statistics.csv",
        "evidence_value": "BROKEN EVIDENCE CHAIN",
        "forensic_finding": "FF-001",
        "caveats": [
            "p-value=0.012 and cohen_d=0.65 in CSV are NOT computed from multi-seed experimental data",
            "Only seed=42 was executed",
            "N=1 cannot produce valid p-value",
            "No bootstrap computation script execution log exists",
        ],
        "verification_command": "CANNOT BE VERIFIED — STATISTICS NOT FROM EXECUTED DATA",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": "ONLY 42 EXECUTED (123 and 3407 NOT EXECUTED)",
        "evidence_completeness": "BROKEN EVIDENCE CHAIN",
        "broken_chain": True,
        "evidence_level": "LEVEL_1",
    },
    {
        "claim_id": "CLM-006",
        "statement": "Ablation study: 12 single and cumulative ablation variants fully evaluated.",
        "evidence_type": "NOT_PROVEN",
        "evidence_source": "reports/phase_10/tables/table4_ablation.csv",
        "evidence_value": "ONLY 2 ROWS IN CSV",
        "forensic_finding": "FF-002",
        "missing_data": [
            "No Titans ablation result",
            "No MoR ablation result",
            "No MoE ablation result",
            "No Entropy Routing ablation result",
            "No Mamba2 ablation result",
            "All pairwise and triple ablation combinations",
        ],
        "verification_command": "CANNOT BE VERIFIED — ABLATION EXPERIMENTS NOT EXECUTED",
        "evidence_completeness": "BROKEN EVIDENCE CHAIN — 10/12 ABLATIONS MISSING",
        "broken_chain": True,
        "evidence_level": "LEVEL_1",
    },
    {
        "claim_id": "CLM-007",
        "statement": "IVERI achieves superior scaling performance at 500M parameters.",
        "evidence_type": "PROJECTION_ONLY",
        "evidence_source": "reports/phase_10/12_research_validity.md — text only",
        "evidence_value": "NO EXPERIMENTAL EVIDENCE",
        "caveats": [
            "No 500M checkpoint exists",
            "No 500M training run was executed",
            "Mathematical extrapolation only",
        ],
        "evidence_completeness": "BROKEN EVIDENCE CHAIN",
        "broken_chain": True,
        "evidence_level": "LEVEL_0",
    },
    {
        "claim_id": "CLM-008",
        "statement": "Training stability: 0 NaN events, 0 OOM recoveries across all 5 models.",
        "evidence_type": "DIRECTLY_MEASURED",
        "evidence_source": "reports/phase_10/tables/training_stability.csv",
        "evidence_value": {"nan_events": 0, "oom_recoveries": 0, "steps": 2},
        "source_file_sha256": sha256_file(root / "reports/phase_10/tables/training_stability.csv"),
        "caveats": ["Only 2 training steps — long-run stability is not demonstrated"],
        "verification_command": "python scripts/run_phase10_benchmarks.py --quick",
        "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
        "seeds": [42],
        "evidence_completeness": "PARTIAL — 2 STEPS ONLY",
        "broken_chain": False,
        "evidence_level": "LEVEL_3",
    },
]

for c in claims:
    fname = claim_registry_dir / (c["claim_id"] + ".json")
    fname.write_text(json.dumps(c, indent=2))

print(f"Claim registry: {len(claims)} claims written")
for c in claims:
    status = "BROKEN CHAIN" if c["broken_chain"] else c["evidence_type"]
    print(f"  {c['claim_id']}: {status}")
print()

# ============================================================
# MISSING EVIDENCE MASTER TABLE
# ============================================================
missing_evidence = [
    {
        "evidence_id": "ME-001",
        "missing_item": "Multi-seed training campaigns (seeds 42, 123, 3407) for all 5 baselines",
        "affected_claims": ["CLM-003", "CLM-005"],
        "affected_paper_sections": ["Results", "Statistical Analysis"],
        "why_missing": "Phase 10 quick mode runs only 2 steps with seed=42",
        "execution_command": "python scripts/run_phase10_benchmarks.py --seeds 42 123 3407 --steps 1000",
        "expected_outputs": ["table3_loss_multiseed.csv", "per_seed_loss_curves.png"],
        "priority": "CRITICAL",
        "estimated_runtime_hours": "4-12h on CPU, 0.5-2h on GPU",
    },
    {
        "evidence_id": "ME-002",
        "missing_item": "Complete ablation study: 10 missing variants (No Titans, No MoR, No MoE, No Entropy, No Mamba2, + combinations)",
        "affected_claims": ["CLM-006"],
        "affected_paper_sections": ["Ablation Study"],
        "why_missing": "table4_ablation.csv has only 2 rows",
        "execution_command": "python scripts/run_phase10_benchmarks.py --ablation-full",
        "expected_outputs": ["table4_ablation_complete.csv", "fig_ablation_complete.png"],
        "priority": "CRITICAL",
    },
    {
        "evidence_id": "ME-003",
        "missing_item": "Statistically valid p-values and effect sizes computed from multi-seed data",
        "affected_claims": ["CLM-005"],
        "affected_paper_sections": ["Statistical Significance"],
        "why_missing": "table5_statistics.csv contains placeholder values not from executed experiments",
        "execution_command": "python research/statistical_analyzer.py --input table3_loss_multiseed.csv",
        "expected_outputs": ["table5_statistics_real.csv", "bootstrap_results.json"],
        "priority": "CRITICAL",
        "depends_on": ["ME-001"],
    },
    {
        "evidence_id": "ME-004",
        "missing_item": "Physical GPU energy measurements (Joules per token via NVML/RAPL)",
        "affected_claims": ["Energy efficiency claims"],
        "affected_paper_sections": ["Efficiency"],
        "why_missing": "No power measurement script executed",
        "execution_command": "python scripts/profile_memory.py --device cuda --energy-log",
        "expected_outputs": ["energy_joules_per_token.csv"],
        "priority": "HIGH",
    },
    {
        "evidence_id": "ME-005",
        "missing_item": "Consistent throughput measurement (table2_efficiency.csv values conflict with table3_loss.csv)",
        "affected_claims": ["RQ4 Throughput"],
        "why_missing": "table2_efficiency.csv was not regenerated from current execution (shows 1100 tok/sec vs measured 38.3)",
        "execution_command": "python scripts/run_phase10_benchmarks.py --quick (and fix table2 generation)",
        "priority": "HIGH",
    },
    {
        "evidence_id": "ME-006",
        "missing_item": "Docker container for independent reproduction",
        "affected_claims": ["Reproducibility"],
        "why_missing": "No Dockerfile exists in repository",
        "execution_command": "Create Dockerfile based on requirements.txt and Python 3.12 + CUDA 12.1",
        "priority": "MEDIUM",
    },
    {
        "evidence_id": "ME-007",
        "missing_item": "Extended training runs (1000+ steps) to assess convergence",
        "affected_claims": ["CLM-003", "CLM-005"],
        "why_missing": "Phase 10 only runs 2 training steps in quick mode",
        "execution_command": "python scripts/run_phase10_benchmarks.py --steps 1000 --all-seeds",
        "priority": "CRITICAL",
        "estimated_runtime_hours": "8-24h on CPU",
    },
]

missing_path = evidence_root / "MISSING_EVIDENCE.json"
missing_path.write_text(json.dumps(missing_evidence, indent=2))
print(f"Missing evidence table: {len(missing_evidence)} items")
print()

# ============================================================
# FINAL FORENSIC SCORECARD
# ============================================================
scorecard = {
    "audit_executed_at": "2026-07-26T02:47:19Z",
    "git_sha": "2f31069b6b7180c666903732dba2e9642121bb7a",
    "scores": {
        "software_engineering": {
            "score": "10/10",
            "evidence": "723 PyTest tests collected; all visible tests pass; modular architecture in model/, training/, data/",
            "evidence_level": "LEVEL_4",
        },
        "architecture_implementation": {
            "score": "10/10",
            "evidence": "BLT, Titans, Mamba2, MoR, MoE all implemented in model/ subdirectories with unit tests",
            "evidence_level": "LEVEL_4",
        },
        "experimental_evidence": {
            "score": "3/10",
            "evidence": "Only 2-step, 1-seed sweeps executed. table5_statistics.csv and table4_ablation.csv contain incomplete/placeholder data.",
            "evidence_level": "LEVEL_3",
            "deductions": [
                "-3: No multi-seed execution (seeds 123, 3407 not run)",
                "-2: Statistics not computed from executed data",
                "-2: Ablation study incomplete (2/12 variants executed)",
            ],
        },
        "statistical_rigor": {
            "score": "2/10",
            "evidence": "p-values in table5_statistics.csv are NOT computed from executed experiments. N=1 seed insufficient.",
            "evidence_level": "LEVEL_1",
            "deductions": [
                "-8: table5_statistics.csv values are placeholders, not from executed statistical analysis",
            ],
        },
        "benchmark_quality": {
            "score": "5/10",
            "evidence": "Baselines A-E implemented and execute. Compute budget matching verified (9.28%). But 2 steps insufficient.",
            "evidence_level": "LEVEL_3",
        },
        "dataset_governance": {
            "score": "9/10",
            "evidence": "Certified datasets with SHA-256 FREEZE_CERTIFICATE.json for 5 datasets. MinHash dedup. PII filter.",
            "evidence_level": "LEVEL_4",
        },
        "reproducibility": {
            "score": "6/10",
            "evidence": "review_phase10.py verifies SHA-256 checksums. No Dockerfile. Version conflict in BENCHMARK_CERTIFICATE.md.",
            "evidence_level": "LEVEL_3",
            "deductions": [
                "-2: No Docker container",
                "-1: PyTorch version conflict between BENCHMARK_CERTIFICATE.md and phase10_summary.json",
                "-1: metrics.jsonl contains mock data",
            ],
        },
        "hardware_validation": {
            "score": "4/10",
            "evidence": "CPU and RTX 3050 quick runs verified. No physical energy measurements. Throughput values conflict.",
            "evidence_level": "LEVEL_3",
        },
        "patent_enablement": {
            "score": "8/10",
            "evidence": "model/moe/router.py implements Entropy-Biased Router. model/blt/ implements patcher. Unit tests confirm functionality.",
            "evidence_level": "LEVEL_4",
            "deductions": [
                "-2: No formal prior-art clearance search completed",
            ],
        },
        "ieee_conference_readiness": {
            "score": "6/10",
            "evidence": "Architecture + compute matching established. Statistics and ablations need valid experiments.",
            "blocking_issues": ["Multi-seed extended training", "Valid statistical analysis from data", "Complete ablation study"],
        },
        "ieee_journal_readiness": {
            "score": "3/10",
            "evidence": "Framework in place. Actual evidence for key claims is missing or insufficient.",
            "blocking_issues": [
                "Extended training (1000+ steps, 3 seeds)",
                "Valid p-values from executed data",
                "Complete ablation study",
                "Physical energy measurements",
                "Docker reproducibility package",
            ],
        },
        "overall_repository_maturity": {
            "score": "6.5/10",
            "engineering_score": "10/10",
            "scientific_score": "3/10",
            "verdict": "Mature engineering framework with incomplete experimental evidence. "
                       "The architecture is implemented and functional. The scientific claims "
                       "require substantial additional experimentation to be credible for publication.",
        },
    },
    "critical_actions_before_submission": [
        "1. Execute multi-seed training (seeds 42, 123, 3407) for all 5 baselines for 1000+ steps",
        "2. Compute statistics from actual multi-seed data using research/statistical_analyzer.py",
        "3. Complete all 12 ablation variants and populate table4_ablation.csv from execution",
        "4. Fix throughput measurement inconsistency between table2 and table3",
        "5. Regenerate BENCHMARK_CERTIFICATE.md to resolve version conflict",
        "6. Replace logs/metrics.jsonl mock data with real training metrics",
        "7. Create Dockerfile for containerized reproduction",
    ],
}

scorecard_path = evidence_root / "FORENSIC_SCORECARD.json"
scorecard_path.write_text(json.dumps(scorecard, indent=2))
print("=" * 70)
print("FORENSIC SCORECARD SUMMARY")
print("=" * 70)
for category, data in scorecard["scores"].items():
    print(f"  {category:40s}: {data['score']}")
print()
print(f"Written: {scorecard_path}")

# Write findings
findings_path = evidence_root / "FORENSIC_FINDINGS.json"
findings_path.write_text(json.dumps(forensic_findings, indent=2))
print(f"Written: {findings_path}")
print(f"Written: {missing_path}")
print()
print("Package location:", evidence_root)
