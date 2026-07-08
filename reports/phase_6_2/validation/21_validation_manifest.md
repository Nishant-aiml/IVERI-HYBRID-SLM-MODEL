# IVERI Core Phase 6.2 Validation Report — Validation Manifest

- **Manifest Version**: `1`
- **Schema Version**: `1.0`

## 1. Scope
This manifest provides a comprehensive cryptographic verification and configuration receipt of the Phase 6.2 Engineering Validation. It guarantees the integrity of the validated codebase state, data pipelines, SQLite experiments database, and all generated validation reports.

## 2. Freeze & Version Metadata
- **Engineering Freeze Status**: `PASS`
- **Scientific Freeze Status**: `Pending`
- **Product Version**: `1.6.0`
- **Architecture Version**: `1.0`
- **Freeze Tag**: `v1.0-engineering-freeze`

## 3. Environment Metadata
- **Git Commit Hash**: `1a47d4de6b1fb455291dd01b21c2e59740a7cd38` (Verified code + reports tree)
- **Audit Timestamp**: `2026-07-08T05:43:00Z`
- **Operating System**: `Windows 11` (win32)
- **Python Version**: `3.12.10`
- **PyTorch Version**: `2.5.1+cu121`
- **CUDA Toolkit Version**: `12.1`
- **Target GPU**: `NVIDIA GeForce RTX 3050 Laptop GPU`

## 4. Repository Cleanliness State
- **Git Branch**: `main`
- **Git Status**: `clean` (working tree clean)
- **Untracked Files**: `0`
- **Modified Files**: `0`

## 5. Automated Validation Results

### Test Suite Execution
- **Tests Executed**: `687`
- **Passed**: `683`
- **Skipped**: `4` (FlashAttention-2 and custom cuda scan backends, marked as: `NOT VERIFIED (Windows fallback)`)
- **Failed**: `0`
- **Duration**: `100.94 seconds`
- **pytest Command**: `python -m pytest tests/`

### Runtime Validation Audit
- **Runtime audit**: `64 / 64 passed`
- **Pass Rate**: `100%`
- **Report Path**: `scratch/freeze_audit_results.json`

### Benchmark Measurements (GeForce RTX 3050)
- **Forward Latency**: `0.038 seconds`
- **Backward Latency**: `0.038 seconds`
- **Peak VRAM Allocated**: `175.1 MB`
- **Reserved VRAM**: `258.0 MB`
- **Throughput**: `1686 bytes/sec`

---

## 6. Cryptographic Source & Database Hashes (SHA-256)
These signatures guarantee that the source files, database, and configurations are frozen exactly as validated:

| Target File Path | SHA-256 Signature Hash | Verification Status |
| :--- | :--- | :--- |
| `VERSION` | `4c98106bedc8dfa99aa70c55db63b226827e88fc1c8dba7b896f7c43beaca7f6` | **VERIFIED** |
| `pyproject.toml` | `d8716d2e672b1a9f1e3badb278c11f7243c9e3e6bd5d22d64f2bf95e2121bca2` | **VERIFIED** |
| `core/constants.py` | `0182466cbaa2a1aa2b5183a16a57407f6072d44be155c9696f254f45c89b4cf7` | **VERIFIED** |
| `configs/base_config.py` | `bff95a3cbfe0680c6d1a97a6de280d3e13d9064bb22c7a67ddbb024a79659bd9` | **VERIFIED** |
| `model/iveri_core.py` | `a1f1fe55f5f4311defe810094e6eafc5059127137882b7edb544ebab36c70e11` | **VERIFIED** |
| `model/backbone.py` | `a10f657f99bf6f67299d45355e68b1ffbae60c6084cc6835b88dec9edd9d53dc` | **VERIFIED** |
| `training/pretrain_runner.py` | `25edca825b624f0e8ce7bcab1c5dd1c56fd51fe9747b3bd9cdd2e827c3251f41` | **VERIFIED** |
| `training/checkpointing.py` | `3940e08fb6a7e93f55dc7144f41a5e1afc9d9ca9d1cbb3be85ea8bf3d1e6ad85` | **VERIFIED** |
| `research/campaign_runner.py` | `b25c43b5b5e95b937e05367893eb40911732fec3fb5c1fc5376698f79d22b841` | **VERIFIED** |
| `research/experiments.db` | `72dc575b6ed16275e0a4b4ccdebc9314a673ce0dd79509215fbc6135b999ba30` | **VERIFIED (Fail-closed on dev logs)** |

---

## 7. Cryptographic Report Hashes (SHA-256)
These signatures guarantee the integrity of all compiled validation documents under `reports/phase_6_2/validation/`:

| Report File Name | SHA-256 Signature Hash | Verdict |
| :--- | :--- | :--- |
| **01_executive_summary.md** | `defc293e905e03dc58c5dd23edbf290f4b41721e25cd536b6bf208f2baaa0f3a` | GO WITH MINOR ISSUES |
| **02_compliance_matrix.md** | `35f3ad08dd9065b677f5676e61ec4e7f4b66ccd324cff4b16f85cabdb15c0947` | COMPLIANT |
| **03_architecture_validation.md** | `46c32726e644d4b3820127895b2e51926580b6ae48fdd19dc15eaa7d5b4326b6` | PASS |
| **04_data_pipeline_validation.md** | `88c713cf0c4fdf6f92ca248a269de896ca415a5a783b02eb0fe2f8c0773096ef` | PASS |
| **05_blt_validation.md** | `d372e3084ebb3f242cdca2b3c2b78f6f1103c24d99f5e9c4769133e9c027992e` | PASS |
| **06_titans_validation.md** | `6f132111e351f27a7e71d36222d09535078a344fe6f62f40a44e368c686a8db8` | PASS |
| **07_mor_validation.md** | `b5abcf6f5273dcd61f476df1ae823e00b855f7633781789bb2f1bcebeca4896d` | PASS |
| **08_mamba2_validation.md** | `521adf3ebbe1b7077f9dfa0edcde9336b570e93caefc60cfa1f763a7e489b134` | PASS |
| **09_moe_validation.md** | `0e9bb7fdcf201e42d3ece79cdd4ddee0cbecaf9851b08f95d556ea48ac5aaff7` | PASS |
| **10_backbone_integration.md** | `6638bf7c36316dd5c6b672aa021f874542af2dc4975aa35ba4ddf7d07d38a873` | PASS |
| **11_end_to_end_integration.md** | `044788f9ae20aa09b6450f91168efa0c6fd13397b6a13cef78cc56cc8de22bce` | PASS |
| **12_optimization_performance.md** | `2b4a236e52b108a6d13aa6b950a3668457cfc6664ecf697586e556da1b946f2f` | PASS WITH LIMITATIONS |
| **13_cuda_memory.md** | `1647664a055af21c9437f767c90e2282f1c19406dfbf807af016ed2b07b466e6` | PASS |
| **14_engineering_quality.md** | `47e906188fb0dd26c15fac951ebded840bcc77f26c56216c1b2bb3babebace95` | EXCELLENT |
| **15_stress_testing.md** | `1eb905054d98530730e6d77c81e6d3cf7518b93fb8efcf6a8955e8c9863630d0` | PASS |
| **16_regression.md** | `a750215c34dfa2d268a848398adb0a9cd68c6390588f4454dfb412127fe34436` | PASS |
| **17_dependency_environment.md** | `8b7e0618cca6d8a56ee9be81079481da3e3a5c2b08b5f394c6685216d53ef908` | PASS |
| **18_remaining_issues.md** | `4094f841ba76c4807afdfdbe66172dad211608ae5f43ee2836c7f2a5f52f831c` | MINOR ISSUES FOUND |
| **19_risk_assessment.md** | `9b6cf969802c93fb2e6039a27f478afe2a027e90d43586dc5fa87b94c233b620` | LOW-MEDIUM RISK |
| **20_readiness_certificate.md** | `f4c7f9583cb040e4ddbcb1d588f60192c3a0c32eb31b40abea3d61b395153a10` | GO WITH MINOR ISSUES |

## 8. Verification Verdict
**GO WITH MINOR ISSUES**
Signed and cryptographically cataloged.
