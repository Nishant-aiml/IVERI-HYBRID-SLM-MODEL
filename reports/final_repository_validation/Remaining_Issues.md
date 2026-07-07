# Remaining Issues

**Generated:** 2026-07-06T18:39:39Z  
**Audit mode:** Independent re-verification (prior reports not trusted)  
**Ground truth:** `IVERI_PROJECT_MASTER.md`, `IVERI_DATA_PIPELINE_COMPLETE.md`  

## Open Issues

1. No dedicated inference/ package (generation only via IVERIModel.generate).
1. Stage 3B proprietary dataset directories are empty (.gitkeep only).
1. research/generate_reports.py is Phase 3.5 scratch with mock metrics — must not be used for publication (publication_manager is fail-closed).
1. Only phase_0_plan.md exists; Phases 1–6.3 implementation plans missing.
1. Campaign metadata labels Phase 5.0 while artifacts live under reports/phase_6_3/.
