# 09 MULTI-SEED RESULTS
**STATUS: FINAL — Data incorporated from EXP-019 (Valid Primary Baseline)**

The comparison satisfies the preregistered parameter-matching criterion of <=5% relative difference (3.07% delta). Both models trained at identical lr=3e-3.

## IVERI-Nano (10.03M params)
- seed42=3.1087, seed43=3.0325, seed44=2.9944, seed45=3.1308, seed46=3.0374
- **SUMMARY**: mean=3.0608, std=0.0569, ppl=21.34, diverged=0/5, latency=3098.1ms/step, VRAM=336MB, grad_norm range=6.98-9.26

## DenseTransformer_PreLN_SinusoidalPE (9.72M params, delta=3.07%)
- seed42=4.8396, seed43=4.2607, seed44=NaN(DIVERGED), seed45=7.2323, seed46=6.7163
- **SUMMARY**: mean_stable=5.7622, std_stable=1.4349, ppl_stable=318.05, diverged=1/5, latency=36.4ms/step, VRAM=206MB, grad_norm range=34.5-55.6
