# IVERI Core — Statistical Validation Report
Generated: 2026-07-26T07:04:37Z

## Critical Statistical Warning

> **UNDERPOWERED**: n=3 seeds per model is scientifically insufficient.
> Two-sample t-test at n=3 has approximately **5-15% power** for medium effects (d=0.5).
> This means the probability of CORRECTLY detecting a real effect is only 5-15%.
> All findings below are **preliminary observations**, NOT validated scientific claims.
> Minimum required for reliable inference: **n >= 8 seeds per model**.

## Tests Used
- Welch two-sample t-test (unequal variance assumed)
- Mann-Whitney U (non-parametric fallback)
- Shapiro-Wilk normality test (noting it has near-zero power at n=3)
- Levene's test for variance homogeneity
- Bootstrap 95% CI (10,000 resamples)
- Cohen's d (parametric effect size)
- Cliff's delta (non-parametric effect size)
- Estimated statistical power

## Pairwise Comparison Results (vs IVERI Full v0.91)

### IVERI_Full_v0.91_vs_Baseline_Dense_Transformer
**n_ref=3, n_comp=3** (UNDERPOWERED)

| Metric | REF (Full) | COMP | Delta | Welch-t p | MW-U p | Cohen d | Effect | Bootstrap 95%CI Contains 0? |
|--------|-----------|------|-------|-----------|--------|---------|--------|----------------------------|
| Final Loss | 3.1425 | 1.3664 | -1.7761 | <0.001 | =0.100 | 150.0445 | large | False |
| Throughput | 173 | 13016 | +7443.6% | <0.05 | =0.100 | - | - | - |
| VRAM (MB) | 828 | 394 | -52.5% | - | - | - | - | - |

**Verdict**: LIKELY_SIGNIFICANT (very large effect, low n)
**Power estimate**: 1.0 (at n=3 — unreliable)
**Bootstrap 95% CI (ref-comp loss diff)**: (1.756104, 1.793768)

### IVERI_Full_v0.91_vs_IVERI_No_Entropy
**n_ref=3, n_comp=3** (UNDERPOWERED)

| Metric | REF (Full) | COMP | Delta | Welch-t p | MW-U p | Cohen d | Effect | Bootstrap 95%CI Contains 0? |
|--------|-----------|------|-------|-----------|--------|---------|--------|----------------------------|
| Final Loss | 3.1425 | 3.1420 | -0.0005 | =0.931 | =0.700 | 0.0919 | negligible | True |
| Throughput | 173 | 186 | +8.0% | =0.440 | =0.700 | - | - | - |
| VRAM (MB) | 828 | 828 | -0.1% | - | - | - | - | - |

**Verdict**: NOT_SIGNIFICANT (negligible effect)
**Power estimate**: 0.0504 (at n=3 — unreliable)
**Bootstrap 95% CI (ref-comp loss diff)**: (-0.007949, 0.008976)

### IVERI_Full_v0.91_vs_IVERI_No_Titans
**n_ref=3, n_comp=3** (UNDERPOWERED)

| Metric | REF (Full) | COMP | Delta | Welch-t p | MW-U p | Cohen d | Effect | Bootstrap 95%CI Contains 0? |
|--------|-----------|------|-------|-----------|--------|---------|--------|----------------------------|
| Final Loss | 3.1425 | 3.1420 | -0.0006 | =0.926 | =0.700 | 0.0989 | negligible | True |
| Throughput | 173 | 271 | +57.2% | <0.01 | =0.100 | - | - | - |
| VRAM (MB) | 828 | 691 | -16.6% | - | - | - | - | - |

**Verdict**: NOT_SIGNIFICANT (negligible effect)
**Power estimate**: 0.0504 (at n=3 — unreliable)
**Bootstrap 95% CI (ref-comp loss diff)**: (-0.007804, 0.008912)

### IVERI_Full_v0.91_vs_IVERI_No_MoR
**n_ref=3, n_comp=3** (UNDERPOWERED)

| Metric | REF (Full) | COMP | Delta | Welch-t p | MW-U p | Cohen d | Effect | Bootstrap 95%CI Contains 0? |
|--------|-----------|------|-------|-----------|--------|---------|--------|----------------------------|
| Final Loss | 3.1425 | 3.1365 | -0.0060 | =0.514 | =0.700 | 0.737 | medium | True |
| Throughput | 173 | 359 | +108.2% | <0.01 | =0.100 | - | - | - |
| VRAM (MB) | 828 | 423 | -49.0% | - | - | - | - | - |

**Verdict**: INCONCLUSIVE (n=3 too small to distinguish)
**Power estimate**: 0.0777 (at n=3 — unreliable)
**Bootstrap 95% CI (ref-comp loss diff)**: (-0.006421, 0.018595)

### Phase-7: Warmup Sensitivity

| Warmup Steps | Final Loss | Explosions |
|-------------|------------|------------|
| 0 | 3.140556 | 2 |
| 25 | 3.142124 | 2 |
| 50 | 3.143753 | 2 |
| 75 | 3.142014 | 2 |
| 100 | 3.142990 | 2 |
| 200 | 3.142467 | 2 |

**Loss range**: 0.003197 (0.1017% of mean)
**All explosions identical**: True
**Conclusion**: EXPLOSION_INVARIANT: All warmup candidates produce identical explosion count. Loss variance across warmup candidates = 0.1017% of mean. Warmup schedule has NO measurable effect on training stability.
