# IVERI Core — EXP-020: LR Fairness Campaign Results
## 20_LR_FAIRNESS_CAMPAIGN.md
## STATUS: COMPLETE — results supersede EXP-019 comparative interpretation
## Raw artifact: EXP020_LR_FAIRNESS_CAMPAIGN.json
## Git SHA: 7012134f44403f6732333bba9e93bbc7ea73b867
## Timestamp: 2026-08-14T06:56:14Z

---

## Provenance

| Field | Value |
|---|---|
| Experiment ID | LR_FAIRNESS_CAMPAIGN_V1 |
| Script | `scratch/lr_fairness_campaign.py` |
| Protocol | Preregistered — LR grid and train/val/test split defined before evaluation |
| Corpus hash | `4ac89968d4a4a22f` (same corpus as EXP-019) |
| Train/Val/Test split | 6,072 / 2,024 / 2,024 bytes (60/20/20) |
| LR grid (preregistered) | [1e-4, 3e-4, 1e-3, 3e-3] |
| Val seeds | [100, 101, 102] |
| Test seeds | [42, 43, 44, 45, 46] (same as EXP-019) |
| Param delta | 3.07% — satisfies preregistered <=5% criterion |

---

## Phase 1 — Transformer LR Selection (Validation Set, Test Untouched)

| LR | Val mean loss | n_valid | n_diverged | Notes |
|---|---|---|---|---|
| 1e-4 | 3.2188 | 3/3 | 0 | Too slow to converge in 10 steps |
| **3e-4** | **2.4858** | **3/3** | **0** | **SELECTED — lowest val loss** |
| 1e-3 | 2.5352 | 3/3 | 0 | Close second |
| 3e-3 | 4.3852 | 3/3 | 0 | Worst of four — used in EXP-019 |

**Selected Transformer LR: 3e-4.** LR frozen before test set was evaluated.

> [!CAUTION]
> **lr=3e-3 (used in EXP-019) is the worst-performing Transformer LR across the full grid.**
> The Transformer at lr=3e-4 achieves val_mean=2.486 vs. 4.385 at lr=3e-3 — a difference
> of 1.9 nats. EXP-019's Transformer arm was trained at its worst hyperparameter configuration.

---

## Phase 1 — IVERI LR Audit (Same Protocol)

| LR | Val mean loss | n_valid | n_diverged | Notes |
|---|---|---|---|---|
| 1e-4 | 5.0423 | 3/3 | 0 | Far too slow |
| 3e-4 | 4.3687 | 3/3 | 0 | Still slow |
| 1e-3 | 3.0859 | 3/3 | 0 | Near-optimal |
| **3e-3** | **3.0098** | **3/3** | **0** | **Best — confirms EXP-019 LR was fair for IVERI** |

**IVERI LR audit finding:** lr=3e-3 used in EXP-019 is confirmed as IVERI's optimal
learning rate by the validation protocol. **IVERI was NOT disadvantaged in EXP-019.**

The asymmetry is: IVERI performed best at lr=3e-3 (validation-confirmed), while
the Transformer performed worst at lr=3e-3. EXP-019 was therefore inadvertently
fair to IVERI while simultaneously sub-optimal for the Transformer.

---

## Phase 2 — Final Test (Frozen LRs, Untouched Test Set)

| Model | Frozen LR | Test seed results | Mean | Std | Diverged |
|---|---|---|---|---|---|
| IVERI-Nano | 3e-3 | 3.0701, 3.0752, 3.1130, 3.1876, 3.1133 | **3.1118** | 0.0470 | 0/5 |
| DenseTransformer | **3e-4** | 2.6472, 2.4773, 2.9374, 2.8129, 2.6504 | **2.7051** | 0.1760 | 0/5 |

---

## Statistical Analysis (Exact, from scipy.stats.ttest_ind Welch)

| Statistic | Value |
|---|---|
| Welch t | 4.9944 |
| Exact p (two-tailed) | **0.005282** |
| Cohen's d | **3.157** (very large) |
| n_IVERI | 5 |
| n_Transformer | 5 |
| Mean difference | 0.4067 nats (Transformer lower = better) |

---

## Corrected Fair Conclusion

> [!IMPORTANT]
> **The script output labelled this INCONCLUSIVE due to a threshold bug (required >0.5 nat gap, actual gap=0.407 nats). This label is INCORRECT. The correct conclusion is stated below.**

**After fair hyperparameter selection via a preregistered validation protocol, the parameter-matched Pre-LN Dense Transformer (lr=3e-4, selected by validation) achieved lower test cross-entropy loss than IVERI-Nano (lr=3e-3, confirmed as optimal by validation): Transformer 2.705 ± 0.176 vs. IVERI 3.112 ± 0.047. The difference is 0.407 nats (Welch t=4.994, p=0.005, Cohen's d=3.157, n=5/5).**

**Caveats that must accompany this conclusion:**
1. n=5 per group — very small. The p-value and effect size are directionally clear but statistically fragile at this sample size.
2. 10 training steps only — this is early-training dynamics, not convergence quality.
3. Synthetic byte corpus — generalizability to real text unknown.
4. BLT/input representation confound not resolved — IVERI uses entropy BLT patching internally; Transformer sees raw bytes.
5. No ablation run — component contributions unknown.

---

## Impact on EXP-019 Interpretation

EXP-019 result: **IVERI 3.061 vs Transformer 5.762** (IVERI wins by 2.701 nats)

EXP-020 result: **IVERI 3.112 vs Transformer 2.705** (Transformer wins by 0.407 nats)

The difference between these outcomes is entirely explained by the Transformer LR:
- EXP-019 Transformer LR: 3e-3 (validation-confirmed worst)
- EXP-020 Transformer LR: 3e-4 (validation-confirmed best)

**Conclusion for the research record:** The IVERI advantage observed in EXP-019 was largely or entirely attributable to the Transformer being evaluated at a sub-optimal learning rate. Under a fair protocol with validation-selected hyperparameters, the Transformer outperforms IVERI under this early-training configuration.

**EXP-019's comparative claim must be WITHDRAWN.** EXP-019 data remains valid for other purposes (IVERI stability, IVERI latency, IVERI convergence profile) but cannot be used to claim IVERI superiority over a fairly-tuned Transformer.

---

## Remaining Confound

The BLT input processing confound is NOT resolved by this experiment. Both EXP-019 and EXP-020 compare (IVERI architecture + BLT patching) against (raw-byte Transformer). If BLT patching provides a representation advantage, its contribution to the loss difference in either direction remains unquantified.

---

## What This Means for the Research Program

The falsification campaign has done its job correctly. The hypothesis that IVERI achieves better early-training quality than a matched Transformer **has been falsified under fair conditions**.

The remaining scientifically valuable questions are:
1. **Why does IVERI converge to a higher loss than the Transformer?** Is it BLT overhead, the recurrent Titans bottleneck, or the hybrid routing complexity?
2. **Is there any regime (longer training, real data, long-context tasks) where IVERI has an advantage?**
3. **What is the Titans component's contribution to the overhead?** (Next: EXP-021b)
4. **Can IVERI be targeted-modified to match the Transformer's quality at lower overhead?**

These questions now define the research program — not the original claim of superiority.
