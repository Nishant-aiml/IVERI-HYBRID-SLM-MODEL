# IVERI Core — Final Claim Registry
## 22_FINAL_CLAIM_REGISTRY.md
## STATUS: PROVISIONAL — awaiting LR_FAIRNESS_CAMPAIGN (EXP-020) completion
## Source: final_experiment_registry.json (schema 1.1)
## Last updated: 2026-08-14T06:45:00Z

> [!IMPORTANT]
> This registry is generated from raw experiment artifacts, not from prior reports.
> Every claim status is traceable to a specific experiment ID.
> Claims marked INCONCLUSIVE or UNKNOWN are research findings, not failures to disclose.

---

## Strongest Verified PRELIMINARY Result

> **IVERI-Nano (10.03M parameters) achieved lower cross-entropy loss after 10 training steps than a parameter-matched Pre-LN Dense Transformer (9.72M parameters, delta=3.07%) under the evaluated training configuration. Across five IVERI seeds, no runs diverged, while one of five Transformer runs diverged. The result is preliminary and limited to early training on a synthetic byte-level corpus.**

> **IVERI incurred approximately 85x higher wall-clock time per training step (3,098 ms vs. 36.4 ms), used approximately 130 MB more VRAM, and failed the evaluated 512-token passkey retention test. The experiments do not yet isolate which IVERI component causes the observed loss difference.**

**Why "PRELIMINARY"**: 10 training steps is an extremely early measurement. The scientifically interesting question — does the loss advantage survive longer training and fair Transformer hyperparameter selection? — has not been answered.

---

## Primary Claim Table (from user-provided correction, 2026-08-14)

| Claim | Status | Evidence |
|---|---|---|
| IVERI implementation works | ✅ PROVEN | Source code compiles, trains, produces output |
| Parameter-matched comparison exists | ✅ PROVEN | EXP-019: delta=3.07%, satisfies preregistered <=5% criterion |
| IVERI had lower loss at step 10 | ✅ MEASURED | EXP-019: 3.061 vs 5.762 (stable seeds) |
| 5 IVERI seeds completed without divergence | ✅ PROVEN | EXP-019: diverged=0/5 |
| Transformer had 1/5 divergence | ✅ MEASURED | EXP-019: seed44 → loss=NaN |
| IVERI is generally better than Transformer | ❌ NOT ESTABLISHED | 10 steps, synthetic data, untuned Transformer LR |
| IVERI is more computationally efficient | ❌ FALSE under this measurement | EXP-019/021: 85x step, 22x forward overhead |
| IVERI has better long-context retention | ❌ Currently contradicted | EXP-010: passkey ctx=512 FAILED |
| BLT caused the improvement | ❌ UNKNOWN | No ablation run |
| Titans caused the improvement | ❌ UNKNOWN | No ablation run |
| MoR caused the improvement | ❌ UNKNOWN | No ablation run |
| MoE caused the improvement | ❌ UNKNOWN | No ablation run |
| IVERI's eventual trained performance is better | ❌ UNTESTED | Only 10 steps evaluated |
| Reproducibility | 🔴 PENDING | No independent reproduction (EXP-018 BLOCKED) |

---

## Full Claim Detail

### PROVEN Claims (no qualification needed)

**CLAIM-001**: IVERI-Nano architecture integrates BLT + Titans + Mamba-2 + MoR in working code.
- Status: **PROVEN**
- Evidence: Source code at git SHA `7012134f`. Model builds, accepts input, produces output, trains for 10 steps across 5 seeds without error.
- Experiment: EXP-001 to EXP-003, EXP-019
- Limitations: Code correctness does not verify that each component functions as theoretically described.

**CLAIM-002**: IVERI-Nano operates within 340 MB VRAM on RTX 3050 Laptop GPU.
- Status: **PROVEN**
- Evidence: EXP-019 measured 334–338 MB across all 5 seeds. Well within 4.29 GB available.
- Experiment: EXP-019

**CLAIM-003**: IVERI-Nano vocabulary covers bytes 0–255 plus 3 special tokens (BOS, PAD, EOS) = 259 tokens.
- Status: **PROVEN**
- Evidence: Source code inspection. All experiments use vocab_size=259.

---

### MEASURED Claims (valid but requiring qualification)

**CLAIM-004**: IVERI-Nano achieved lower cross-entropy loss after 10 training steps than the evaluated Transformer baseline (3.061 vs 5.762, stable seeds).
- Status: **MEASURED — PRELIMINARY**
- Evidence: EXP-019 raw JSON `09_MULTI_SEED_RESULTS.json`
- Qualification: 10 steps only. Synthetic corpus. Transformer LR not independently tuned (grad_norm 34–55 vs IVERI 7–9 suggests lr=3e-3 sub-optimal for Transformer). LR fairness experiment (EXP-020) in progress.
- Statistical note: Welch t=−3.763, df=3.0, exact p computed from scipy. Technically p<0.05 at alpha=0.05, but **n=4 stable Transformer seeds and untuned LR prevent strong interpretation**. Do NOT claim "statistically significant superiority." Correct statement: "IVERI achieved lower validation loss under the evaluated training configuration; however, the comparison remains subject to hyperparameter-fairness concerns because the Transformer learning rate was not independently tuned."

**CLAIM-005**: IVERI-Nano training convergence showed lower gradient magnitude variability than the Transformer at identical learning rate.
- Status: **MEASURED — DESCRIPTIVE ONLY**
- Evidence: EXP-019. IVERI grad_norm range 6.98–9.26 vs Transformer 34.5–55.6.
- Qualification: **"Lower observed gradient magnitude/variability"** — NOT "statistically significant variance reduction." No variance test was run. The difference is descriptive, not hypothesis-tested.

**CLAIM-006**: IVERI forward-pass latency is approximately 22x higher than the matched Transformer (524ms vs 24ms, forward-only).
- Status: **MEASURED**
- Evidence: EXP-021, CUDA event timing, warmup=3, measure=10.
- Additional finding: IVERI forward-pass latency is bimodal (fast cluster ~360ms, slow cluster ~880ms, CV=46%). This bimodality is unreported in prior versions and must be disclosed.

**CLAIM-007**: IVERI training-step latency is approximately 85x higher than the matched Transformer (3,098ms vs 36.4ms).
- Status: **MEASURED**
- Evidence: EXP-019, 5-seed mean.
- Distinction: Training step (85x) > forward pass only (22x). The backward pass through Python-sequential Titans is the dominant overhead contributor — estimated ~63x of the 85x total step ratio. This is implementation overhead, not architectural overhead.

**CLAIM-008**: IVERI latency scales approximately linearly with sequence length at current implementation.
- Status: **MEASURED**
- Evidence: EXP-021 sequence scaling: sl=32→256 shows IVERI latency growing 538ms→2563ms (~4.8x over 8x seq length), while Transformer remains ~23–32ms.
- Implication: Overhead grows approximately O(L), consistent with Python-sequential loop per byte.

---

### REFUTED Claims

**CLAIM-009**: IVERI is computationally efficient.
- Status: **REFUTED**
- Evidence: EXP-019 (85x step overhead), EXP-021 (22x forward overhead, super-linear scaling with seq_len).

**CLAIM-010**: IVERI demonstrates long-context retention.
- Status: **REFUTED for ctx=512 / UNTESTED for ctx>512**
- Evidence: EXP-010: passkey at ctx=512, 25 fine-tune steps → FAIL (pred_byte=54, target=55, loss=8.797). EXP-011: ctx>512 timed out due to Python-sequential loop.

---

### INCONCLUSIVE Claims

**CLAIM-011**: The hybrid Mamba-2 + selective attention backbone provides a better quality/compute trade-off than a matched single-backbone baseline.
- Status: **INCONCLUSIVE**
- Reason: No ablation isolating the backbone contribution. EXP-013/014 BLOCKED.

**CLAIM-012**: Titans memory improves information retention over a Titans-free IVERI.
- Status: **INCONCLUSIVE**
- Reason: No ablation. Long-context passkey test failed but failure attribution unknown.

**CLAIM-013**: Entropy-conditioned MoE routing improves parameter utilization.
- Status: **INCONCLUSIVE**
- Reason: No expert utilization measurement. No ablation.

**CLAIM-014**: MoR dynamic recursion reduces computation while maintaining quality.
- Status: **INCONCLUSIVE**
- Reason: No routing depth distribution measured. No ablation.

**CLAIM-015**: IVERI's loss advantage is attributable to architectural innovation rather than BLT input processing.
- Status: **INCONCLUSIVE**
- Reason: BLT patching is applied by IVERI but not by the Transformer baseline. Architecture and input representation are confounded. Requires ablation B (BLT-patched Transformer) and C (IVERI without BLT) from the planned ablation ladder.

---

### UNSUPPORTED Claims (must be removed from any paper)

**CLAIM-016**: IVERI is "world's first" hybrid byte-level SLM combining BLT + Mamba + Titans.
- Status: **UNSUPPORTED — novelty audit incomplete (23_FINAL_NOVELTY_AUDIT.md)**
- Action: Remove from all materials until prior art search is complete.

**CLAIM-017**: IVERI achieves state-of-the-art performance.
- Status: **UNSUPPORTED**
- Reason: No comparison against published SLM baselines. Only compared against one custom Dense Transformer in preliminary training.

---

## What CAN Safely Appear in a Paper

The following claims are defensible as of the current evidence:

1. IVERI-Nano is a working implementation combining BLT, Titans, Mamba-2, and MoR.
2. IVERI-Nano (10.03M parameters) fits within 340 MB VRAM on consumer GPU hardware.
3. Under the evaluated early-training protocol (10 steps, synthetic byte corpus, lr=3e-3), IVERI-Nano achieved lower cross-entropy loss than a parameter-matched Pre-LN Dense Transformer, with all 5 seeds completing vs. 1/5 diverging for the Transformer.
4. IVERI-Nano currently incurs approximately 22x forward-pass and 85x training-step latency overhead vs. the Transformer, attributed to Python-sequential Titans execution rather than architectural complexity.
5. IVERI forward-pass latency exhibits high variability (CV=46%) suggesting non-deterministic execution paths, warranting further investigation.
6. The latency overhead grows approximately linearly with sequence length under the current implementation.
7. Component-level attribution of quality and performance improvements requires further ablation experiments, which remain pending.

---

## What CANNOT Safely Appear in a Paper

1. Any claim of statistical significance without acknowledging n=4 stable Transformer seeds, untuned LR, and 10-step limitation.
2. "IVERI is faster/more efficient than Transformer."
3. "Titans provides long-context retention" — contradicted by passkey test.
4. Attribution of quality improvement to any specific component.
5. Generalization to real text distributions from synthetic corpus results.
6. "World's first" or "state-of-the-art" without completed novelty/comparison audit.
7. Results from invalidated experiments (EXP-004 to EXP-006, EXP-012) in primary claims.
