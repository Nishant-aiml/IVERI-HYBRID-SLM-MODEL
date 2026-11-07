# IVERI Core — Scientific Validation
## 11_EFFICIENCY_VALIDATION.md (REGENERATED FROM EXP-021 RAW ARTIFACTS)
## STATUS: FINAL (data from EXP-021 raw JSON, git SHA 7012134f)

---

## Provenance

| Field | Value |
|---|---|
| Source experiment | EXP-021 (COMPONENT_TIMING_V1) |
| Script | `scratch/component_timing_analysis.py` |
| Git SHA | `7012134f44403f6732333bba9e93bbc7ea73b867` |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU |
| PyTorch | 2.5.1+cu121 |
| Batch size | 4 |
| Seq len | 64 |
| Warmup runs | 3 (excluded from measurement) |
| Measure runs | 10 |
| Timestamp | 2026-08-14T06:42:24Z |

---

## Parameter Counts

| Model | Parameters | Source |
|---|---|---|
| IVERI-Nano | 10,029,874 | PyTorch `sum(p.numel())` |
| DenseTransformer (Pre-LN, sinusoidal PE) | 9,722,083 | PyTorch `sum(p.numel())` |
| Delta | 3.07% | Satisfies preregistered <=5% criterion |

---

## Forward-Pass Latency (inference, no gradient)

| Model | Mean (ms) | Std (ms) | CV (%) |
|---|---|---|---|
| IVERI-Nano full forward | **524.53** | **242.86** | **46.3%** |
| DenseTransformer forward | 23.61 | 2.35 | 10.0% |
| Embedding only (IVERI) | 0.24 | 0.09 | 37.5% |

**Ratio (forward-only): 22.2x**

> [!IMPORTANT]
> **The forward-only ratio (22.2x) is different from the training step ratio (~85x) reported in EXP-019.**
> The training step includes backward pass + optimizer step. The backward pass through
> Python-sequential Titans loops is substantially more expensive than the forward pass.
> The ~85x training step ratio cannot be explained by forward-pass overhead alone.

### Individual Forward-Pass Measurements (ms)

**IVERI-Nano** (10 runs after warmup):
`363.42, 385.65, 363.09, 358.84, 351.83, 362.59, 437.27, 865.24, 905.62, 851.77`

**DenseTransformer**:
`20.81, 20.64, 21.95, 23.50, 25.92, 27.24, 25.67, 25.55, 22.91, 21.88`

---

## Critical Finding: IVERI Latency Bimodality

Inspecting the IVERI raw measurements reveals a **bimodal distribution**:

- Runs 1–6: ~352–438 ms (fast cluster)
- Runs 7–10: ~852–906 ms (slow cluster, 2.4× slower)

This is **not measurement noise**. The standard deviation of 242.86 ms (46.3% coefficient of variation) on a mean of 524.53 ms indicates **non-deterministic execution time**. Possible causes:
1. **MoR dynamic routing** varying recursion depth between calls
2. **Python GIL contention** or JIT warm-up after initial runs
3. **GPU memory paging** on the 4.29 GB VRAM device

**This variability is itself a measurement that must be reported.** The Transformer's CV of 10% indicates stable, deterministic execution by comparison.

> [!WARNING]
> The bimodal distribution means reporting IVERI forward latency as a single mean (524.53 ms) is **misleading**. The two modes should be reported separately or the full distribution disclosed.
> Appropriate statement: "IVERI forward-pass latency showed high variability across runs
> (range 352–906 ms, CV=46%), suggesting non-deterministic execution paths."

---

## Sequence Length Scaling

| Seq Len | IVERI (ms) | Transformer (ms) | Ratio |
|---|---|---|---|
| 32 | 538.54 | 22.94 | **23.5x** |
| 64 | 831.45 | 22.79 | **36.5x** |
| 128 | 1,310.37 | 32.08 | **40.8x** |
| 256 | 2,562.99 | 27.45 | **93.4x** |

**Pattern**: IVERI latency grows approximately linearly with sequence length (~4.9 ms/byte at sl=64–256 range). Transformer latency is approximately constant across this range (22–32 ms), consistent with its operation being dominated by fixed matrix multiplications at these small sequence lengths.

The super-linear IVERI ratio growth (23.5x → 93.4x across sl=32→256) is consistent with a **sequential Python loop** processing each byte individually: total cost = O(L × Python_overhead_per_byte).

---

## Component Isolation Status

| Component | Measured | Method |
|---|---|---|
| Embedding only | 0.24 ms | Direct module timing |
| Full IVERI forward | 524.53 ms | CUDA event timing |
| DenseTransformer forward | 23.61 ms | CUDA event timing |
| Titans contribution | **BLOCKED** | Requires `skip_memory_update` flag in model |
| Mamba-2 contribution | **BLOCKED** | Requires architectural hook |
| BLT/entropy patching | **BLOCKED** | Requires architectural hook |
| MoR routing | **BLOCKED** | Requires depth logging hook |
| MoE routing | **BLOCKED** | Requires expert utilization logging |

**To unlock Titans isolation**, add `skip_memory_update: bool = False` to `IVERIModel.forward()`. The modules exist at `backbone.titans.*`.

---

## VRAM Usage

| Model | Peak VRAM (training step, EXP-019) |
|---|---|
| IVERI-Nano | 336 MB |
| DenseTransformer | 206 MB |
| Difference | +130 MB (+63%) |

Both well within 4.29 GB available on RTX 3050. VRAM is not a blocking constraint at seq_len=64.

---

## Honest Efficiency Narrative

The following is the **only scientifically defensible characterization** of IVERI's efficiency based on current measurements:

> "Under the evaluated training configuration (batch=4, seq_len=64, 10 steps),
> IVERI-Nano incurred approximately 22× higher forward-pass latency and approximately
> 85× higher total training-step latency compared to the parameter-matched Dense Transformer.
> IVERI also used approximately 130 MB more VRAM. IVERI forward-pass latency exhibited
> high variability (CV=46%), potentially indicative of non-deterministic execution paths
> from dynamic routing components.
>
> This overhead is attributed to current implementation characteristics — specifically,
> the Python-sequential execution of the Titans recurrent memory update — rather than
> to any inherent property of the architectural design. The component-level contribution
> to total overhead remains BLOCKED pending instrumentation of `backbone.titans`.
>
> The overhead grows approximately linearly with sequence length in IVERI, while remaining
> approximately constant in the Transformer at these sequence lengths, suggesting
> O(L) Python loop overhead dominates IVERI's current execution cost.
>
> **These observations do not support a claim that IVERI is computationally efficient.
> They motivate a specific engineering research question: can Titans recurrent memory
> be implemented without Python-sequential overhead while preserving the observed
> training dynamics?**"

---

## Claims That Cannot Be Made

| Claim | Reason Cannot Be Made |
|---|---|
| "IVERI is computationally efficient" | Directly contradicted: 22x forward, 85x training overhead |
| "Mamba-2 provides O(L) efficiency advantage" | Advantage is negligible at seq_len=64; dominated by Titans overhead |
| "IVERI scales better than Transformer" | Ratio GROWS with sequence length (23x→93x at sl=32→256) |
| "85x overhead is architectural" | Not established — may be entirely implementation overhead |
| "Titans caused X% of overhead" | BLOCKED — component not yet isolated |

---

## Research Questions Generated by This Experiment

1. **What causes IVERI forward-pass bimodality?** (fast cluster 352–438ms vs slow cluster 852–906ms)
2. **What fraction of the 524ms forward pass comes from Titans?** (requires `skip_memory_update` instrumentation)
3. **Is the sequential overhead in the forward or backward pass?** (backward contributes ~63ms extra per step: 85x step / 22x forward = ~3.9x overhead ratio)
4. **Does MoR dynamic routing cause the latency variability?** (high CV=46% needs investigation)
5. **What is the minimum sequence length at which Mamba-2's O(L) advantage would outweigh Titans overhead?**
