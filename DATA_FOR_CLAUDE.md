# Master Data & File Directory for Claude

> **Project:** IVERI-HYPERION (Entropy-Driven Neural State Network - ENSN)  
> **Repository Root:** `c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\`  
> **Date:** October 2026  
> **Status:** Multi-seed empirical validation completed; full academic papers and LaTeX compiled.

---

## 1. Top 5 Files to Give to Claude (Upload Priority)

If you are uploading files to Claude (e.g., Claude Projects or chat attachments), give Claude these **5 core files**:

| Priority | File Path | Why Claude Needs It |
| :---: | :--- | :--- |
| **1** | [`PAPERS/STATE_AND_MEMORY_COMPREHENSIVE_GUIDE.md`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/PAPERS/STATE_AND_MEMORY_COMPREHENSIVE_GUIDE.md) | **The Master Explanation:** Has all Mermaid diagrams, formulas, commented PyTorch & Triton codes, and authenticated benchmarks in plain English. |
| **2** | [`reports/final_validation/multi_seed_statistical_validation.json`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/reports/final_validation/multi_seed_statistical_validation.json) | **Raw Empirical Ground Truth:** Multi-seed runs across seeds 42, 43, 44, mean $\pm$ std, Welch's $t$-test ($p < 10^{-7}$), and paired $t$-test. |
| **3** | [`PAPERS/STATE_AND_MEMORY_ARE_ALL_YOU_NEED.tex`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/PAPERS/STATE_AND_MEMORY_ARE_ALL_YOU_NEED.tex) | **Camera-Ready Academic Paper:** Full 15-page LaTeX paper formatted with TikZ, booktabs tables, and reverse-adjoint derivations. |
| **4** | [`model/v1/iveri_v1.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/model/v1/iveri_v1.py) | **Architecture Code:** The real PyTorch implementation of S4D SSM, Titans Memory, and Causal Entropy Gating. |
| **5** | [`SHOWCASE_PROFESSOR/benchmark_all_10m_architectures.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/SHOWCASE_PROFESSOR/benchmark_all_10m_architectures.py) | **Benchmark Harness:** The script enforcing strict single-digit parameter parity ($\pm 0.0137\%$) and lockstep training on TinyStories. |

---

## 2. Executive Empirical Data Summary (Copy-Pasteable for Claude)

You can copy and paste this table directly into Claude:

### A. Multi-Seed Statistical Significance Benchmark (Seeds 42, 43, 44 on TinyStories English)
| Architecture | Paradigm | Exact Parameters | Delta vs. Target | Val Loss (Mean $\pm$ Std) | Byte-PPL (Mean $\pm$ Std) | Speed (tok/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Transformer** | Softmax Attention | 9,984,556 | -0.0089% | $3.1286 \pm 0.0299$ | $22.85 \pm 0.69$ | 32,973.9 |
| **Linear Attention** | Kernel Associative | 9,986,816 | +0.0137% | $3.1896 \pm 0.0013$ | $24.28 \pm 0.03$ | 2,653.3 |
| **Pure Mamba SSM** | State Space Model | 9,985,664 | +0.0022% | $1.2462 \pm 0.0017$ | $3.47 \pm 0.01$ | 2,419.2 |
| **IVERI-HYPERION** | **SSM + Titans Hybrid** | **9,985,449** | **0.0000%** | **$1.2102 \pm 0.0168$** | **$3.35 \pm 0.06$** | **2,400.8** |

- **Welch's Two-Sample t-test (vs. Dense Transformer):** $t = -96.79$, $p = 7.1 \times 10^{-8}$ (Decisive statistical significance, $> 5\sigma$).
- **Paired t-test (vs. Pure Mamba SSM):** IVERI won on 100% of seeds (Seed 42: $1.2291$ vs $1.2453$; Seed 43: $1.2044$ vs $1.2482$; Seed 44: $1.1970$ vs $1.2451$), mean reduction $-0.0360$ loss ($t = -3.61, p = 0.069$).

### B. Hardware & Systems Profiling (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Decoding Memory Invariance:**  
  - Step 1: 224.3701 MB  
  - Step 100: 224.3701 MB  
  - Step 500: 224.3701 MB  
  - **Memory Delta: 0.000000 MB (Strictly flatline $\mathcal{O}(1)$ scaling)**
  - At 32K tokens context, Transformer KV cache requires $188.0\text{ MB}$ vs. IVERI's fixed $0.30\text{ MB}$ ($627\times$ memory reduction).
- **Triton Fused GPU Kernel:**  
  - PyTorch interpreted loop: $4,289.46\text{ ms}$  
  - Triton fused kernel: $173.05\text{ ms}$  
  - **Speedup: 24.8×** (hidden state kept resident in 32-bit registers across 512 steps).

### C. Honest Scientific Boundary (Synthetic Passkey Needle-in-a-Haystack)
- Tested across context lengths $S \in \{64, 128, 256, 512, 1024\}$.
- **Zero-shot retrieval accuracy:** $0.00\%$ with mean byte loss $\sim 10.6$.
- **Finding:** The associative memory matrix boosts natural language perplexity by $11.2\%$, but verbatim multi-thousand byte key-value retention requires targeted synthetic multi-task pretraining.

---

## 3. Complete Directory Map of All Data in the Workspace

### 📊 Empirical Results & Raw JSON Benchmarks
All raw JSON test outputs and audit certificates live in:
`c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\reports\final_validation\`
- `multi_seed_statistical_validation.json` — The raw seed data (42, 43, 44), loss, PPL, and t-tests.
- `four_way_10m_benchmark_results.json` — Parameter counts and throughput numbers.
- `FINAL_CONCLUDED_BENCHMARK.json` — Concluded summary comparison.
- `passkey_retrieval_capacity_results.json` — Needle-in-a-haystack capacity audit across sequence lengths 64 to 1024.
- `EXP_ABLATION_STUDY.json` — Component ablation numbers.
- `FINAL_RESEARCH_VALIDATION_CERTIFICATE.md` — Complete research certification audit.

### 📝 Paper Manuscripts & Publications
All manuscripts and guides live in:
`c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\PAPERS\`
- `STATE_AND_MEMORY_COMPREHENSIVE_GUIDE.md` — The intuitive master markdown guide with diagrams and code.
- `STATE_AND_MEMORY_ARE_ALL_YOU_NEED.tex` — Camera-ready LaTeX manuscript (NeurIPS / IEEE style with TikZ and booktabs).
- `references.bib` — 35 academic citations in BibTeX format.
- `STATE_AND_MEMORY_ARE_ALL_YOU_NEED_15_PAGES.docx` — Complete 15-page Word document.
- `STATE_AND_MEMORY_ARE_ALL_YOU_NEED_15_PAGES.md` — Full 15-page academic markdown manuscript.

### 💻 Model Architecture Code
The production model code lives in:
`c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\model\`
- `model/v1/iveri_v1.py` — Complete IVERI-HYPERION architecture (S4D SSM, Fast Linear Titans Memory, Causal 1D Conv Entropy Gate).
- `model/kernels/triton_scan.py` — The original high-performance Triton fused recurrence kernel (`triton_fused_ssm_recurrence`).
- `model/v1/triton_fused_ssm.py` — Mirror alias of the Triton kernel in `model/v1/`.

### 🔬 Validation & Benchmark Harnesses
All reproducible benchmark runners live in:
`c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\SHOWCASE_PROFESSOR\`
- `benchmark_multi_seed_statistical.py` — The multi-seed runner (Seeds 42, 43, 44).
- `benchmark_all_10m_architectures.py` — The 4-way parameter-parity benchmark runner.
- `benchmark_passkey_needle.py` — The needle-in-a-haystack passkey test.
- `generate_15_page_docx.py` — The generator for the 15-page Word manuscript.

---

## 4. Suggested Prompt to Give to Claude

When you start a chat with Claude, you can paste this prompt:

> *"Hi Claude, I am working on the research paper 'State and Memory Are All You Need: Sub-Quadratic Sequence Modeling with Entropy-Gated Associative Neural Networks'. We have completed multi-seed empirical validation at the 10M parameter scale comparing IVERI-HYPERION (an SSM + Titans associative memory hybrid) against Dense Transformers, Linear Attention, and Pure Mamba SSMs under strict parameter parity (±0.0137%).*
>
> *I have attached the master guide (`STATE_AND_MEMORY_COMPREHENSIVE_GUIDE.md`) and the empirical JSON benchmark (`multi_seed_statistical_validation.json`). Please review the architecture, the empirical data, and help me analyze/refine the findings."*
