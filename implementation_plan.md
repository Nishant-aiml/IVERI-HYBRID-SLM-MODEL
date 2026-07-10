# Phase 7: IVERI Product Development -- 10M Foundation Model
## Implementation Plan v5.0 (Frozen Specification)

## Governing Question

Every task in Phase 7 must answer one question:
> Does this make IVERI a working, stable, demonstrably usable 10M hybrid SLM?

If the answer is no, it moves to a later phase.

## Removed from Phase 7 (belongs to Phase 8-11)
- REST API, Python SDK, Docker, ONNX export, HuggingFace wrapper, SafeTensors, Deployment
- Version tag (e.g. v1.0.0-phase7-freeze), version bump (remains at 0.1.0)
- Cryptographic manifests, engineering freeze of the codebase

## Added to Phase 7 (v5.0 additions)
- **SFT Loss Masking Fix**: Align data/pipeline/dataloader.py's SFTByteDataset to return (x, y, loss_mask) using LossMaskBuilder to match 	raining/sft_dataset.py exactly.
- **Architecture Regression Suite (Phase 7.x)**: Run regression checks (imports, fwd, bwd, grad flow, checkpoint save/load, inference, training step, architecture health, memory profile, performance profile) at the end of each sub-phase.
- **Training Stability gates**: Stage training at 10 -> 50 -> 100 -> 500 -> 1,000 -> 10,000 steps.
- **Architecture Health History**: Save logs/architecture_health_history.json tracking step metrics (MoE load, Titans updates, MoR depth, entropy, gradients, VRAM, latency).
- **Scaling Readiness Report**: Projections from 10M up to 500M (10M, 35M, 70M, 150M, 300M, 500M) detailing parameters, FLOPs, VRAM, recommended GPU, and expected throughput.
- **Product Freeze Specification**: Generate PRODUCT_FREEZE.md as the final specification document.
- **Detailed Config Validation**: Generate parameter_breakdown.json with FLOPs, activation memory, KV cache size, optimizer memory, estimated training VRAM, inference VRAM, and per-component parameter distribution.
- **Training Component Latency**: Log component execution latency (BLT, Titans, MoE, MoR, decoder, memory, fwd%, bwd%, opt%).
- **Data Pipeline Shard Validation**: Save, reload, and verify dataset shards (compare SHA-256, sample counts, statistics).
- **Expanded Training Dashboard**: Show estimated finish time, remaining epochs/steps, average step time, best/worst steps.
- **Failure Analyzer Classification**: Classify failures (Critical, High, Medium, Low) and flag whether they are recoverable (yes/no).
- **Product Validation Platform**: Focus verification specifically on Windows environment compatibility since it is the primary environment.
- **Overall Product Score**: Calculate a final score out of 100 based on individual subsystem scores.
- **Postponed Baselines**: Baseline comparisons (Transformer, Mamba) will run only after IVERI's pilot training successfully converges.

## Phase Roadmap Context
Phase 7  = Build and validate a real 10M product.  <- THIS PLAN
Phase 8  = Scale the architecture (35M-500M).
Phase 9  = Scientific benchmarking and research.
Phase 10 = Publications and patents.
Phase 11 = Production deployment and commercialization.

---

## Phase 7.1 -- Repository Health

Objective: Verify repo is in clean, fully functioning state before training.
Gate question: Does every component import, initialize, and forward-pass cleanly?

### 1.1 Full Test Suite: python -m pytest tests/ -v --tb=short
Required: >= 683 pass, 0 failures, skips only for GPU-specific on CPU.

### 1.2 Import Health Check
Verify imports of IVERIModel, BaselineTransformer, TinyMamba, pretrain_runner, sft_runner, IVERIInferenceEngine.

### 1.3 Forward Pass Smoke Test
IVERIModel nano config, raw_bytes (2,64). Verify:
- logits shape (2,64,256), no NaN, finite aux_loss, telemetry keys present
- Backward produces gradients on all parameters.

### 1.4 Checkpoint Round-Trip
Save step 0, load, compare logits. Max difference < 1e-6.

### 1.5 Version Sync
pyproject.toml remains at 0.1.0 to match current development phase.

Exit Gate: All PASS.

---

## Phase 7.x -- Architecture Regression (Executed after every sub-phase)

Objective: Prevent silent regressions during codebase edits.
Checklist:
1. Verify imports of all critical classes.
2. Run a single forward pass and backward pass.
3. Verify gradient flow on all active weights.
4. Verify checkpoint save/load round-trip.
5. Verify inference engine produces tokens.
6. Verify one training step runs successfully.
7. Output health status to console.

---

## Phase 7.2 -- Configuration Validation

Gate question: Is actual parameter count approximately 10M?

### 2.1 Named Config Factory Functions [MODIFY configs/base_config.py]
get_nano_config()   -> 10M,   hidden_dim=256, num_layers=6,  num_heads=4
get_small_config()  -> ~35M,  hidden_dim=384, num_layers=8,  num_heads=8
get_medium_config() -> ~70M,  hidden_dim=512, num_layers=10, num_heads=8
get_large_config()  -> ~150M, hidden_dim=768, num_layers=12, num_heads=12

### 2.2 Config YAML Presets [NEW]
configs/presets/nano_10m.yaml
configs/presets/small_35m.yaml
configs/presets/medium_70m.yaml
configs/presets/large_150m.yaml

### 2.3 Parameter Count & Breakdown Script [NEW: scripts/count_params.py]
Instantiates IVERIModel at each preset. Reports parameters per component.
Generates configs/parameter_breakdown.json containing:
- Parameter count per module (BLT encoder, decoder, Titans, Mamba, MoE, MoR)
- FLOPs estimate
- Activation memory estimate
- KV cache size estimate
- Optimizer memory estimate
- Estimated training VRAM & inference VRAM
FAIL if total parameter count is outside +/-20% of 10M target.

### 2.4 Config Cross-Validation
Verify hidden_dim % num_heads == 0, d_inner % num_heads == 0 (Mamba2),
batch_size x grad_accum <= 4096, warmup_steps < max_steps,
BLT patch_size_min <= patch_size_max.

### 2.5 Hardware-Specific Config
Verify nano config fits RTX 3050 (10GB) at seq_len=256, batch_size=8.

Exit Gate: Param count within +/-20% of 10M. configs/parameter_breakdown.json generated. All validations pass. Run Phase 7.x regression.

---

## Phase 7.3 -- Data Pipeline

Gate question: Is TinyStories correctly prepared and safe to train on? Is the SFT masking bug resolved?

### 3.1 Squash SFT Loss Masking Bug [MODIFY data/pipeline/dataloader.py]
Update SFTByteDataset in data/pipeline/dataloader.py to import LossMaskBuilder and ConversationFormatter, and return (x, y, loss_mask) where prompt tokens are 0 and response tokens are 1, matching training/sft_dataset.py.

### 3.2 Pipeline Module Audit
Verify 16 modules in data/pipeline/ handle edge cases (empty, whitespace-only, non-UTF8, >100K char lines, duplicates).

### 3.3 TinyStories Preparation [NEW: data/prepare_tinystories.py]
python data/prepare_tinystories.py --output-dir data/processed/tinystories --seed 42
Steps: download -> license check -> UTF-8 validation -> quality filter -> dedup -> language detect -> split (98/1/1) -> binary shards -> version hash.

### 3.4 Shard Validation & Deep Statistics
Save binary shards, reload them, and verify SHA-256, sample counts, and sequence counts are identical.
Report per split: samples, total bytes, mean/std length, byte/entropy distribution histograms, duplicate rate, corruption rate.

### 3.5 Dataloader Integration Test & Reproducibility
Run prepare twice. SHA-256 of shards must match exactly.
100 batches: shape (batch_size, seq_len) long, values [0,255], no NaN/Inf.
1% corruption injection test: pipeline must reject all.

Exit Gate: Prepared dataset SHA-256 matches, data pipeline rejects all corruption. Run Phase 7.x regression.

---

## Phase 7.4 -- Architecture Validation

Gate question: Do all components (BLT, Titans, MoE, MoR, Mamba2) dynamically run, update, and contribute gradients?

### 4.1 Component Activation Monitor [NEW: scripts/architecture_health.py]
Forward hooks on all key components. Forward + backward pass. Dashboard showing active state, call counts, shape, forward time, VRAM, and gradient flow. FAIL if any component shows FwdCalled=0 or GradFlow=NO.

### 4.2 Expert Utilization & MoR Depth Verification
100 steps: Verify all 4 experts receive tokens (no expert >60% load). Verify non-uniform depths (>= 3 distinct depths).

### 4.3 Titans Weight Updates & Entropy Routing Verification
(weights_before - weights_after).abs().max() > 1e-8.
Verify low-entropy vs high-entropy inputs yield different patch counts (patches_random > patches_repeated).

### 4.4 Gradient Health Per Component
Verify all component gradient norms > 1e-8. FAIL if any component has all-zero gradients.

Exit Gate: All components active, updating, routing, and training. Run Phase 7.x regression.

---

## Phase 7.5 -- Architecture Debugging

Gate question: If training or convergence fails, how do we systematically diagnose it?

### 5.1 Component Disable Debugging [MODIFY train.py & model/iveri_core.py]
Add run-time flags: --disable-blt, --disable-moe, --disable-titans, --disable-mor, --disable-mamba.
This allows training the model with specific components bypassed to isolate instability (e.g. if model stabilizes with Titans disabled, the issue is isolated to Titans).

### 5.2 Instability Diagnostics
Track layer-wise statistics:
- Gradient and activation histograms per layer.
- Residual, attention, state, and memory norms per step.
- Per-layer loss sensitivity (perturb weights and measure loss change).
Log these automatically to logs/debug_diagnostics.json if a NaN or loss divergence is detected.

### 5.3 Divergence Layer Tracker
A monitor that identifies which specific layer first starts producing abnormally large values (outliers) before a NaN crash or gradient explosion occurs.

Exit Gate: Component disable flags verified functional. Diagnostics tracking verified on small synthetic run. Run Phase 7.x regression.

---

## Phase 7.6 -- Real Training

Gate question: Does the model converge? How does it compare to standard baselines?

### 6.1 Training Stability Gates
Run training staged incrementally:
1. **10 steps** -> verify loss decreases, gradients finite, no NaNs, optimizer healthy
2. **50 steps** -> verify continuation, check metrics curves, save checkpoint
3. **100 steps** -> verify checkpoint save/load round-trip
4. **500 steps** -> verify longer-term stability
5. **1,000 steps** -> complete full pilot run
Each gate must pass all safety checks before proceeding to the next.

### 6.2 Training Run (Pilot: 1,000 steps)
python train.py --config configs/presets/nano_10m.yaml --dataset data/processed/tinystories
Monitor: loss, perplexity, learning rate, gradient norms, NaN/overflow counts, MoE expert loads, MoR depths.
FAIL if loss does not decrease over 1000 steps (initial: ~5.5, step 1000: <4.0).

### 6.3 Component Latency & Timing Profile
Log component execution latency at every step:
- BLT, Titans, MoE, MoR, and decoder latency
- Forward %, backward %, optimizer %
- Memory latency (IO/Dataloader wait time)

### 6.4 Live Dashboard [NEW: scripts/training_dashboard.py]
Console dashboard updating every 5 seconds with live training and component statistics. Shows:
- Step, loss, PPL, LR, GPU%, VRAM, tokens/sec, grad norm
- Estimated finish time, remaining epochs/steps, average step time, best/worst steps
- MoE expert loads, MoR depth distribution, active components

### 6.5 Failure Analysis & Recovery [NEW: training/failure_analyzer.py]
Auto-detects NaN, gradient explosion, divergence, expert collapse, dead MoR, OOM.
Classifies failures (Critical, High, Medium, Low) and flags whether they are recoverable (yes/no).
Logs JSON: {step, failure_type, severity, recoverable, root_cause, fix_applied, recommendation}.

### 6.6 Product Demo Outputs
At steps 1000, 5000, and 10000, generate and save outputs for:
- Story: \"Once upon a time\"
- Code: \"def reverse_string(s):\"
- Q&A: \"What is the capital of India?\"
Save generated text files to eports/phase_7/demo_outputs/ to visually audit learning.

### 6.7 Architecture Health History
Save logs/architecture_health_history.json for every run, tracking step metrics (MoE load, Titans updates, MoR depth, entropy, gradients, VRAM, latency).

### 6.8 Short Run (10,000 steps, after pilot passes)
Target: perplexity < 50 on val set.

### 6.9 Baseline Comparison (Transformer vs Mamba vs IVERI 10M)
After the pilot run successfully converges, train BaselineTransformer and TinyMamba on identical dataset, seed, optimizer, and steps.
Compare: final loss, step speed, peak VRAM, tokens/sec, and convergence rate. Log results to eports/phase_7/05_training_baseline_comparison.md.

Exit Gate: Training loss decreases. All components active. Baseline comparison logged. Run Phase 7.x regression.

---

## Phase 7.7 -- Optimization Validation

Gate question: Are optimizations active and reducing VRAM and training time?

### 7.1 Mixed Precision (AMP) & Checkpointing
Verify Mamba2 matmuls in fp16, layer norms in fp32.
Compare VRAM: gc=True vs gc=False. Document VRAM reduction (target: 30-60%).

### 7.2 Gradient Accumulation Correctness
Verify 4 steps of batch=8 accumulated equals 1 step of batch=32 within 1e-5.

### 7.3 Deep Memory Profiler [NEW: scripts/profile_memory.py]
Breakdown VRAM (in MB) during a training step:
- Model Weights (BLT, Titans, MoE, MoR, decoder, backbone)
- Activation Tensors
- Gradient Checkpoint cache
- Optimizer states (AdamW m and v)
- Temporary/Workspace tensors

### 7.4 Timing Profile [NEW: scripts/profile_training_step.py]
50-step CUDA event timing: Dataloader, BLT Encoder, Titans, Mamba2, Attention, MoE FFN, MoR Router, BLT Decoder, Backward, Optimizer.
Identify top-3 bottlenecks.

Exit Gate: AMP confirmed active, memory breakdown and step timings documented. Run Phase 7.x regression.

---

## Phase 7.8 -- Inference Validation

Gate question: Can IVERI generate coherent text and support reproducible generation?

### 8.1 Inference Engine & CLI [MODIFY: inference/engine.py & cli.py]
Verify greedy, temperature, top-k, nucleus sampling on CPU and GPU.
Verify CLI commands work: generate, benchmark, info.

### 8.2 Inference Profiling [NEW: scripts/profile_inference.py]
Record: TTFT (time-to-first-token), tokens/sec, peak VRAM, and latency.

### 8.3 Generation Coherence & Reproducibility
FAIL if generation outputs invalid UTF-8, repeats prompt indefinitely, or is completely uniform.
Verify manual_seed(42) produces identical output on sequential greedy generation calls.

Exit Gate: Sampling runs on CPU/GPU, CLI functional, generation is reproducible. Run Phase 7.x regression.

---

## Phase 7.9 -- Product Validation

Gate question: Can a new user run the complete workflow end-to-end on Windows?

### 9.1 End-to-End User Scenario Test
Run from clean env on Windows: clone -> install -> prepare dataset -> train (1000 steps) -> resume training (to 2000 steps) -> generate -> benchmark.
FAIL if any step requires source code edits or manual path fixes.

### 9.2 Checkpoint Fidelity & Continuation
Verify training resumed from step 1000 to 2000 produces exact same loss, optimizer moments, Titans fast-weights, and random state as uninterrupted training. Verify resume after 100, 500, 1000, 5000, 10000 steps.

### 9.3 Documentation
README.md: update with verified commands.
docs/QUICKSTART.md: zero-assumption copy-pasteable guide.
docs/TRAINING_GUIDE.md: configuration and troubleshooting.

Exit Gate: E2E scenario runs without manual fixes. Continuation fidelity verified. README verified. Run Phase 7.x regression.

---

## Phase 7.10 -- Scaling Readiness

Gate question: Does the codebase support scaling to 500M without redesign?

### 10.1 Multi-Scale Instantiation
Verify Nano (10M), Small (35M), Medium (70M), Large (150M), XLarge (300M), Max (500M) configurations instantiate and execute forward/backward on CPU.
FAIL on hardcoded dimensions or shape mismatch.

### 10.2 Configuration Scaling Report [NEW: docs/SCALING_GUIDE.md]
Projections from 10M to 500M detailing:
- Parameter increase
- FLOPs increase
- VRAM increase
- Expected throughput
- Recommended GPU

Exit Gate: All 6 scales run forward/backward. Scaling guide written. Run Phase 7.x regression.

---

## Phase 7.11 -- Master Product Audit

### 11.1 Master Acceptance Checklist (32 checks)
- Architecture (6): active, gradients, Titans update, MoE load, MoR depth, entropy routing
- Data Pipeline (4): SFT masking fix, UTF-8 clean, no corruption, stats generated
- Training & Debugging (6): loss decreases, no NaN, checkpoint saved, metrics logged, disable component flags, layer-wise diagnostics
- Optimization (3): AMP active, VRAM reduced, memory profiled
- Inference & Comparison (5): checkpoint loads, greedy, nucleus, determinism, CLI works, baseline comparison
- Product & Scaling (8): clean install, resume correct, continuation fidelity, README, multi-scale, scaling guide, shard validation, demo outputs

### 11.2 Overall Product Score
Calculate a unified score out of 100 based on individual subsystem scores:
- Repository Health (7.1)
- Configuration Validation (7.2)
- Data Pipeline (7.3)
- Architecture Validation (7.4)
- Debugging (7.5)
- Training (7.6)
- Optimization (7.7)
- Inference (7.8)
- Product Validation (7.9)
- Scaling Readiness (7.10)

### 11.3 Final Product Freeze Specification [NEW: PRODUCT_FREEZE.md]
Creates the final spec document including architecture, configurations, datasets, checkpoints, benchmarks, health score, limitations, known issues, and future roadmap.

### 11.4 Reports Directory
Generate 11 reports in reports/phase_7/:
  01_repository_health.md
  02_configuration_validation.md
  03_data_pipeline_validation.md
  04_architecture_runtime_validation.md
  05_training_report.md
  06_architecture_debugging.md
  07_optimization_validation.md
  08_inference_report.md
  09_product_validation.md
  10_scaling_readiness.md
  11_phase_completion_report.md (includes Health Score, overall Product Score, and links to PRODUCT_FREEZE.md)

Exit Gate: >= 28/32 PASS. No Critical/High blockers.

---

## Explicitly OUT of Phase 7 (Deferred)
REST API, Python SDK, Docker, ONNX, HuggingFace wrapper, SafeTensors, publications/patents.
