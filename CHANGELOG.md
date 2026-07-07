# Changelog

All notable changes to the **IVERI CORE** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-07-07

### Added — Phase 6.3.3 Engineering Stabilization
- **`inference/` package:** `InferenceEngine`, `ByteTokenizer`, `Sampler`, checkpoint loader, streaming, CLI (`python -m inference.cli`), and `benchmark_inference()`.
- **Stage 3B ingestion:** `data/pipeline/proprietary_ingest.py` and `scripts/ingest_stage3b.py` for proprietary JSON validation, PII cleaning, 90/5/5 splits, and `data/processed/stage3b/` manifests.
- **Deployment docs:** `docs/deployment/INFERENCE.md` and `data/proprietary/FORMAT.md`.
- **Measured CPU benchmark:** `scripts/run_inference_benchmark.py` → `reports/phase_6_3_3/inference_benchmark.json`.
- **Phase 6.3.3 reports:** `reports/phase_6_3_3/` (14 deliverables + `validation_state.json`).

### Fixed
- Full test suite green: **611 passed**, 4 skipped (byte vocab migration, Windows W&B fallbacks, DataLoader `num_workers=0` in tests).
- `training/loss_mask.py` list padding; `generation_inspector` safe byte decode; logger `PermissionError` fallbacks.

### Changed
- `tests/conftest.py`: `base_config` sets `num_workers=0` and CPU fallback when CUDA unavailable.
- `ARCHITECTURE_VERSION` remains `0.2.0-byte-vocab` (frozen architecture unchanged).

## [1.5.0] - 2026-07-06

### Added — Phase 6.3.2 Scientific Integrity Restoration
- **Objective 1 — BLT causality:** Causal Conv1d, encoder/decoder attention masks; `research/causality_probe.py` runtime proof.
- **Objective 2 — Titans integration:** `Backbone.forward` uses `TitansMemory.forward_with_injection()` for online read/update/write.
- **Objective 3 — Entropy MoE:** `SparseMoERouter` entropy logit bias; `research/entropy_routing_audit.py`.
- **Objective 4 — Physical ablations:** `ModelConfig` flags (`use_titans`, `use_blt`, `use_mor`, `use_moe`, `use_entropy_routing`) with runtime path disabling.
- **Objective 5 — Publication fail-closed:** `PublicationManager` blocks failures/non-MEASURED provenance; no mock figure placeholders.
- **Objective 6 — Replay integrity:** `replay_campaign.py` pre-flight registry, claim-chain, and figure gates with non-zero exit on failure.
- **Objective 7 — Byte vocabulary:** Collision-free specials BOS=256, PAD=257, EOS=258; `BYTE_VOCAB_SIZE=259`; `ARCHITECTURE_VERSION=0.2.0-byte-vocab`.
- **Objective 8 — Documentation sync:** README, CHANGELOG, master doc, and phase index aligned with implementation.
- **Audit harnesses:** `research/*_audit.py` modules and `reports/scientific_integrity_audit/` measured reports (protocol `Phase-6.3.2-OBJ*`).
- **Migration notes:** `docs/migrations/PHASE_6_3_2_OBJ*.md` (Objectives 1–8).

### Changed
- Model embeddings and logits expanded to 259-token vocabulary.
- `matplotlib>=3.9.0` added to `requirements-dev.txt` for publication figure tests.

## [1.4.0] - 2026-07-04

### Added
- **Phase 4.0 (Large-Scale Validation Campaign - Stage 6A):** Implemented a production campaign manager, declarative configurations, cost estimators, hardware adapters, checkpoint lifecycles, and health monitors.
- **Campaign Config (`research/campaign_config.py`):** Configures declarative experiment profiles (`verification`, `pilot`, `full`, `paper`) adjusting steps, models, seeds, and benchmarks.
- **CLI Campaign Entry (`run_campaign.py`, `research/campaign_runner.py`):** CLI entry point supporting resume strategies (`AUTO`, `FROM_LAST`, `FROM_BEST`, `FROM_GOLDEN`, `FROM_CHECKPOINT_ID`) and `--dry-run` pre-flight checks. Generates master `Campaign_Report.md`.
- **Cost Estimator (`research/cost_estimator.py`):** Predicts GPU runtime, energy footprint, disk requirements, and cloud dollars.
- **Execution Backend (`research/execution_backend.py`):** Automatically configures workers, precision, and gradient accumulation for RTX3050, Kaggle, Colab, Vast.ai, and Lambda Labs.
- **Dataset Validator (`research/campaign_dataset_validator.py`):** Audits processed splits and licenses and outputs `dataset_manifest.json`.
- **Agnostic Grader (`research/external_eval.py`):** Abstract `ExternalModel` adapter evaluating prompts against commercial APIs, skipping execution if keys are missing from environment.
- **Checkpoint lifecycle (`research/golden.py`):** Formalizes checkpoint lifecycle states (`Candidate` -> `Validated` -> `Golden` -> `Paper` -> `Released` -> `Archived`) and top-performing metrics screening.
- **Campaign Lock (`research/campaign_lock.py`):** Freezes codebase, configurations, and dataset hashes to prevent contamination on paper runs.
- **Campaign Health Monitor (`research/campaign_health_monitor.py`):** Detects NaNs, stalled training steps, dead GPUs, and cost spikes to auto-pause campaigns.
- **Experiment Manifest (`research/experiment_manifest.py`):** Compiles complete environment, OS, hardware, and PyTorch/CUDA specs to `manifest.json`.
- **Publication Manager (`research/publication_manager.py`):** Packages vector figures, tables, and manifestations to `reproducibility_package.zip`.
- **Test suite (`tests/test_production_campaign.py`):** 35 unit and integration tests verifying all campaign runner components.

## [1.3.0] - 2026-07-04

### Added
- **Phase 3.6 (Orchestration Campaign, Validation & Paper Traceability - Stage 5B):** Created relational experiment tracking databases and schedulers to manage large research campaigns.
- **Relational registry (`research/experiment_registry.py`):** SQLite (`experiments.db`) tracking runs, metrics, hardware, datasets, checkpoints, failures, notes, and assets.
- **Topological Scheduler (`research/experiment_scheduler.py`):** Dependency sorting, priority queueing, and recovery.
- **Run Comparator (`research/compare_runs.py`):** Quantifies deltas in loss and perplexity with statistics (t-test, Wilcoxon, Cohen's d, bootstrap).
- **Regression guards (`research/regression_detector.py`):** Multi-level severities (`INFO`, `WARNING`, `CRITICAL`, `FATAL`) for quality drops.
- **Golden Checkpoint Manager (`research/golden.py`):** Handles references and model parameter rollbacks.
- **Failure Replay Engine (`research/failure_replay.py`):** Complete RNG states serialization (PyTorch, NumPy, Python, CUDA) on failures.
- **Artifact Graph (`research/artifacts_graph.py`):** Verifies file paths and config/dataset checksum hash integrity.
- **Dashboard & Scorecard (`research/dashboard.py`, `research/scorecard.py`):** Console visualizations and Paper Submission Checklist.
- **Traceability Manifest (`research/paper_artifact_generator.py`):** Compiles `paper_manifest.json` linking all LaTeX tables and Matplotlib plots to commits and configurations.
- **Test suite (`tests/test_experimental_campaign.py`):** 27 tests validating locks release, RNG states, and regression severities.
- **Orchestration Reports:** Generated 9 markdown reports in `reports/phase_3_6/` outlining timeline, databases, regressions, and timeline histories.

## [1.2.0] - 2026-07-04

### Added
- **Phase 3.5 (Research Validation, Benchmarking & Ablation Suite - Stage 5):** Created unified benchmarking and validation infrastructure to prove or reject architectural claims.
- **Baseline Models (`research/baselines.py`):** Added matched BaselineMamba2, BaselineHybrid, and BaselineTransformer configurations matching IVERI parameter/FLOP budgets.
- **Checkpoint Manager (`research/checkpoint_manager.py`):** Verified baseline parameters parity, checksum hashes, git commits, and checkpoint registry tracking.
- **Ablation Suite (`research/ablation.py`):** Dynamic config overrides for Titans, MoR, MoE, and BLT subsystems.
- **Experiment Runners (`research/experiment_runner.py`, `research/multi_seed.py`):** Run single-run training and 5-seed metrics aggregation (variance, standard deviation, 95% confidence intervals).
- **Profilers (`research/flops.py`, `research/profiler.py`, `research/energy_profiler.py`):** Multi-dimensional VRAM, RAM, latency (TTFT, decode), power draw (Watts, Joules/token), cloud cost estimation, and analytical FLOP counts.
- **Confidence Calibration (`research/calibration.py`):** Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and NLL with reliability diagrams.
- **Statistics & Claims Validator (`research/statistics.py`, `research/claim_validator.py`, `research/hypothesis.py`):** Standardizes paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes, bootstrap confidence intervals, Reproducibility Score, Research Integrity Score, and H1/H2/H3 hypothesis evaluation.
- **Paper helper generators (`research/paper_figures.py`, `research/paper_tables.py`, `research/paper_summary.py`, `research/artifacts.py`):** Matplotlib vector figures, LaTeX tables, draft Discussion outlines, and zipped reproducibility manifests.
- **Research Test Suite (`tests/test_research.py`):** 40 unit and integration tests verifying all benchmarking and analytics modules.

## [1.1.0] - 2026-07-04

### Added
- **Phase 3.4 (Preference Optimization & Alignment - Stage 4):** Implemented preference-tuning training loops, custom loss functions, qualitative inspection, and safety diagnostic reporting.
- **Preference Config (`configs/preference_config.py`):** Added configuration fields for DPO, SimPO, IPO, and Conservative DPO algorithms, beta scaling, and device offloading.
- **Preference Dataset Loader (`training/preference_dataset.py`):** Supports license compliance verification and Stage 4 data curation checks.
- **Preference Loss (`training/preference_loss.py`):** Log-probability sequence computations for DPO, SimPO, IPO, and Conservative DPO with NaN guards.
- **Reference Model Manager (`training/reference_model.py`):** Offloads reference model parameters to CPU/GPU with strict startup parameter checks and version compatibility.
- **Alignment Prompt Suite (`evaluation/alignment_prompt_suite.py`):** Defines 50 prompts across 11 science and engineering domains.
- **Alignment Inspector (`evaluation/alignment_inspector.py`):** Scans generations for over-refusals, mode collapse, copy loops, and reward hacking.
- **Preference Benchmark (`evaluation/preference_benchmark.py`):** Computes win rates and chosen/rejected reward margin distributions.
- **Alignment Evaluator (`evaluation/alignment_evaluator.py`):** Orchestrates validation loss, win rates, qualitative logging, and retention of coding/instruction capabilities.
- **Preference Test Suite (`tests/test_preference_training.py`):** 25 unit and integration tests covering all alignment pipeline layers.

## [1.0.0] - 2026-07-02
- **Phase 3.3 (Coding Specialization - Stage 3A):** Implemented the complete coding specialization training and multi-language evaluation framework.
- **Coding Config (`configs/coding_config.py`):** Added a new `CodingConfig` class for pipeline parameters, step budgets, retention checks, and security warnings.
- **Coding Dataset Loader (`training/coding_dataset.py`):** Handles both `pretrain` (full sequence loss) and `sft` (masked loss) formats with stage check version validation.
- **Code Formatter (`training/code_formatter.py`):** Injects programming language header blocks and delegates to the base formatter.
- **Coding Curriculum (`training/coding_curriculum.py`):** Coordinates 3-stage training: Code Fluency (0–33%) → Code Instruction (33–66%) → Competitive Programming (66–100%).
- **Coding Runner (`training/coding_runner.py`):** Orchestrates SFT tuning starting from the Stage 2 checkpoint.
- **Coding Evaluator (`evaluation/coding_evaluator.py`):** Computes validation loss and coding metrics like cyclomatic complexity and comment/docstring ratios.
- **Instruction Retention Evaluator (`evaluation/instruction_retention.py`):** Verifies instruction-following capability is retained on the Stage 2 prompt suite, rejecting regressed checkpoints.
- **Code Executor (`evaluation/code_execution.py`):** Compiles and runs generated Python code in a safe, sandboxed subprocess with timeouts to verify accuracy.
- **Code Quality Analyzer (`evaluation/code_quality_analyzer.py`):** Evaluates cyclomatic complexity (radon or AST branching fallback) and docstring/comment coverage.
- **Security Scanner (`evaluation/security_scanner.py`):** Warns about potential security vulnerabilities (e.g. `eval()`, hardcoded secrets, shell executions).
- **Contamination Checker (`evaluation/contamination_checker.py`):** Matches n-gram fingerprints to ensure benchmark prompts are clean from training datasets.
- **HumanEval & MBPP Benchmarks (`evaluation/humaneval_benchmark.py`, `evaluation/mbpp_benchmark.py`):** Computes pass@1 scores.
- **Coding Test Suite (`tests/test_coding_specialization.py`):** Added 25 unit and integration tests covering all coding sub-components. All 377 tests pass.

## [0.9.0] - 2026-07-02

### Added
- **Phase 3.2 (Supervised Fine-Tuning - SFT - Stage 2):** Implemented the complete SFT training, evaluation, and quality inspection pipeline.
- **Conversation Formatter (`training/conversation_formatter.py`):** Translates Alpaca single-turn and Multi-turn Chat formats into deterministic byte sequences with standard role delimiters (`### System:`, `### Instruction:`, `### User:`, `### Response:`).
- **Loss Mask Builder (`training/loss_mask.py`):** Generates boolean loss masks to only compute next-byte cross-entropy loss on assistant responses (`train_on_prompt=False`), masking out prompts and padding.
- **SFT Byte Dataset (`training/sft_dataset.py`):** Extends the base dataset to support selective training on assistant responses, custom formatters, padding, truncation, and autoregressive byte shifts.
- **Instruction Dataset Ingest Loader (`training/instruction_dataset.py`):** Implements strict validation checks (research-license checks, manifest SHA-256 integrity, stage = 2 provenance check, and SFTValidator structure checks).
- **SFT Runner Orchestrator (`training/sft_runner.py`):** Manages SFT fine-tuning loops from pretrained checkpoints with masked loss, dynamic evaluations, qualitative generation checks, and checkpoint selection.
- **SFT Evaluator (`evaluation/sft_evaluator.py`):** Calculates validation loss, perplexity, and byte-prediction accuracies (top-1 and top-5) exclusively on response bytes, plus qualitative prompt suite generation.
- **Response Inspector (`evaluation/response_inspector.py`):** Assesses generation quality, detecting issues like token collapse, repetition, loop structures, and UTF-8 encoding corruption.
- **Prompt Suite (`evaluation/prompt_suite.py`):** Fixed set of 35 deterministic evaluation prompts across 14 CS and General QA categories to evaluate model convergence.
- **SFT Checkpoint Selector (`training/model_selection.py`):** Promotes SFT quality scores to rank checkpoints jointly by loss and generation quality.
- **SFT Test Suite (`tests/test_instruction_tuning.py`):** Added 13 SFT-specific tests, bringing the codebase to 352 passing tests (100% Green).

## [0.8.0] - 2026-07-02

### Added
- **Phase 3.1 (Foundation Pretraining - Stage 1):** Implemented, verified, and benchmarked the complete foundation pretraining pipeline for both IVERI CORE and Baseline Transformer control models.
- **Pretraining Runner Orchestrator (`training/pretrain_runner.py`):** Implemented the pretraining loop with dataset loader validation, curriculum scheduling, model forward/backward passes, evaluation, generation inspection, checkpoint selection, and live training CSV logging.
- **Curriculum Scheduler (`training/curriculum.py`):** Manages dynamic sequence length scaling and learning rate decay schedules (warmup + cosine decay).
- **Loss Monitor (`training/loss_monitor.py`):** Tracks convergence loss, perplexity, Bits-Per-Byte (BPB), and gradient health statistics (norms, max/min, zero ratio, variance).
- **Checkpoint Selector (`training/model_selection.py`):** Periodically saves, ranks, and registers best-performing checkpoints with robust metadata.
- **Generation Inspector (`evaluation/generation_inspector.py`):** Monitors text generation quality (average entropy, invalid UTF-8 counts, repetition collapse).
- **Baseline Control (`baselines/baseline_transformer.py`):** Integrated a standard Byte-level Transformer baseline matching the structural layout.
- **Phase 3.1 Verification Reports:** Created 10 comprehensive verification reports detailing license checks, numerical stability, curriculum scheduler, checkpoint selector, gradient health, generation inspection, and comparison metrics.
- **Pretraining Test Suite (`tests/test_pretraining.py`):** Added 11 new integration tests covering all pretraining sub-components. All 339 tests pass.

## [0.6.0] - 2026-06-30

### Added
- **Phase 2.5 (Evaluation Pipeline & Benchmark Infrastructure):** Implemented a production-quality, modular, and strictly read-only evaluation framework entirely decoupled from the training engine.
- **Language Modeling Evaluator:** Computes Cross Entropy, Negative Log Likelihood, and Perplexity metrics with full NaN/Inf guards and numerical overflow protection.
- **Generative Decoding Evaluator:** Implements `greedy`, `temperature`, `top_k`, and `top_p` sampling decoding strategies, measuring generation latency, bytes/sec throughput, and sequence length.
- **Inference & Memory Benchmark:** Measures CPU/GPU utilization, RAM/VRAM, parameter/activation memory, and estimates FLOPs per inference step.
- **Architecture Telemetry distributions:** Processes subsystem telemetry, returning metrics and detailed histograms (expert load histograms, routing entropy, MoR recursion depth profiles, Titans reads/writes, Mamba2 state variance, and layer runtimes).
- **Centralized Report Generator:** Outputs evaluation summaries in JSON, CSV, and Markdown report structures.
- **Checkpoint Comparator:** Performs structural checks (params size, arch version, shape configuration hash) to prevent invalid comparisons, outputting a warning if `NOT DIRECTLY COMPARABLE`, and computing deltas otherwise.
- **Configuration Compatibility:** Extends `base_config.py` with `EvaluationConfig` sub-class, enforcing backward-compatible deserialization where missing keys use defaults and unknown keys trigger warnings.
- **Evaluation Test Suite:** Developed 14 comprehensive unit and integration tests in `tests/test_evaluation.py`, all passing successfully.

## [0.5.0] - 2026-06-30

### Added
- **Phase 2.4 (Experiment Logging & Telemetry Infrastructure):** Implemented production-quality `ExperimentLogger` (`training/logger.py`) with four backends — Weights & Biases (online/offline), TensorBoard, CSV, and JSONL — operating under a fail-safe cascade (W&B → TB → CSV → JSONL). Training is never interrupted by a logging failure.
- **Experiment Metadata Logging:** Automatic one-time logging of IVERI/architecture version, git commit/branch, Python/PyTorch/CUDA versions, GPU/CPU/RAM/VRAM specs, random seed, dataset version/hash, and full `IVERIConfig` snapshot at run start.
- **Architecture Telemetry:** Per-step logging of model forward-pass telemetry (BLT entropy/patch stats, MoE expert utilization, MoR recursion depth, Titans gate activations, Mamba2 norms) from `outputs["telemetry"]`.
- **Gradient & Parameter Telemetry:** Per-layer gradient norms (L2, max, min), global gradient norm, gradient clipping count, per-layer weight norms, and total/trainable/frozen parameter counts.
- **Memory Telemetry:** GPU allocated/reserved/peak memory (MB) and CPU process RSS via `psutil`.
- **NaN/Inf Sanitisation:** All metric values sanitised before dispatch — `NaN` and `±Inf` replaced with `0.0`.
- **Extended `LoggingConfig`:** Added `run_name`, `run_id`, `resume`, `tags`, `notes`, `log_frequency`, `checkpoint_frequency`, `system_monitor_interval`, and 14 other experiment-management fields with full `__post_init__` validation.
- **Logging Test Suite:** Created 22-test suite (`tests/test_logging.py`) covering disabled mode, local CSV/JSONL, NaN/Inf sanitisation, metadata, hyperparameters, architecture/gradient/memory telemetry, fallback recovery, large dicts, 10 000-step simulation, trainer integration, and latency benchmarks. All 22 tests pass.

## [0.4.0] - 2026-06-30


### Added
- **Phase 2.2 (Trainer & Optimization Pipeline):** Implemented a production-quality Trainer (`training/trainer.py`), parameter-decay grouped AdamW optimizer (`training/optimizer.py`), automatic mixed precision wrapper (`training/mixed_precision.py`), and metadata-validated checkpoint loading and saving (`training/checkpointing.py`).
- **Trainer Testing:** Created a 6-test suite (`tests/test_training.py`) verifying precision states, decay parameter routing, model state saving and loading, and training loop iteration. All 221 tests pass.

## [0.3.0] - 2026-06-30

### Added
- **Phase 2.1 (Raw Byte Dataset & DataLoader):** Implemented raw UTF-8 byte-level dataset preprocessing (`data/preprocessing.py`) and dataset utilities (`data/dataset_utils.py`). Built map-style `ByteDataset` and iterable-style `StreamingByteDataset` in `data/dataloader.py` with multi-worker partition support and validation.
- **Dataloader Testing:** Created a 23-test suite (`tests/test_dataset.py`) verifying boundary conditions, multilingual UTF-8 handling, duplicate checks, determinism, and performance benchmarks. All 215 tests pass.

## [0.1.0] - 2026-06-29

### Added
- **Project Structure:** Fully scavenged workspace folders per architecture guidelines.
- **Config System:** Strongly-typed dataclass-based base configuration (`base_config.py`) matching the 10M parameter nano defaults.
- **Core Infrastructure:** Component registry, factory pattern, interfaces, constant definitions, and custom exception classes.
- **Logging Pipeline:** Structured console format and rotating metric logger outputting steps/loss/memory performance.
- **Validation Utilities:** Multi-dimensional shape matching (with wildcards), NaN/Inf guards, gradient norm limits, and GPU/CPU resource counters.
- **Quality Checks:** Integrated local QA gate scripts for Ruff (linting), Black (formatting), Mypy (typing), and Pytest.
- **Documentation:** Mapped awesome agentic skills, dependency order flowcharts, contributing rules, and licensing documents.

## [0.2.0] - 2026-06-30

### Added
- **Phase 1.1 (Core Math Layers):** Implemented production-quality `RMSNorm`, `RotaryEmbedding` (RoPE), and `SwiGLUFFN` layers with safety controls for mixed precision and dynamic caching.
- **Phase 1.2 (Sparse MoE):** Implemented Sparse GShard-style routing with auxiliary load balancing, token capacity management, and expert dispatch pipelines.
- **Phase 1.3 (Mamba2 SSD Block):** Implemented Structured State Space Duality (SSD) discretization, sequential recurrence scan equivalence checks, and Mamba2 block integration.
- **Phase 1.4 (Flash Attention Wrapper):** Implemented unified, backend-independent attention dispatcher supporting FlashAttention-2 and PyTorch SDPA with dynamic causal masking and in-place KV caching.
- **Phase 1.5 (Mixture of Recursions — MoR):** Implemented `RecursionDepthRouter` (Option C production mode: `D_p = 1 + floor(E_p × (max_depth - 1))`), `RecursionEngine` with active-mask selective bypassing, and `SelectiveKVCache`. All 100% gradient flow verified.
- **Phase 1.6 (Byte Latent Transformer — BLT):** Implemented `ByteEntropyModel` (output: `(B, S, 1)` float32), `DynamicPatcher` (output: `(B, S)` bool boundary map), `BLTByteEncoder` (within-patch MHA + mean pooling → `(B, P, D)`), and `BLTByteDecoder` (cross-attention → `(B, S, 256)`). Multilingual UTF-8 validated; 10 tests pass.
- **Phase 1.7 (Titans Neural Memory):** Implemented `TitansMemory` with frozen online update rule `Sₜ = η·Sₜ₋₁ − θₜ∇W·ℓ(Wₜ₋₁)`, `Wₜ = (1-α)·Wₜ₋₁ + Sₜ`, entropy-gated injection, and comprehensive telemetry. Inherits `BaseMemory`.
- **Phase 1.8 (Backbone Assembly):** Assembled unified `BackboneBlock` (RMSNorm → Mamba2 → Flash Attention → Sparse MoE FFN) wrapped in `RecursionEngine` with global Titans memory gating. 100% test success.
- **Phase 1.9 (Full Model Integration):** Integrated all components into `IVERIModel` with frozen pipeline: ByteEntropyModel → DynamicPatcher → BLTByteEncoder → patch entropy aggregation → Backbone × L → BLTByteDecoder. Checkpoint save/load API with architecture version enforcement. All 177 tests pass.

## [0.2.1] - 2026-06-30

### Frozen
- **Phase 1.9.1 (Architecture Freeze):** IVERI CORE v0.1.0-optionC architecture is officially frozen. All Phase 1 components locked. Option C entropy signal (ByteEntropyModel) drives exactly 4 consumers: DynamicPatcher, RecursionDepthRouter, SparseMoERouter, TitansMemory. No further architectural changes permitted without regression test suite confirmation.
- **Documentation Audit:** Performed complete documentation audit across README, CHANGELOG, RESEARCH_LOG, all architecture docs, and completion reports. Updated tensor_interfaces.md with missing interface contracts; corrected `TitansNeuralMemory` → `TitansMemory` class name in module_dependencies.md; updated README phase status table; added Phase 1.9.1 log entry to RESEARCH_LOG.

## [0.1.0] - 2026-06-30
### Phase 1.9.1 — Architecture Freeze & Comprehensive Verification
- Complete engineering audit of Phase 1 architecture
- Static analysis: import boundaries, inheritance, registry verified
- Mathematical verification of all components confirmed
- Stress testing: multilingual, boundary conditions, 100-run determinism
- Regression suite: 177+ tests pass
- Quality gate: Lint + Format + TypeCheck + Tests = PASSED
- tensor_interfaces.md updated with all missing interface contracts
- module_dependencies.md corrected (TitansNeuralMemory → TitansMemory)
- Phase 1 (Core Architecture) officially declared complete and frozen

