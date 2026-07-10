# Phase 7 Task Tracker (v4.0 Frozen Checklist)

- [x] **Phase 7.1: Repository Health**
  - [x] Run full test suite: python -m pytest tests/
  - [x] Verify imports of IVERIModel, BaselineTransformer, TinyMamba, pretrain_runner, sft_runner, IVERIInferenceEngine
  - [x] Run forward pass smoke test (nano config, B=2, S=64, check logits, no NaNs, finite loss, telemetry)
  - [x] Verify checkpoint save/load round-trip
  - [x] Verify pyproject.toml version is 0.1.0

- [x] **Phase 7.2: Configuration Validation**
  - [x] Add factory presets to configs/base_config.py: get_nano_config, get_small_config, get_medium_config, get_large_config, get_xlarge_config, get_max_config
  - [x] Create presets YAML files in configs/presets/ (nano, small, medium, large, xlarge, max)
  - [x] Write parameter count and VRAM/FLOP breakdown script (scripts/count_params.py)
  - [x] Run scripts/count_params.py to generate configs/parameter_breakdown.json
  - [x] Verify that nano config is within +/-20% of 10M parameters (9.3M, OK)
  - [x] Run Phase 7.x Architecture Regression

- [x] **Phase 7.3: Data Pipeline**
  - [x] Align data/pipeline/dataloader.py's SFTByteDataset to return (x, y, loss_mask) using LossMaskBuilder
  - [x] Audit 16 modules in data/pipeline/ for edge cases
  - [x] Implement dataset preparation script data/prepare_tinystories.py
  - [x] Save, reload, and verify dataset shards (compare SHA-256, sample counts, statistics)
  - [x] Generate deep dataset statistics and verify reproducibility
  - [x] Test dataloader integration & corruption injection rejection
  - [x] Run Phase 7.x Architecture Regression


- [x] **Phase 7.4: Architecture Validation**
  - [x] Create architecture monitor script scripts/architecture_health.py
  - [x] Hook all modules, verify active state, fwd counts, grad flow
  - [x] Verify MoE expert utilization (all experts, no >60% load)
  - [x] Verify MoR depth routing (non-uniform, >=3 depths)
  - [x] Verify Titans online weight updates & entropy routing variations
  - [x] Run Phase 7.x Architecture Regression


- [x] **Phase 7.5: Debugging**
  - [x] Implement component disable flags in train.py & model/iveri_core.py
  - [x] Implement layer-wise instability diagnostics tracking to logs/debug_diagnostics.json
  - [x] Add divergence layer tracker for outlier identification before NaNs
  - [x] Run Phase 7.x Architecture Regression


- [/] **Phase 7.6: Real Training**
  - [x] Implement live dashboard: scripts/training_dashboard.py (showing steps, finish time, step speed stats, etc.)
  - [x] Implement failure analyzer and recovery: training/failure_analyzer.py (with severity levels and recoverability flags)
  - [x] Log component execution latency (BLT, Titans, MoE, MoR, decoder, memory, fwd%, bwd%, opt%)
  - [/] Run IVERI 10M Pilot training run (1,000 steps)
  - [ ] Generate and save product demo outputs at 1,000 steps
  - [ ] Store historical metrics curves (expert usage, Titans updates, depths, entropy, grads)

  - [ ] Execute baseline comparison runs: Transformer vs Mamba vs IVERI (1000 steps) after pilot success
  - [ ] Run IVERI 10M Short training run (10,000 steps)
  - [ ] Generate and save product demo outputs at 5,000 and 10,000 steps
  - [ ] Run Phase 7.x Architecture Regression

- [x] **Phase 7.7: Optimization**
  - [x] Verify mixed precision (fp16) & gradient checkpointing VRAM reduction
  - [x] Verify gradient accumulation correctness
  - [x] Profile detailed memory usage: scripts/profile_memory.py (weights, activations, optimizer, temp)
  - [x] Profile step timing: scripts/profile_training_step.py
  - [x] Run Phase 7.x Architecture Regression


- [ ] **Phase 7.8: Inference**
  - [ ] Verify inference engine on CPU/GPU (greedy, temp, top-k, nucleus)
  - [ ] Expand CLI generate, enchmark, info in inference/cli.py
  - [ ] Profile inference performance: scripts/profile_inference.py (TTFT, tokens/sec, latency)
  - [ ] Run Phase 7.x Architecture Regression

- [ ] **Phase 7.9: Product Validation**
  - [ ] E2E user scenario test on Windows (clone, install, prepare, train, resume, infer, benchmark)
  - [ ] Verify checkpoint fidelity & exact continuation after resume (100, 500, 1000, 5000, 10000 steps)
  - [ ] Verify documentation: README, QUICKSTART, TRAINING_GUIDE
  - [ ] Run Phase 7.x Architecture Regression

- [ ] **Phase 7.10: Scaling**
  - [ ] Test multi-scale instantiation on CPU (nano, small, medium, large)
  - [ ] Write scaling readiness report docs/SCALING_GUIDE.md (expected throughput, GPU, FLOPs, VRAM)
  - [ ] Run Phase 7.x Architecture Regression

- [ ] **Phase 7.11: Master Audit**
  - [ ] Implement scripts/generate_phase7_reports.py
  - [ ] Compile 11 markdown reports under eports/phase_7/
  - [ ] Calculate final Architecture Health Score & Overall Product Score out of 100
