# Telemetry Report — Phase 2.4
**IVERI CORE v1.0 | Phase 2.4**
**Date:** 2026-06-30

---

## 1. Telemetry Channels

### Experiment Metadata (logged once at run start)
| Key | Description |
|---|---|
| `meta/iveri_version` | IVERI package version |
| `meta/architecture_version` | Architecture freeze version |
| `meta/run_name` | Human-readable run label |
| `meta/run_id` | UUID hex run identifier |
| `meta/timestamp` | ISO-8601 start time |
| `meta/random_seed` | Global random seed |
| `meta/dataset_version` | Dataset version string |
| `meta/dataset_hash` | Dataset content hash |
| `meta/git_commit` | Short git commit hash |
| `meta/git_branch` | Git branch name |

### System Information (logged with metadata)
| Key | Description |
|---|---|
| `system/python_version` | Python interpreter version |
| `system/pytorch_version` | PyTorch version |
| `system/cuda_version` | CUDA toolkit version |
| `system/os` | OS platform string |
| `system/cpu` | CPU identifier |
| `system/gpu_name` | GPU device name |
| `system/gpu_vram_mb` | GPU VRAM total (MB) |
| `system/ram_total_mb` | Host RAM total (MB) |

---

## 2. Architecture Telemetry (per training step)

Architecture-specific metrics are collected from `outputs["telemetry"]` returned by the model forward pass. Any scalar value in that dict is logged under the `telemetry/` prefix.

| Expected Key | Source Module |
|---|---|
| `telemetry/average_entropy` | BLT |
| `telemetry/average_patch_length` | BLT |
| `telemetry/compression_ratio` | BLT |
| `telemetry/boundary_count` | BLT |
| `telemetry/patch_count` | BLT |
| `telemetry/expert_utilization` | MoE |
| `telemetry/router_entropy` | MoE |
| `telemetry/load_balance_loss` | MoE |
| `telemetry/average_recursion_depth` | MoR |
| `telemetry/early_exit_rate` | MoR |
| `telemetry/hidden_state_norm` | Mamba2 |
| `telemetry/state_update_norm` | Mamba2 |
| `telemetry/memory_read_gate` | Titans |
| `telemetry/memory_write_gate` | Titans |
| `telemetry/attention_backend` | Flash Attention |

---

## 3. Gradient & Parameter Telemetry

| Key Pattern | Description |
|---|---|
| `grad/total_norm` | Global gradient L2 norm |
| `grad/clipping_count` | Params exceeding clip threshold |
| `grad_norm/<layer>` | Per-layer gradient norm |
| `grad_max/<layer>` | Per-layer gradient max absolute value |
| `grad_min/<layer>` | Per-layer gradient min absolute value |
| `param/total_count` | Total parameter count |
| `param/trainable_count` | Trainable parameter count |
| `param/frozen_count` | Frozen parameter count |
| `param/total_norm` | Global parameter weight norm |
| `param_norm/<layer>` | Per-layer weight norm |

---

## 4. Memory Telemetry

| Key | Description |
|---|---|
| `memory/gpu_allocated_mb` | GPU memory currently allocated (MB) |
| `memory/gpu_reserved_mb` | GPU memory reserved by PyTorch (MB) |
| `memory/gpu_peak_mb` | Peak GPU memory allocated (MB) |
| `memory/cpu_ram_mb` | Process RSS memory (MB, via psutil) |

---

## 5. Training Performance Telemetry

| Key | Description |
|---|---|
| `train/loss` | Cross-entropy loss |
| `train/aux_loss` | Load-balancing auxiliary loss |
| `train/learning_rate` | Current LR from scheduler |
| `timing/dataloader_seconds` | Time to fetch batch |
| `timing/forward_seconds` | Forward pass duration |
| `timing/backward_seconds` | Backward pass duration |
| `timing/optimizer_seconds` | Optimiser step duration |
| `timing/scheduler_seconds` | Scheduler step duration |
| `timing/step_total_seconds` | Full step duration |
| `performance/samples_per_sec` | Throughput in samples/sec |
| `performance/tokens_per_sec` | Throughput in tokens/sec |
| `val/loss` | Validation cross-entropy loss |
| `val/aux_loss` | Validation auxiliary loss |

---

## 6. Verification

All telemetry channels verified by:
- `test_architecture_telemetry_from_dict` — architecture dict forwarded correctly
- `test_gradient_and_param_stats_logged` — per-layer norms and global norm computed
- `test_memory_telemetry_logged` — GPU keys always present
- `test_experiment_metadata_logs_keys` — system and meta keys present
- `test_trainer_logs_train_and_val_metrics` — train/loss, val/loss, and timing keys logged
