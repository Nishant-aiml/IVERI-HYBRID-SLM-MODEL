# Quality Assurance Report — Phase 0

**Date:** 2026-06-30 23:32:48
**Overall Status:** FAILED

## Check Summary

| Check | Status | Duration |
|---|---|---|
| Lint | FAILED | 0.30s |
| Format | FAILED | 3.32s |
| TypeCheck | FAILED | 3.69s |
| Tests | PASSED | 239.12s |

## Detailed Logs

### Lint
```
Running Linting Checks...
[91mFAILED: Linting checks found issues:[0m
I001 [*] Import block is un-sorted or un-formatted
  --> tests\test_logging.py:6:1
   |
 4 |   """Unit and integration tests for IVERI CORE logging and telemetry (Phase 2.4)."""
 5 |
 6 | / from __future__ import annotations
 7 | |
 8 | | import csv
 9 | | import json
10 | | import pathlib
11 | | import shutil
12 | | import tempfile
13 | | import time
14 | | from typing import Any
15 | |
16 | | import pytest
17 | | import torch
18 | | import torch.nn as nn
19 | | from torch.utils.data import DataLoader, TensorDataset
20 | |
21 | | from configs.base_config import get_base_config
22 | | from training.logger import ExperimentLogger, _flatten_dict
23 | | from training.trainer import Trainer
   | |____________________________________^
   |
help: Organize imports

F401 [*] `shutil` imported but unused
  --> tests\test_logging.py:11:8
   |
 9 | import json
10 | import pathlib
11 | import shutil
   |        ^^^^^^
12 | import tempfile
13 | import time
   |
help: Remove unused import: `shutil`

I001 [*] Import block is un-sorted or un-formatted
  --> training\__init__.py:6:1
   |
 4 |   """Training infrastructure â€” trainer, optimizer, checkpointing, mixed precision, scheduler, logger."""
 5 |
 6 | / from __future__ import annotations
 7 | |
 8 | | from training.trainer import Trainer
 9 | | from training.optimizer import get_optimizer
10 | | from training.checkpointing import save_checkpoint, load_checkpoint
11 | | from training.mixed_precision import PrecisionHandler
12 | | from training.scheduler import IVERIScheduler, SchedulerFactory
13 | | from training.logger import ExperimentLogger
   | |____________________________________________^
14 |
15 |   __all__ = [
   |
help: Organize imports

F401 [*] `numpy` imported but unused
  --> training\logger.py:28:17
   |
26 | from typing import Any
27 |
28 | import numpy as np
   |                 ^^
29 | import torch
30 | import torch.nn as nn
   |
help: Remove unused import: `numpy`

SIM105 Use `contextlib.suppress(Exception)` instead of `try`-`except`-`pass`
   --> training\logger.py:401:13
    |
399 |               return
400 |           if self.use_wandb:
401 | /             try:
402 | |                 _wandb.finish()  # type: ignore[union-attr]
403 | |             except Exception:
404 | |                 pass
    | |____________________^
405 |           if self.use_tb and self.tb_writer is not None:
406 |               try:
    |
help: Replace `try`-`except`-`pass` with `with contextlib.suppress(Exception): ...`

SIM105 Use `contextlib.suppress(Exception)` instead of `try`-`except`-`pass`
   --> training\logger.py:406:13
    |
404 |                   pass
405 |           if self.use_tb and self.tb_writer is not None:
406 | /             try:
407 | |                 self.tb_writer.close()
408 | |             except Exception:
409 | |                 pass
    | |____________________^
410 |
411 |       # â”€â”€ Private file writers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    |
help: Replace `try`-`except`-`pass` with `with contextlib.suppress(Exception): ...`

I001 [*] Import block is un-sorted or un-formatted
  --> training\trainer.py:10:1
   |
 8 |   """
 9 |
10 | / from __future__ import annotations
11 | |
12 | | import pathlib
13 | | import time
14 | | from typing import Any
15 | | import torch
16 | | import torch.nn as nn
17 | | from torch.utils.data import DataLoader
18 | |
19 | | from configs.base_config import IVERIConfig
20 | | from training.checkpointing import load_checkpoint, save_checkpoint
21 | | from training.mixed_precision import PrecisionHandler
22 | | from training.optimizer import get_optimizer
23 | | from training.logger import ExperimentLogger
24 | | from utils.validation import get_gpu_memory_usage
   | |_________________________________________________^
   |
help: Organize imports

Found 7 errors.
[*] 5 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

### Format
```
Running Formatting Checks...
[91mFAILED: Formatting issues found. Run 'black .' to fix:[0m
None
Exception in thread Thread-1 (_readerthread):
Traceback (most recent call last):
  File "C:\Python314\Lib\threading.py", line 1082, in _bootstrap_inner
    self._context.run(self.run)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "C:\Python314\Lib\threading.py", line 1024, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python314\Lib\subprocess.py", line 1614, in _readerthread
    buffer.append(fh.read())
                  ~~~~~~~^^
  File "C:\Python314\Lib\encodings\cp1252.py", line 23, in decode
    return codecs.charmap_decode(input,self.errors,decoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 27252: character maps to <undefined>
```

### TypeCheck
```
Running Type Checking...
[91mFAILED: Type checking errors found:[0m
training\logger.py:50: error: Cannot assign to a type  [misc]
training\logger.py:50: note: Error code "misc" not covered by "type: ignore[assignment]" comment
training\logger.py:206: error: Argument "mode" to "init" has incompatible type "str"; expected "Literal['online', 'offline', 'disabled', 'shared'] | None"  [arg-type]
training\logger.py:208: error: Argument "resume" to "init" has incompatible type "str | None"; expected "Literal['allow', 'never', 'must', 'auto'] | bool | None"  [arg-type]
tests\test_logging.py:83: error: Returning Any from function declared to return "dict[str, Any] | Tensor"  [no-any-return]
evaluation\arch_eval.py:227: error: Incompatible types in assignment (expression has type "Tensor | Module | Any", variable has type "int")  [assignment]
Found 5 errors in 3 files (checked 101 source files)
```

### Tests
```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0 -- C:\Python314\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\datta.000\Desktop\iveri core nexus\iveri-core
configfile: pyproject.toml
plugins: anyio-4.13.0, cov-7.1.0, timeout-2.4.0
collecting ... collected 265 items

tests/test_attention.py::test_attention_forward_shape[cpu] PASSED        [  0%]
tests/test_attention.py::test_attention_gradflow[cpu] PASSED             [  0%]
tests/test_attention.py::test_attention_causal_masking[cpu] PASSED       [  1%]
tests/test_attention.py::test_attention_kv_caching[cpu] PASSED           [  1%]
tests/test_attention.py::test_attention_reset_parameters[cpu] PASSED     [  1%]
tests/test_attention.py::test_attention_mixed_precision[dtype0-cpu] PASSED [  2%]
tests/test_attention.py::test_attention_mixed_precision[dtype1-cpu] SKIPPED [  2%]
tests/test_attention.py::test_attention_mixed_precision[dtype2-cpu] SKIPPED [  3%]
tests/test_backbone.py::test_backbone_block_creation_and_shapes PASSED   [  3%]
tests/test_backbone.py::test_backbone_full_orchestration_and_gradients PASSED [  3%]
tests/test_backbone.py::test_backbone_telemetry_compilation PASSED       [  4%]
tests/test_backbone.py::test_backbone_residual_norm_and_order PASSED     [  4%]
tests/test_backbone.py::test_integration_stress_boundary_conditions PASSED [  4%]
tests/test_backbone.py::test_moe_expert_imbalance_and_titans_saturation PASSED [  5%]
tests/test_backbone.py::test_multilingual_utf8_pipeline_equivalence PASSED [  5%]
tests/test_backbone.py::test_seed_determinism PASSED                     [  6%]
tests/test_blt.py::test_entropy_model_output_and_configurability[cnn_mlp-cpu] PASSED [  6%]
tests/test_blt.py::test_entropy_model_output_and_configurability[lstm-cpu] PASSED [  6%]
tests/test_blt.py::test_entropy_model_output_and_configurability[linear-cpu] PASSED [  7%]
tests/test_blt.py::test_patcher_determinism_and_reconstruction[cpu] PASSED [  7%]
tests/test_blt.py::test_multilingual_utf8_validation[cpu] PASSED         [  7%]
tests/test_blt.py::test_encoder_decoder_roundtrip_gradient_flow[cpu] PASSED [  8%]
tests/test_blt.py::test_blt_validation_checks PASSED                     [  8%]
tests/test_blt.py::test_blt_numerical_stability[cpu] PASSED              [  9%]
tests/test_blt.py::test_blt_telemetry_collection[cpu] PASSED             [  9%]
tests/test_blt.py::test_patch_reconstruction_determinism[cpu] PASSED     [  9%]
tests/test_config.py::test_default_config_creation PASSED                [ 10%]
tests/test_config.py::test_default_values_match_nano PASSED              [ 10%]
tests/test_config.py::test_nested_config_access PASSED                   [ 10%]
tests/test_config.py::test_to_dict_returns_dict PASSED                   [ 11%]
tests/test_config.py::test_from_dict_roundtrip PASSED                    [ 11%]
tests/test_config.py::test_save_load_roundtrip PASSED                    [ 12%]
tests/test_config.py::test_get_base_config_no_overrides PASSED           [ 12%]
tests/test_config.py::test_get_base_config_with_overrides PASSED         [ 12%]
tests/test_config.py::test_invalid_hidden_dim_zero PASSED                [ 13%]
tests/test_config.py::test_invalid_hidden_dim_not_divisible_by_heads PASSED [ 13%]
tests/test_config.py::test_invalid_active_experts_exceeds_total PASSED   [ 13%]
tests/test_config.py::test_invalid_learning_rate_zero PASSED             [ 14%]
tests/test_config.py::test_invalid_min_lr_exceeds_lr PASSED              [ 14%]
tests/test_config.py::test_invalid_mixed_precision PASSED                [ 15%]
tests/test_config.py::test_invalid_log_level PASSED                      [ 15%]
tests/test_config.py::test_invalid_patch_size PASSED                     [ 15%]
tests/test_config.py::test_effective_batch_size_limit PASSED             [ 16%]
tests/test_config.py::test_warmup_exceeds_max_steps PASSED               [ 16%]
tests/test_dataset.py::test_validate_utf8_valid PASSED                   [ 16%]
tests/test_dataset.py::test_validate_utf8_invalid PASSED                 [ 17%]
tests/test_dataset.py::test_clean_invalid_bytes PASSED                   [ 17%]
tests/test_dataset.py::test_normalize_whitespaces PASSED                 [ 18%]
tests/test_dataset.py::test_text_to_bytes PASSED                         [ 18%]
tests/test_dataset.py::test_chunk_sequence PASSED                        [ 18%]
tests/test_dataset.py::test_pad_sequence PASSED                          [ 19%]
tests/test_dataset.py::test_dataset_statistics PASSED                    [ 19%]
tests/test_dataset.py::test_detect_duplicates PASSED                     [ 20%]
tests/test_dataset.py::test_find_text_files_and_loaders PASSED           [ 20%]
tests/test_dataset.py::test_byte_dataset_shapes_and_types PASSED         [ 20%]
tests/test_dataset.py::test_streaming_byte_dataset PASSED                [ 21%]
tests/test_dataset.py::test_dataloader_seed_determinism PASSED           [ 21%]
tests/test_dataset.py::test_multilingual_dataset_validation[Hello English world!-English] PASSED [ 21%]
tests/test_dataset.py::test_multilingual_dataset_validation[\u0928\u092e\u0938\u094d\u0924\u0947 \u0939\u093f\u0902\u0926\u0940 \u0926\u0941\u0928\u093f\u092f\u093e!-Hindi] PASSED [ 22%]
tests/test_dataset.py::test_multilingual_dataset_validation[\u4f60\u597d\u4e2d\u6587\u4e16\u754c\uff01-Chinese] PASSED [ 22%]
tests/test_dataset.py::test_multilingual_dataset_validation[\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 \u0627\u0644\u0639\u0631\u0628\u064a!-Arabic] PASSED [ 23%]
tests/test_dataset.py::test_multilingual_dataset_validation[\u3053\u3093\u306b\u3061\u306f\u65e5\u672c\uff01-Japanese] PASSED [ 23%]
tests/test_dataset.py::test_multilingual_dataset_validation[\uc548\ub155\ud558\uc138\uc694 \ud55c\uad6d!-Korean] PASSED [ 23%]
tests/test_dataset.py::test_multilingual_dataset_validation[\U0001f30d\U0001f680\U0001f525-Emoji] PASSED [ 24%]
tests/test_dataset.py::test_dataset_empty_docs PASSED                    [ 24%]
tests/test_dataset.py::test_single_sample_batch PASSED                   [ 24%]
tests/test_dataset.py::test_dataset_performance PASSED                   [ 25%]
tests/test_environment.py::test_python_version PASSED                    [ 25%]
tests/test_environment.py::test_torch_import PASSED                      [ 26%]
tests/test_environment.py::test_numpy_import PASSED                      [ 26%]
tests/test_environment.py::test_einops_import PASSED                     [ 26%]
tests/test_environment.py::test_tqdm_import PASSED                       [ 27%]
tests/test_environment.py::test_core_import PASSED                       [ 27%]
tests/test_environment.py::test_config_import PASSED                     [ 27%]
tests/test_environment.py::test_utils_logging_import PASSED              [ 28%]
tests/test_environment.py::test_utils_validation_import PASSED           [ 28%]
tests/test_environment.py::test_cuda_availability_reported PASSED        [ 29%]
tests/test_evaluation.py::test_perplexity_computation PASSED             [ 29%]
tests/test_evaluation.py::test_perplexity_nan_handling PASSED            [ 29%]
tests/test_evaluation.py::test_generation PASSED                         [ 30%]
tests/test_evaluation.py::test_inference_benchmark PASSED                [ 30%]
tests/test_evaluation.py::test_memory_tracking PASSED                    [ 30%]
tests/test_evaluation.py::test_memory_growth PASSED                      [ 31%]
tests/test_evaluation.py::test_architecture_statistics PASSED            [ 31%]
tests/test_evaluation.py::test_report_generation PASSED                  [ 32%]
tests/test_evaluation.py::test_large_report_generation PASSED            [ 32%]
tests/test_evaluation.py::test_checkpoint_version_mismatch PASSED        [ 32%]
tests/test_evaluation.py::test_checkpoint_hash_mismatch PASSED           [ 33%]
tests/test_evaluation.py::test_repeated_evaluation PASSED                [ 33%]
tests/test_evaluation.py::test_device_switch PASSED                      [ 33%]
tests/test_evaluation.py::test_cpu_vs_gpu_consistency PASSED             [ 34%]
tests/test_experts.py::test_experts_forward_shapes[cpu] PASSED           [ 34%]
tests/test_experts.py::test_experts_capacity_dropping[cpu] PASSED        [ 35%]
tests/test_experts.py::test_experts_gradcheck[cpu] PASSED                [ 35%]
tests/test_experts.py::test_experts_reset_parameters[cpu] PASSED         [ 35%]
tests/test_iveri_core.py::test_model_initialization_and_forward PASSED   [ 36%]
tests/test_iveri_core.py::test_end_to_end_gradient_flow PASSED           [ 36%]
tests/test_iveri_core.py::test_tensor_signature_contract_validation PASSED [ 36%]
tests/test_iveri_core.py::test_multilingual_utf8_pipeline PASSED         [ 37%]
tests/test_iveri_core.py::test_boundary_conditions PASSED                [ 37%]
tests/test_iveri_core.py::test_checkpoint_save_and_load PASSED           [ 38%]
tests/test_iveri_core.py::test_checkpoint_incompatibility PASSED         [ 38%]
tests/test_iveri_core.py::test_inference_and_determinism_consistency PASSED [ 38%]
tests/test_iveri_core.py::test_device_transfer_compatibility PASSED      [ 39%]
tests/test_iveri_core.py::test_memory_leak_sanity PASSED                 [ 39%]
tests/test_logging.py::test_logger_disabled_is_noop PASSED               [ 40%]
tests/test_logging.py::test_logger_disabled_mode_is_noop PASSED          [ 40%]
tests/test_logging.py::test_logger_missing_api_key_falls_back PASSED     [ 40%]
tests/test_logging.py::test_csv_backend_writes_correctly PASSED          [ 41%]
tests/test_logging.py::test_jsonl_backend_writes_correctly PASSED        [ 41%]
tests/test_logging.py::test_multiple_log_calls_append PASSED             [ 41%]
tests/test_logging.py::test_nan_metric_replaced_with_zero PASSED         [ 42%]
tests/test_logging.py::test_inf_metric_replaced_with_zero PASSED         [ 42%]
tests/test_logging.py::test_neg_inf_metric_replaced_with_zero PASSED     [ 43%]
tests/test_logging.py::test_experiment_metadata_logs_keys PASSED         [ 43%]
tests/test_logging.py::test_hyperparameter_logging_flattens_config PASSED [ 43%]
tests/test_logging.py::test_architecture_telemetry_from_dict PASSED      [ 44%]
tests/test_logging.py::test_gradient_and_param_stats_logged PASSED       [ 44%]
tests/test_logging.py::test_memory_telemetry_logged PASSED               [ 44%]
tests/test_logging.py::test_corrupted_log_dir_recovery PASSED            [ 45%]
tests/test_logging.py::test_large_telemetry_dict PASSED                  [ 45%]
tests/test_logging.py::test_logging_frequency_config PASSED              [ 46%]
tests/test_logging.py::test_logger_overhead_under_10ms PASSED            [ 46%]
tests/test_logging.py::test_long_run_simulation_10k_steps PASSED         [ 46%]
tests/test_logging.py::test_trainer_logs_train_and_val_metrics PASSED    [ 47%]
tests/test_logging.py::test_flatten_dict_nested PASSED                   [ 47%]
tests/test_logging.py::test_flatten_dict_with_prefix PASSED              [ 47%]
tests/test_mamba2_block.py::test_mamba2_block_forward_shapes[cpu] PASSED [ 48%]
tests/test_mamba2_block.py::test_mamba2_block_gradient_flow[cpu] PASSED  [ 48%]
tests/test_mamba2_block.py::test_mamba2_block_reset_parameters[cpu] PASSED [ 49%]
tests/test_mamba2_block.py::test_mamba2_block_mixed_precision[dtype0-cpu] PASSED [ 49%]
tests/test_mamba2_block.py::test_mamba2_block_mixed_precision[dtype1-cpu] SKIPPED [ 49%]
tests/test_mamba2_block.py::test_mamba2_block_mixed_precision[dtype2-cpu] SKIPPED [ 50%]
tests/test_mamba2_block.py::test_mamba2_block_stress_shapes[1-128-256-cpu] PASSED [ 50%]
tests/test_mamba2_block.py::test_mamba2_block_stress_shapes[4-512-256-cpu] PASSED [ 50%]
tests/test_mamba2_block.py::test_mamba2_block_stress_shapes[2-1024-256-cpu] PASSED [ 51%]
tests/test_mamba2_integration.py::test_mamba2_and_math_layers_coexistence[cpu] PASSED [ 51%]
tests/test_mamba2_integration.py::test_mamba2_backward_with_norms[cpu] PASSED [ 52%]
tests/test_mamba2_math.py::test_discretize_parameters_shapes[euler-cpu] PASSED [ 52%]
tests/test_mamba2_math.py::test_discretize_parameters_shapes[zoh-cpu] PASSED [ 52%]
tests/test_mamba2_math.py::test_discretize_zoh_stability_near_zero[cpu] PASSED [ 53%]
tests/test_mamba2_math.py::test_discretize_gradcheck[cpu] PASSED         [ 53%]
tests/test_mamba2_math.py::test_ssd_matrix_computation[cpu] PASSED       [ 53%]
tests/test_mamba2_math.py::test_ssd_matrix_gradcheck[cpu] PASSED         [ 54%]
tests/test_mamba2_math.py::test_math_property_and_stress[cpu] PASSED     [ 54%]
tests/test_mamba2_scan.py::test_scan_equivalence_to_expanded_ssd[cpu] PASSED [ 55%]
tests/test_mamba2_scan.py::test_scan_gradcheck[cpu] PASSED               [ 55%]
tests/test_mamba2_scan.py::test_scan_long_sequence_stability[128-cpu] PASSED [ 55%]
tests/test_mamba2_scan.py::test_scan_long_sequence_stability[512-cpu] PASSED [ 56%]
tests/test_mamba2_scan.py::test_scan_long_sequence_stability[1024-cpu] PASSED [ 56%]
tests/test_mamba2_scan.py::test_scan_long_sequence_stability[2048-cpu] PASSED [ 56%]
tests/test_mamba2_scan.py::test_scan_long_sequence_stability[4096-cpu] PASSED [ 57%]
tests/test_mamba2_scan.py::test_scan_property_and_determinism[cpu] PASSED [ 57%]
tests/test_math_layers.py::test_rmsnorm_forward_shape_and_dtype[dtype0-cpu] PASSED [ 58%]
tests/test_math_layers.py::test_rmsnorm_forward_shape_and_dtype[dtype1-cpu] PASSED [ 58%]
tests/test_math_layers.py::test_rmsnorm_mathematical_correctness[cpu] PASSED [ 58%]
tests/test_math_layers.py::test_rmsnorm_gradient_flow[cpu] PASSED        [ 59%]
tests/test_math_layers.py::test_rmsnorm_numerical_stability[cpu] PASSED  [ 59%]
tests/test_math_layers.py::test_rope_rotation_correctness[cpu] PASSED    [ 60%]
tests/test_math_layers.py::test_rope_dynamic_extension[cpu] PASSED       [ 60%]
tests/test_math_layers.py::test_swiglu_activation_correctness[cpu] PASSED [ 60%]
tests/test_math_layers.py::test_swiglu_ffn_dimension_rounding[cpu] PASSED [ 61%]
tests/test_math_layers.py::test_interfaces_and_reset_parameters[cpu] PASSED [ 61%]
tests/test_math_layers.py::test_seed_determinism PASSED                  [ 61%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape0-cpu] PASSED  [ 62%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape1-cpu] PASSED  [ 62%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape2-cpu] PASSED  [ 63%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape3-cpu] PASSED  [ 63%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape4-cpu] PASSED  [ 63%]
tests/test_math_layers.py::test_stress_matrix_shapes[shape5-cpu] PASSED  [ 64%]
tests/test_math_layers.py::test_rmsnorm_gradcheck[cpu] PASSED            [ 64%]
tests/test_math_layers.py::test_swiglu_ffn_gradcheck[cpu] PASSED         [ 64%]
tests/test_math_layers.py::test_rmsnorm_scale_invariance_property[cpu] PASSED [ 65%]
tests/test_math_layers.py::test_rope_norm_preservation_property[cpu] PASSED [ 65%]
tests/test_moe_integration.py::test_moe_end_to_end_pipeline[cpu] PASSED  [ 66%]
tests/test_moe_integration.py::test_moe_end_to_end_gradcheck[cpu] PASSED [ 66%]
tests/test_moe_integration.py::test_moe_flop_savings_invariant[cpu] PASSED [ 66%]
tests/test_mor.py::test_router_entropy_mapping[cpu] PASSED               [ 67%]
tests/test_mor.py::test_router_learned_mode[cpu] PASSED                  [ 67%]
tests/test_mor.py::test_recursion_engine_execution[cpu] PASSED           [ 67%]
tests/test_mor.py::test_mor_gradflow[cpu] PASSED                         [ 68%]
tests/test_mor.py::test_selective_kv_cache[cpu] PASSED                   [ 68%]
tests/test_mor.py::test_router_validation_checks PASSED                  [ 69%]
tests/test_mor.py::test_mor_numerical_stability[cpu] PASSED              [ 69%]
tests/test_router.py::test_router_shapes_and_outputs[cpu] PASSED         [ 69%]
tests/test_router.py::test_router_weights_sum_to_one[cpu] PASSED         [ 70%]
tests/test_router.py::test_router_gradcheck[cpu] PASSED                  [ 70%]
tests/test_router.py::test_router_determinism[cpu] PASSED                [ 70%]
tests/test_router.py::test_router_stress_shapes[shape0-cpu] PASSED       [ 71%]
tests/test_router.py::test_router_stress_shapes[shape1-cpu] PASSED       [ 71%]
tests/test_router.py::test_router_stress_shapes[shape2-cpu] PASSED       [ 72%]
tests/test_scheduler.py::test_scheduler_factory_defaults PASSED          [ 72%]
tests/test_scheduler.py::test_scheduler_factory_ratio_override PASSED    [ 72%]
tests/test_scheduler.py::test_invalid_scheduler_config_type PASSED       [ 73%]
tests/test_scheduler.py::test_constant_lr_progression PASSED             [ 73%]
tests/test_scheduler.py::test_linear_warmup_and_cosine_decay PASSED      [ 73%]
tests/test_scheduler.py::test_linear_decay_progression PASSED            [ 74%]
tests/test_scheduler.py::test_polynomial_decay_progression PASSED        [ 74%]
tests/test_scheduler.py::test_step_decay_intervals PASSED                [ 75%]
tests/test_scheduler.py::test_exponential_decay PASSED                   [ 75%]
tests/test_scheduler.py::test_scheduler_state_dict_roundtrip PASSED      [ 75%]
tests/test_scheduler.py::test_scheduler_zero_warmup PASSED               [ 76%]
tests/test_scheduler.py::test_scheduler_long_horizon_simulation PASSED   [ 76%]
tests/test_stress_1_9_1.py::test_empty_sequence PASSED                   [ 76%]
tests/test_stress_1_9_1.py::test_single_token PASSED                     [ 77%]
tests/test_stress_1_9_1.py::test_large_batch PASSED                      [ 77%]
tests/test_stress_1_9_1.py::test_long_sequence PASSED                    [ 78%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[Hello, world! IVERI test.-English] PASSED [ 78%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[\xe0\xa4\xaf\xe0\xa4\xb9 \xe0\xa4\xaa\xe0\xa4\xb0\xe0\xa5\x80\xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7\xe0\xa4\xa3 \xe0\xa4\xb9\xe0\xa5\x88-Hindi] PASSED [ 78%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[\xe6\xb5\x8b\xe8\xaf\x95\xe5\x8f\xa5\xe5\xad\x90-Chinese] PASSED [ 79%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7-Arabic] PASSED [ 79%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[\xf0\x9f\x94\xa5\xf0\x9f\x9a\x80\xf0\x9f\x92\xa1-Emoji] PASSED [ 80%]
tests/test_stress_1_9_1.py::test_multilingual_utf8[Hello \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 \xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x8c\x8d-Mixed] PASSED [ 80%]
tests/test_stress_1_9_1.py::test_all_zeros PASSED                        [ 80%]
tests/test_stress_1_9_1.py::test_all_max_bytes PASSED                    [ 81%]
tests/test_stress_1_9_1.py::test_determinism_100_runs PASSED             [ 81%]
tests/test_stress_1_9_1.py::test_repeated_inference_no_memory_leak PASSED [ 81%]
tests/test_stress_1_9_1.py::test_checkpoint_5_cycles PASSED              [ 82%]
tests/test_stress_1_9_1.py::test_optimizer_compatibility PASSED          [ 82%]
tests/test_stress_1_9_1.py::test_train_eval_mode_switch PASSED           [ 83%]
tests/test_stress_1_9_1.py::test_cpu_device_compatibility PASSED         [ 83%]
tests/test_stress_1_9_1.py::test_bos_eos_pad_bytes PASSED                [ 83%]
tests/test_structure.py::test_required_directories_exist PASSED          [ 84%]
tests/test_structure.py::test_init_files_exist PASSED                    [ 84%]
tests/test_structure.py::test_infrastructure_files_exist PASSED          [ 84%]
tests/test_structure.py::test_core_files_exist PASSED                    [ 85%]
tests/test_structure.py::test_config_files_exist PASSED                  [ 85%]
tests/test_structure.py::test_utils_files_exist PASSED                   [ 86%]
tests/test_structure.py::test_no_model_implementation_files PASSED       [ 86%]
tests/test_structure.py::test_docs_structure PASSED                      [ 86%]
tests/test_structure.py::test_experiments_structure PASSED               [ 87%]
tests/test_structure.py::test_reports_structure PASSED                   [ 87%]
tests/test_titans.py::test_lr_generator_shapes_and_bounds PASSED         [ 87%]
tests/test_titans.py::test_updater_equations PASSED                      [ 88%]
tests/test_titans.py::test_titans_memory_shapes_and_registration PASSED  [ 88%]
tests/test_titans.py::test_differentiability_and_gradient_flow PASSED    [ 89%]
tests/test_titans.py::test_reconstruction_loss_reduction PASSED          [ 89%]
tests/test_titans.py::test_entropy_gated_injection PASSED                [ 89%]
tests/test_titans.py::test_persistence_and_initialization_determinism PASSED [ 90%]
tests/test_titans.py::test_telemetry_collection PASSED                   [ 90%]
tests/test_titans.py::test_extreme_numerical_conditions PASSED           [ 90%]
tests/test_titans.py::test_invalid_shape_validation PASSED               [ 91%]
tests/test_titans.py::test_interface_compliance PASSED                   [ 91%]
tests/test_training.py::test_precision_handler_context PASSED            [ 92%]
tests/test_training.py::test_optimizer_parameter_decay_groups PASSED     [ 92%]
tests/test_training.py::test_checkpoint_save_and_load_roundtrip PASSED   [ 92%]
tests/test_training.py::test_checkpoint_compatibility_assertion PASSED   [ 93%]
tests/test_training.py::test_trainer_training_step PASSED                [ 93%]
tests/test_training.py::test_trainer_eval_and_checkpoint PASSED          [ 93%]
tests/test_validation.py::test_validate_shape_correct PASSED             [ 94%]
tests/test_validation.py::test_validate_shape_wildcard PASSED            [ 94%]
tests/test_validation.py::test_validate_shape_mismatch PASSED            [ 95%]
tests/test_validation.py::test_check_nan_inf_clean PASSED                [ 95%]
tests/test_validation.py::test_check_nan_inf_with_nan PASSED             [ 95%]
tests/test_validation.py::test_check_nan_inf_with_inf PASSED             [ 96%]
tests/test_validation.py::test_validate_dtype_correct PASSED             [ 96%]
tests/test_validation.py::test_validate_dtype_mismatch PASSED            [ 96%]
tests/test_validation.py::test_tensor_stats PASSED                       [ 97%]
tests/test_validation.py::test_gradient_stats_and_flow_with_simple_model PASSED [ 97%]
tests/test_validation.py::test_validate_config_valid PASSED              [ 98%]
tests/test_validation.py::test_validate_architecture_consistency_valid PASSED [ 98%]
tests/test_validation.py::test_validate_device_compatibility PASSED      [ 98%]
tests/test_validation.py::test_estimate_model_memory PASSED              [ 99%]
tests/test_validation.py::test_memory_tracker_cpu PASSED                 [ 99%]
tests/test_validation.py::test_gpu_memory_usage_no_crash PASSED          [100%]

============================== warnings summary ===============================
tests/test_scheduler.py::test_constant_lr_progression
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:71: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

tests/test_scheduler.py::test_linear_warmup_and_cosine_decay
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:98: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()  # step=1

tests/test_scheduler.py::test_linear_decay_progression
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:142: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

tests/test_scheduler.py::test_polynomial_decay_progression
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:167: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

tests/test_scheduler.py::test_step_decay_intervals
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:187: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

tests/test_scheduler.py::test_exponential_decay
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:207: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

tests/test_scheduler.py::test_scheduler_state_dict_roundtrip
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:224: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    sched1.step()

tests/test_scheduler.py::test_scheduler_state_dict_roundtrip
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\training\scheduler.py:144: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
    self.step(self.last_epoch)

tests/test_scheduler.py::test_scheduler_long_horizon_simulation
  C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\tests\test_scheduler.py:274: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler.step()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 261 passed, 4 skipped, 9 warnings in 233.47s (0:03:53) ============
```
