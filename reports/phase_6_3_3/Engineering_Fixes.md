# Engineering Fixes

- BYTE_VOCAB_SIZE test migration (256 -> 259 logits)
- loss_mask._mask_padding accepts list[int]
- generation_inspector safe byte decode
- inference/ package added
- logger PermissionError fallback for save_dir and W&B init
- coding mock dataset uses torch Dataset; num_workers=0 in tests
- publication backward-compat test seeds MEASURED registry rows
- baseline_transformer asserts RAW_BYTE_VOCAB_SIZE (256)
- logging test mocks wandb.init failure for portable fallback
- conftest base_config: num_workers=0, CPU fallback when no CUDA
- Stage 3B proprietary ingest pipeline (proprietary_ingest.py)
- Deployment guide and measured inference benchmark script
**Generated:** 2026-07-07T04:41:49Z
