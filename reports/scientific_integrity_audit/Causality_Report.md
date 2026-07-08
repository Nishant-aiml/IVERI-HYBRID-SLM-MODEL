# Causality Report — Runtime Perturbation Audit (Phase 6.3.2 / Objective 1)

**Generated:** 2026-07-08T05:31:13Z  
**Protocol:** Phase-6.3.2-OBJ1  
**Device:** cpu  
**Sequence length:** 32  
**Model:** hidden_dim=64, layers=2  

## Executive Verdict

**End-to-end causality:** `PASS`

Perturbation protocol: for each cut index `i`, bytes at positions `> i` are replaced with random values; outputs at positions `<= i` (and patch tensors fully contained in `[0, i]`) must remain identical.

## Causality Restoration (Phase 6.3.2 Objective 1)

BLT stack updated for strict autoregressive byte modeling:

1. **ByteEntropyModel:** left-padded causal Conv1d (no symmetric padding).
2. **BLTByteEncoder:** within-patch causal self-attention mask.
3. **BLTByteDecoder:** cross-attention limited to patches with `patch_end <= query_index`.

## Measured Error Summary

| Corpus | Module | Positions Tested | Leaking | Max Abs Error | Max Rel Error | First Leak @ i |
|--------|--------|------------------:|--------:|--------------:|--------------:|---------------:|
| random | ByteEntropyModel | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| random | DynamicPatcher | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| random | BLTByteEncoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| random | PatchEntropyPool | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| random | Backbone | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| random | BLTByteDecoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | ByteEntropyModel | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | DynamicPatcher | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | BLTByteEncoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | PatchEntropyPool | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | Backbone | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| english | BLTByteDecoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | ByteEntropyModel | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | DynamicPatcher | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | BLTByteEncoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | PatchEntropyPool | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | Backbone | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| code | BLTByteDecoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | ByteEntropyModel | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | DynamicPatcher | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | BLTByteEncoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | PatchEntropyPool | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | Backbone | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |
| binary | BLTByteDecoder | 30 | 0 | 0.000000e+00 | 0.000000e+00 |  |

## Module Notes

### ByteEntropyModel
- **Tensor compared:** `byte_entropy[:, :cut+1, :]`
- **Known masking gap:** Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)

### DynamicPatcher
- **Tensor compared:** `boundary_map[:, :cut+1]`
- **Known masking gap:** Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix

### BLTByteEncoder
- **Tensor compared:** `latent_patches[:, causal_patch_indices, :]`
- **Known masking gap:** Within-patch causal MultiheadAttention (no future byte keys)

### PatchEntropyPool
- **Tensor compared:** `patch_entropy[:, causal_patch_indices, :]`
- **Known masking gap:** Mean pool over patch bytes; invariant when byte entropy is causal

### Backbone
- **Tensor compared:** `backbone_out[:, causal_patch_indices, :]`
- **Known masking gap:** Patch-level causal attention; inherits causal patch inputs

### BLTByteDecoder
- **Tensor compared:** `logits[:, :cut+1, :]`
- **Known masking gap:** Cross-attention keys masked to patches with patch_end <= query byte index

## Pass Tolerance

Positions pass when `torch.allclose(ref, pert, atol=1e-6, rtol=1e-5)`.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ1",
  "timestamp_utc": "2026-07-08T05:31:13Z",
  "device": "cpu",
  "seq_len": 32,
  "model_hidden_dim": 64,
  "model_num_layers": 2,
  "corpora": [
    "random",
    "english",
    "code",
    "binary"
  ],
  "module_summaries": [
    {
      "module": "ByteEntropyModel",
      "tensor_path": "byte_entropy[:, :cut+1, :]",
      "masking_issue": "Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "DynamicPatcher",
      "tensor_path": "boundary_map[:, :cut+1]",
      "masking_issue": "Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteEncoder",
      "tensor_path": "latent_patches[:, causal_patch_indices, :]",
      "masking_issue": "Within-patch causal MultiheadAttention (no future byte keys)",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "PatchEntropyPool",
      "tensor_path": "patch_entropy[:, causal_patch_indices, :]",
      "masking_issue": "Mean pool over patch bytes; invariant when byte entropy is causal",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "Backbone",
      "tensor_path": "backbone_out[:, causal_patch_indices, :]",
      "masking_issue": "Patch-level causal attention; inherits causal patch inputs",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteDecoder",
      "tensor_path": "logits[:, :cut+1, :]",
      "masking_issue": "Cross-attention keys masked to patches with patch_end <= query byte index",
      "corpus": "random",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "ByteEntropyModel",
      "tensor_path": "byte_entropy[:, :cut+1, :]",
      "masking_issue": "Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "DynamicPatcher",
      "tensor_path": "boundary_map[:, :cut+1]",
      "masking_issue": "Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteEncoder",
      "tensor_path": "latent_patches[:, causal_patch_indices, :]",
      "masking_issue": "Within-patch causal MultiheadAttention (no future byte keys)",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "PatchEntropyPool",
      "tensor_path": "patch_entropy[:, causal_patch_indices, :]",
      "masking_issue": "Mean pool over patch bytes; invariant when byte entropy is causal",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "Backbone",
      "tensor_path": "backbone_out[:, causal_patch_indices, :]",
      "masking_issue": "Patch-level causal attention; inherits causal patch inputs",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteDecoder",
      "tensor_path": "logits[:, :cut+1, :]",
      "masking_issue": "Cross-attention keys masked to patches with patch_end <= query byte index",
      "corpus": "english",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "ByteEntropyModel",
      "tensor_path": "byte_entropy[:, :cut+1, :]",
      "masking_issue": "Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "DynamicPatcher",
      "tensor_path": "boundary_map[:, :cut+1]",
      "masking_issue": "Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteEncoder",
      "tensor_path": "latent_patches[:, causal_patch_indices, :]",
      "masking_issue": "Within-patch causal MultiheadAttention (no future byte keys)",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "PatchEntropyPool",
      "tensor_path": "patch_entropy[:, causal_patch_indices, :]",
      "masking_issue": "Mean pool over patch bytes; invariant when byte entropy is causal",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "Backbone",
      "tensor_path": "backbone_out[:, causal_patch_indices, :]",
      "masking_issue": "Patch-level causal attention; inherits causal patch inputs",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteDecoder",
      "tensor_path": "logits[:, :cut+1, :]",
      "masking_issue": "Cross-attention keys masked to patches with patch_end <= query byte index",
      "corpus": "code",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "ByteEntropyModel",
      "tensor_path": "byte_entropy[:, :cut+1, :]",
      "masking_issue": "Causal left-padded Conv1d (kernel_size=3, padding=0 + left pad)",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "DynamicPatcher",
      "tensor_path": "boundary_map[:, :cut+1]",
      "masking_issue": "Sequential boundary scan; entropy at i uses only bytes <= i after causal entropy fix",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteEncoder",
      "tensor_path": "latent_patches[:, causal_patch_indices, :]",
      "masking_issue": "Within-patch causal MultiheadAttention (no future byte keys)",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "PatchEntropyPool",
      "tensor_path": "patch_entropy[:, causal_patch_indices, :]",
      "masking_issue": "Mean pool over patch bytes; invariant when byte entropy is causal",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "Backbone",
      "tensor_path": "backbone_out[:, causal_patch_indices, :]",
      "masking_issue": "Patch-level causal attention; inherits causal patch inputs",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    },
    {
      "module": "BLTByteDecoder",
      "tensor_path": "logits[:, :cut+1, :]",
      "masking_issue": "Cross-attention keys masked to patches with patch_end <= query byte index",
      "corpus": "binary",
      "positions_tested": 30,
      "leaking_positions": 0,
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "first_leak_index": null,
      "passed": true
    }
  ],
  "end_to_end_verdict": "PASS",
  "primary_leak_module": null,
  "primary_tensor": null,
  "primary_attention_path": null,
  "primary_masking_issue": null
}
```
