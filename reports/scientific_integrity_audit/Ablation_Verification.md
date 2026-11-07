# Ablation Verification Report (Phase 6.3.1F)

**Generated:** 2026-08-21T13:14:12Z  
**Protocol:** Phase-6.3.1F  
**Device:** cpu  

## Executive Verdict

**Physical ablation framework:** `PASS`

**Campaign overrides apply to ModelConfig:** `True`

**Pairwise distinct architectures:** `True`

## Ablation Probes

| Ablation | Flag | Absent from forward path | Param Δ | Output Δ (L1) |
|----------|------|:------------------------:|--------:|----------------:|
| no_titans | `use_titans=False` | True | 20708 | 5.6891e-02 |
| no_blt | `use_blt=False` | True | 99779 | 1.3531e-01 |
| no_mor | `use_mor=False` | True | 0 | 4.6923e-01 |
| no_moe | `use_moe=False` | True | 295944 | 3.7101e-01 |
| no_entropy_routing | `use_entropy_routing=False` | True | 0 | 0.0000e+00 |

## Architecture Fingerprints

| Config | Params | Titans | BLT | MoE router | Dense FFN | MoR | Output Σ |
|--------|-------:|:------:|:---:|:----------:|:---------:|:---:|---------:|
| none | 728122 | True | True | True | False | True | 3.3523e+02 |
| no_titans | 707414 | False | True | True | False | True | 4.3016e+02 |
| no_blt | 628343 | True | False | True | False | True | 1.0470e+03 |
| no_mor | 728122 | True | True | True | False | False | 3.1894e+02 |
| no_moe | 432178 | True | True | False | True | True | 3.7441e+02 |
| no_entropy_routing | 728122 | True | True | True | False | True | 3.3573e+02 |

## Pairwise Distinctness

All baseline + ablation configurations produce unique architecture signatures.

## Antipattern Detection

No unused flags, dead configuration, or silent fallback detected.

## Proof Statements

1. **no_titans:** Backbone skips TitansMemory construction and forward_with_injection when use_titans=False.
2. **no_blt:** IVERIModel uses byte_embed + bypass_lm_head instead of BLT stack when use_blt=False.
3. **no_mor:** BackboneBlock calls sub_block once without RecursionEngine when use_mor=False.
4. **no_moe:** BackboneSubBlock uses dense SwiGLU FFN instead of SparseMoERouter when use_moe=False.
5. **no_entropy_routing:** SparseMoERouter ignores entropy bias when use_entropy_routing=False (full_diff=3.312e-02, ablated_diff=0.000e+00).

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.1F",
  "timestamp_utc": "2026-08-21T13:14:12Z",
  "device": "cpu",
  "production_verdict": "PASS",
  "probes": [
    {
      "ablation_key": "no_titans",
      "flag_field": "use_titans",
      "flag_value": false,
      "component_absent": true,
      "parameter_delta": 20708,
      "output_diff_l1": 0.05689110979437828,
      "proof": "Backbone skips TitansMemory construction and forward_with_injection when use_titans=False."
    },
    {
      "ablation_key": "no_blt",
      "flag_field": "use_blt",
      "flag_value": false,
      "component_absent": true,
      "parameter_delta": 99779,
      "output_diff_l1": 0.135306715965271,
      "proof": "IVERIModel uses byte_embed + bypass_lm_head instead of BLT stack when use_blt=False."
    },
    {
      "ablation_key": "no_mor",
      "flag_field": "use_mor",
      "flag_value": false,
      "component_absent": true,
      "parameter_delta": 0,
      "output_diff_l1": 0.4692317247390747,
      "proof": "BackboneBlock calls sub_block once without RecursionEngine when use_mor=False."
    },
    {
      "ablation_key": "no_moe",
      "flag_field": "use_moe",
      "flag_value": false,
      "component_absent": true,
      "parameter_delta": 295944,
      "output_diff_l1": 0.3710099756717682,
      "proof": "BackboneSubBlock uses dense SwiGLU FFN instead of SparseMoERouter when use_moe=False."
    },
    {
      "ablation_key": "no_entropy_routing",
      "flag_field": "use_entropy_routing",
      "flag_value": false,
      "component_absent": true,
      "parameter_delta": 0,
      "output_diff_l1": 0.0,
      "proof": "SparseMoERouter ignores entropy bias when use_entropy_routing=False (full_diff=3.312e-02, ablated_diff=0.000e+00)."
    }
  ],
  "campaign_overrides_applied": true,
  "fingerprints": [
    {
      "label": "none",
      "flags": {
        "use_titans": true,
        "use_blt": true,
        "use_mor": true,
        "use_moe": true,
        "use_entropy_routing": true
      },
      "param_count": 728122,
      "has_titans": true,
      "has_entropy_model": true,
      "has_moe_router": true,
      "has_dense_ffn": false,
      "mor_active": true,
      "output_checksum": 335.2326965332031
    },
    {
      "label": "no_titans",
      "flags": {
        "use_titans": false,
        "use_blt": true,
        "use_mor": true,
        "use_moe": true,
        "use_entropy_routing": true
      },
      "param_count": 707414,
      "has_titans": false,
      "has_entropy_model": true,
      "has_moe_router": true,
      "has_dense_ffn": false,
      "mor_active": true,
      "output_checksum": 430.15887451171875
    },
    {
      "label": "no_blt",
      "flags": {
        "use_titans": true,
        "use_blt": false,
        "use_mor": true,
        "use_moe": true,
        "use_entropy_routing": true
      },
      "param_count": 628343,
      "has_titans": true,
      "has_entropy_model": false,
      "has_moe_router": true,
      "has_dense_ffn": false,
      "mor_active": true,
      "output_checksum": 1047.0025634765625
    },
    {
      "label": "no_mor",
      "flags": {
        "use_titans": true,
        "use_blt": true,
        "use_mor": false,
        "use_moe": true,
        "use_entropy_routing": true
      },
      "param_count": 728122,
      "has_titans": true,
      "has_entropy_model": true,
      "has_moe_router": true,
      "has_dense_ffn": false,
      "mor_active": false,
      "output_checksum": 318.943359375
    },
    {
      "label": "no_moe",
      "flags": {
        "use_titans": true,
        "use_blt": true,
        "use_mor": true,
        "use_moe": false,
        "use_entropy_routing": true
      },
      "param_count": 432178,
      "has_titans": true,
      "has_entropy_model": true,
      "has_moe_router": false,
      "has_dense_ffn": true,
      "mor_active": true,
      "output_checksum": 374.4051208496094
    },
    {
      "label": "no_entropy_routing",
      "flags": {
        "use_titans": true,
        "use_blt": true,
        "use_mor": true,
        "use_moe": true,
        "use_entropy_routing": false
      },
      "param_count": 728122,
      "has_titans": true,
      "has_entropy_model": true,
      "has_moe_router": true,
      "has_dense_ffn": false,
      "mor_active": true,
      "output_checksum": 335.7309265136719
    }
  ],
  "pairwise_distinct": true,
  "antipatterns": []
}
```
