# Entropy → Routing Verification Report (Phase 6.3.2 OBJ3)

**Generated:** 2026-07-06T15:26:25Z  
**Protocol:** Phase-6.3.2-OBJ3  
**Device:** cpu  

## Executive Verdict

**Production entropy-conditioned MoE routing:** `PASS`

- Entropy reaches router module: `True`
- Entropy reaches routing logits: `True`
- Entropy reaches expert probabilities: `True`
- Entropy reaches routing decisions: `True`

## Proof: Entropy Conditions MoE Routing

1. `SparseMoERouter.forward(x, entropy=...)` accepts patch entropy and adds `w_entropy(entropy)` to gating logits before top-k selection.
2. `BackboneSubBlock` passes `entropy` from kwargs into `moe_router`.
3. `RecursionEngine` forwards `entropy` to wrapped blocks (no longer stripped).
4. Fixed-hidden experiment: max logit change = 2.293056e-02, max probability change = 8.716822e-03, decisions changed = True.

## Fixed-Hidden Entropy Perturbation Experiment

Router input hidden states `x` were held fixed while patch entropy was scaled across [0.0, 0.25, 0.5, 0.75, 1.0].

- Max change in gating logits (‖Δlogit‖_∞): `2.293056e-02`
- Max change in routing probabilities (‖Δp‖_∞): `8.716822e-03`
- Routing decisions changed (any top-k index difference): `True`

Entropy is wired into MoE gating via `w_entropy` and reaches logits, probabilities, and routing decisions when hidden states are held fixed.

## Routing Snapshots (fixed hidden)

| Label | Entropy scale | Sample logits | Sample probabilities | Sample expert indices |
|-------|---------------:|--------------|---------------------:|----------------------:|
| baseline | 0.50 | `[0.263, -0.123, 0.070, -0.023]` | `[0.548, 0.452]` | `[0, 2]` |
| fixed_hidden_entropy_scale_0.00 | 0.00 | `[0.286, -0.120, 0.058, -0.010]` | `[0.557, 0.443]` | `[0, 2]` |
| fixed_hidden_entropy_scale_0.25 | 0.25 | `[0.275, -0.122, 0.064, -0.016]` | `[0.552, 0.448]` | `[0, 2]` |
| fixed_hidden_entropy_scale_0.50 | 0.50 | `[0.263, -0.123, 0.070, -0.023]` | `[0.548, 0.452]` | `[0, 2]` |
| fixed_hidden_entropy_scale_0.75 | 0.75 | `[0.252, -0.125, 0.076, -0.029]` | `[0.544, 0.456]` | `[0, 2]` |
| fixed_hidden_entropy_scale_1.00 | 1.00 | `[0.240, -0.127, 0.082, -0.035]` | `[0.539, 0.461]` | `[0, 2]` |

## Backbone Sanity Check

- Expert utilization histogram changed when entropy perturbed: `False`

> Phase 6.3.2 OBJ3 implements Patent 3: MoE expert routing conditioned on BLT byte-patch entropy via `w_entropy` logit bias.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ3",
  "timestamp_utc": "2026-07-06T15:26:25Z",
  "device": "cpu",
  "production_verdict": "PASS",
  "entropy_reaches_router": true,
  "entropy_reaches_routing_logits": true,
  "entropy_reaches_expert_probabilities": true,
  "entropy_reaches_routing_decisions": true,
  "max_logit_diff_with_fixed_hidden": 0.022930562496185303,
  "max_prob_diff_with_fixed_hidden": 0.008716821670532227,
  "changed_decisions_with_fixed_hidden": true,
  "backbone_routing_changed_when_entropy_perturbed": false,
  "break_description": "Entropy is wired into MoE gating via `w_entropy` and reaches logits, probabilities, and routing decisions when hidden states are held fixed.",
  "presence_proof": [
    "`SparseMoERouter.forward(x, entropy=...)` accepts patch entropy and adds `w_entropy(entropy)` to gating logits before top-k selection.",
    "`BackboneSubBlock` passes `entropy` from kwargs into `moe_router`.",
    "`RecursionEngine` forwards `entropy` to wrapped blocks (no longer stripped).",
    "Fixed-hidden experiment: max logit change = 2.293056e-02, max probability change = 8.716822e-03, decisions changed = True."
  ],
  "snapshots": [
    {
      "label": "baseline",
      "entropy_scale": 0.5,
      "sample_logits": [
        0.263336181640625,
        -0.12343103438615799,
        0.07036612927913666,
        -0.022612236440181732
      ],
      "sample_probs": [
        0.5480933785438538,
        0.45190662145614624
      ],
      "sample_indices": [
        0,
        2
      ]
    },
    {
      "label": "fixed_hidden_entropy_scale_0.00",
      "entropy_scale": 0.0,
      "sample_logits": [
        0.2862667143344879,
        -0.12009365856647491,
        0.058421310037374496,
        -0.009966433048248291
      ],
      "sample_probs": [
        0.5567162036895752,
        0.4432837963104248
      ],
      "sample_indices": [
        0,
        2
      ]
    },
    {
      "label": "fixed_hidden_entropy_scale_0.25",
      "entropy_scale": 0.25,
      "sample_logits": [
        0.27480143308639526,
        -0.12176235020160675,
        0.06439372152090073,
        -0.01628933474421501
      ],
      "sample_probs": [
        0.5524086952209473,
        0.44759127497673035
      ],
      "sample_indices": [
        0,
        2
      ]
    },
    {
      "label": "fixed_hidden_entropy_scale_0.50",
      "entropy_scale": 0.5,
      "sample_logits": [
        0.263336181640625,
        -0.12343103438615799,
        0.07036612927913666,
        -0.022612236440181732
      ],
      "sample_probs": [
        0.5480933785438538,
        0.45190662145614624
      ],
      "sample_indices": [
        0,
        2
      ]
    },
    {
      "label": "fixed_hidden_entropy_scale_0.75",
      "entropy_scale": 0.75,
      "sample_logits": [
        0.25187090039253235,
        -0.12509971857070923,
        0.07633854448795319,
        -0.028935138136148453
      ],
      "sample_probs": [
        0.5437707901000977,
        0.4562292695045471
      ],
      "sample_indices": [
        0,
        2
      ]
    },
    {
      "label": "fixed_hidden_entropy_scale_1.00",
      "entropy_scale": 1.0,
      "sample_logits": [
        0.2404056191444397,
        -0.12676841020584106,
        0.08231095224618912,
        -0.03525803983211517
      ],
      "sample_probs": [
        0.5394415259361267,
        0.4605584144592285
      ],
      "sample_indices": [
        0,
        2
      ]
    }
  ]
}
```
