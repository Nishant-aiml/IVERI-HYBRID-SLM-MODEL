# Titans Memory Verification Report (Phase 6.3.2 OBJ2)

**Generated:** 2026-08-21T13:23:00Z  
**Protocol:** Phase-6.3.2-OBJ2  
**Device:** cpu  

## Executive Verdict

**Production integrated path:** `PASS`

**Writes occur in production:** `True`

Instrumentation captures memory state at: **before forward**, **after forward**, **after backward**, **after optimizer**, and **after second forward** (persistence).

## Proof: Online Writes Occur in Production

1. Production `Backbone.forward` calls `self.titans.forward_with_injection(x, entropy)` (model/backbone.py), which invokes `TitansMemory.forward()` for online read/update/write and applies entropy gating.
2. `forward()` runs a sequential loop: read via `_forward_mlp`, compute local loss, and call `MemoryUpdater.update` per patch step.
3. Runtime instrumentation on production path: forward_calls=4, updater_calls=12, read_calls=0, write_calls=0, inject_calls=0.
4. Online weights diverge from expanded base_W1 after production forward (online_weight_delta=5.728700e-01).
5. Titans telemetry reports updates after production forward: update_count=12, avg_update_mag=6.693432e-02.
6. Backbone telemetry `titans_write_count` is sourced from measured Titans telemetry (not hardcoded B*P).

## Path Comparison (runtime call counts)

| Path | read | write | forward | inject | updater.update | Online ΔW1 | Persists 2nd call |
|------|-----:|------:|--------:|-------:|---------------:|-----------:|:---------------:|
| TitansMemory.forward (isolated) | 0 | 0 | 2 | 0 | 12 | 5.992e-01 | yes |
| TitansMemory.write (isolated) | 0 | 2 | 0 | 0 | 12 | 5.780e-01 | no |
| TitansMemory.inject (isolated) | 2 | 0 | 0 | 2 | 0 | 0.000e+00 | yes |
| Backbone.production (forward→update→gate) | 0 | 0 | 4 | 0 | 12 | 5.729e-01 | no |

## Lifecycle Snapshots (production path)

| Stage | base_W1 ‖·‖ | online_W1 ‖·‖ | telemetry updates | avg update mag |
|-------|----------:|---------------:|------------------:|---------------:|
| before_forward | 5.670310 | N/A | 0 | 0.000000e+00 |
| after_forward | 5.670310 | 7.455489 | 12 | 6.693432e-02 |
| after_backward | 5.670310 | 7.455489 | 12 | 6.693432e-02 |
| after_optimizer | 5.670302 | 7.455489 | 12 | 6.693432e-02 |
| after_second_forward | 5.670302 | 7.455480 | 12 | 6.693417e-02 |

## Gradient Flow

| Path | ‖grad base_W1‖ | ‖grad base_W2‖ | ‖grad q_proj‖ | online W changed in fwd | opt changed base_W1 | opt changed online_W1 |
|---|--:|--:|--:|:---:|:---:|:---:|
| Backbone.forward_with_injection (production) | 6.9922e+01 | 6.6095e+01 | 6.5780e+01 | False | True | False |
| TitansMemory.forward (isolated) | 1.5086e+02 | 1.2786e+02 | 1.6047e+02 | False | False | False |

## Capability Matrix (measured)

| Capability | Isolated forward | Isolated write | Production forward |
|------------|:----------------:|:--------------:|:------------------:|
| Memory reads | False | False | False |
| Memory writes | False | True | False |
| Online updates (updater) | True | True | True |
| Online weight replacement | True | True | True |
| Persistence (2nd identical call) | True | False | False |

## Isolated Forward Reference

- `updater.update` calls during isolated `TitansMemory.forward`: **12**
- Average update magnitude (isolated): **6.934527e-02**

> Phase 6.3.2 OBJ2 wires `TitansMemory.forward_with_injection` into `Backbone.forward`.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ2",
  "timestamp_utc": "2026-08-21T13:23:00Z",
  "device": "cpu",
  "production_verdict": "PASS",
  "writes_occur_in_production": true,
  "write_absence_proof": [],
  "write_presence_proof": [
    "Production `Backbone.forward` calls `self.titans.forward_with_injection(x, entropy)` (model/backbone.py), which invokes `TitansMemory.forward()` for online read/update/write and applies entropy gating.",
    "`forward()` runs a sequential loop: read via `_forward_mlp`, compute local loss, and call `MemoryUpdater.update` per patch step.",
    "Runtime instrumentation on production path: forward_calls=4, updater_calls=12, read_calls=0, write_calls=0, inject_calls=0.",
    "Online weights diverge from expanded base_W1 after production forward (online_weight_delta=5.728700e-01).",
    "Titans telemetry reports updates after production forward: update_count=12, avg_update_mag=6.693432e-02.",
    "Backbone telemetry `titans_write_count` is sourced from measured Titans telemetry (not hardcoded B*P)."
  ],
  "path_results": [
    {
      "path_name": "TitansMemory.forward (isolated)",
      "read_calls": 0,
      "write_calls": 0,
      "forward_calls": 2,
      "inject_calls": 0,
      "updater_calls": 12,
      "online_weight_delta_after": 0.5991661548614502,
      "persistence_across_second_call": true,
      "telemetry_reports_writes": true,
      "snapshots": [
        {
          "stage": "before_forward",
          "base_w1_norm": 5.788279056549072,
          "base_w1_sum": -2.4745631217956543,
          "online_w1_norm": null,
          "online_w1_sum": null,
          "online_weights_present": false,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        },
        {
          "stage": "after_forward",
          "base_w1_norm": 5.788279056549072,
          "base_w1_sum": -2.4745631217956543,
          "online_w1_norm": 7.591705799102783,
          "online_w1_sum": -4.585822582244873,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06934526562690735
        },
        {
          "stage": "after_second_forward",
          "base_w1_norm": 5.788279056549072,
          "base_w1_sum": -2.4745631217956543,
          "online_w1_norm": 7.591705799102783,
          "online_w1_sum": -4.585822582244873,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06934526562690735
        }
      ]
    },
    {
      "path_name": "TitansMemory.write (isolated)",
      "read_calls": 0,
      "write_calls": 2,
      "forward_calls": 0,
      "inject_calls": 0,
      "updater_calls": 12,
      "online_weight_delta_after": 0.577969491481781,
      "persistence_across_second_call": false,
      "telemetry_reports_writes": true,
      "snapshots": [
        {
          "stage": "before_forward",
          "base_w1_norm": 5.692613124847412,
          "base_w1_sum": 7.88585901260376,
          "online_w1_norm": null,
          "online_w1_sum": null,
          "online_weights_present": false,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        },
        {
          "stage": "after_forward",
          "base_w1_norm": 5.692613124847412,
          "base_w1_sum": 7.88585901260376,
          "online_w1_norm": 7.485001087188721,
          "online_w1_sum": 14.93669605255127,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06639742851257324
        },
        {
          "stage": "after_second_forward",
          "base_w1_norm": 5.692613124847412,
          "base_w1_sum": 7.88585901260376,
          "online_w1_norm": 6.9518914222717285,
          "online_w1_sum": 14.381261825561523,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06310727695624034
        }
      ]
    },
    {
      "path_name": "TitansMemory.inject (isolated)",
      "read_calls": 2,
      "write_calls": 0,
      "forward_calls": 0,
      "inject_calls": 2,
      "updater_calls": 0,
      "online_weight_delta_after": 0.0,
      "persistence_across_second_call": true,
      "telemetry_reports_writes": false,
      "snapshots": [
        {
          "stage": "before_forward",
          "base_w1_norm": 5.640386581420898,
          "base_w1_sum": -3.75274658203125,
          "online_w1_norm": null,
          "online_w1_sum": null,
          "online_weights_present": false,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        },
        {
          "stage": "after_forward",
          "base_w1_norm": 5.640386581420898,
          "base_w1_sum": -3.75274658203125,
          "online_w1_norm": 7.976712226867676,
          "online_w1_sum": -7.505493640899658,
          "online_weights_present": true,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        },
        {
          "stage": "after_second_forward",
          "base_w1_norm": 5.640386581420898,
          "base_w1_sum": -3.75274658203125,
          "online_w1_norm": 7.976712226867676,
          "online_w1_sum": -7.505493640899658,
          "online_weights_present": true,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        }
      ]
    },
    {
      "path_name": "Backbone.production (forward\u2192update\u2192gate)",
      "read_calls": 0,
      "write_calls": 0,
      "forward_calls": 4,
      "inject_calls": 0,
      "updater_calls": 12,
      "online_weight_delta_after": 0.5728699564933777,
      "persistence_across_second_call": false,
      "telemetry_reports_writes": true,
      "snapshots": [
        {
          "stage": "before_forward",
          "base_w1_norm": 5.6703104972839355,
          "base_w1_sum": -3.0562314987182617,
          "online_w1_norm": null,
          "online_w1_sum": null,
          "online_weights_present": false,
          "telemetry_update_count": 0,
          "telemetry_avg_update_mag": 0.0
        },
        {
          "stage": "after_forward",
          "base_w1_norm": 5.6703104972839355,
          "base_w1_sum": -3.0562314987182617,
          "online_w1_norm": 7.455488681793213,
          "online_w1_sum": -5.691399574279785,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06693431735038757
        },
        {
          "stage": "after_backward",
          "base_w1_norm": 5.6703104972839355,
          "base_w1_sum": -3.0562314987182617,
          "online_w1_norm": 7.455488681793213,
          "online_w1_sum": -5.691399574279785,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06693431735038757
        },
        {
          "stage": "after_optimizer",
          "base_w1_norm": 5.670301914215088,
          "base_w1_sum": -3.0565743446350098,
          "online_w1_norm": 7.455488681793213,
          "online_w1_sum": -5.691399574279785,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06693431735038757
        },
        {
          "stage": "after_second_forward",
          "base_w1_norm": 5.670301914215088,
          "base_w1_sum": -3.0565743446350098,
          "online_w1_norm": 7.455479621887207,
          "online_w1_sum": -5.691999435424805,
          "online_weights_present": true,
          "telemetry_update_count": 12,
          "telemetry_avg_update_mag": 0.06693416833877563
        }
      ]
    }
  ],
  "gradient_results": [
    {
      "path_name": "Backbone.forward_with_injection (production)",
      "base_w1_grad_norm": 69.92208099365234,
      "base_w2_grad_norm": 66.09538269042969,
      "q_proj_grad_norm": 65.78028869628906,
      "online_weights_changed_during_forward": false,
      "optimizer_changed_base_w1": true,
      "optimizer_changed_online_w1": false
    },
    {
      "path_name": "TitansMemory.forward (isolated)",
      "base_w1_grad_norm": 150.86334228515625,
      "base_w2_grad_norm": 127.86161041259766,
      "q_proj_grad_norm": 160.4654998779297,
      "online_weights_changed_during_forward": false,
      "optimizer_changed_base_w1": false,
      "optimizer_changed_online_w1": false
    }
  ],
  "isolated_forward_updates": 12,
  "isolated_forward_avg_update_mag": 0.06934526562690735
}
```
