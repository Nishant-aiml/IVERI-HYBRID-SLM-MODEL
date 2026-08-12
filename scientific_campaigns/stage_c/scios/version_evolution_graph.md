# SciOS Version Evolution Graph
*Updated: 2026-07-26T07:23:07Z*

## Version Nodes
- **v0.90** (LEGACY): Initial IVERI baseline with enabled gradient checkpointing and static MoR.
- **v0.91** (ACTIVE): Disabled gradient checkpointing for short context (+41.7% throughput).
- **v0.91A** (STABLE_PATCH): Added recursive gradient norm clipping within MoR layers.
- **v0.92** (TARGET): Adaptive MoR depth scheduling (depth=0 for seq_len < 128).
- **v1.0** (RELEASE_GOAL): Fully integrated and statistically validated IVERI architecture.

## Transition Edges
- `v0.90` → `v0.91`: Profiling evidence showed 41.7% checkpointing overhead at short seq_len *(Evidence: latency_10x.json)*
- `v0.91` → `v0.91A`: Failure analysis traced explosions to MoR recursive accumulation *(Evidence: lr_warmup_sweep_results.json)*
- `v0.91A` → `v0.92`: Ablations proved No-MoR gives 2x throughput at short context without loss penalty *(Evidence: tier1_results.md)*
