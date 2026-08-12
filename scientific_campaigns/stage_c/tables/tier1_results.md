# Stage C Tier-1 Ablation Results
**Source:** `tier1_ablations_and_failures.json` | SHA256: `61f6c006c771792e` | Generated: `2026-07-26T06:58:00.620785Z`

| Model | Seeds | Mean Loss | ±95%CI | Throughput (tok/s) | VRAM (MB) | Grad Explosions |
|-------|-------|-----------|--------|-------------------|-----------|----------------|
| Baseline Dense Transformer | 3 | 1.3664 | ±0.0179 | 13016.3±2345.8 | 394 | 0 |
| IVERI Full v0.91 | 3 | 3.1425 | ±0.0062 | 172.5±13.0 | 828 | 5 |
| IVERI No Entropy | 3 | 3.1420 | ±0.0065 | 186.3±18.4 | 828 | 5 |
| IVERI No Titans | 3 | 3.1420 | ±0.0065 | 271.2±14.6 | 691 | 7 |
| IVERI No MoR | 3 | 3.1365 | ±0.0115 | 359.2±24.0 | 423 | 1 |

## Pairwise Comparisons (vs IVERI Full v0.91)

| Comparison | dLoss | p-value | Cohen d | dThroughput | dVRAM |
|-----------|-------|---------|---------|-------------|-------|
| Full_vs_Dense | -1.7761 | <0.01 | 150.044 | +7443.6% | -52.5% |
| Full_vs_Entropy | -0.0005 | >0.10 | 0.092 | +8.0% | -0.1% |
| Full_vs_Titans | -0.0006 | >0.10 | 0.099 | +57.2% | -16.6% |
| Full_vs_MoR | -0.0060 | >0.10 | 0.737 | +108.2% | -49.0% |