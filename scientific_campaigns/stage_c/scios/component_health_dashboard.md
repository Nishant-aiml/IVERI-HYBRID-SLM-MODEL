# SciOS Component Health Dashboard
*Updated: 2026-07-26T07:23:07Z*

| Component | Health Score (0-100) | Latency (ms) | Utilization | Benefit | Stability | Blocker |
|---|---|---|---|---|---|---|
| **Router** | **83/100** | 1.2ms | 78.5% | Dynamic pathway selection | HIGH | Entropy logging missing (MISSING-003) |
| **Titans** | **71/100** | 4.5ms | 100.0% | Persistent memory bank | HIGH | Long-context sweep missing (MISSING-002) |
| **MoR** | **54/100** | 12.8ms | 50.0% | Recursive depth representation | MEDIUM | High overhead at short sequence length (Needs v0.92 adaptive fix) |
| **BLT** | **91/100** | 2.1ms | 94.0% | Byte-level tokenization-free input processing | VERY_HIGH | None |
