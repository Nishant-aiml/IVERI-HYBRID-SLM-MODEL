# IVERI Core — Patent Enablement Report
Generated: 2026-07-26T07:04:37Z

> USPTO enablement requirement: A patent claim must be enabled by experimental evidence
> sufficient for a person skilled in the art to reproduce the claimed technical effect.

| Patent | Claim | Enablement Status | Evidence | Blocker |
|--------|-------|-------------------|----------|---------|
| Patent-1 (Entropy Routing) | Entropy-based gating | NOT_YET_ENABLED | No router entropy logs | MISSING-003 |
| Patent-2 (Titans Memory) | Long-context memory advantage | NOT_YET_ENABLED | No long-context data | MISSING-002 |
| Patent-3 (Adaptive MoR) | Depth-adaptive recursion | PARTIALLY_ENABLED | Overhead data exists; adaptive mechanism unimplemented | MISSING-005 |
| Patent-3 (Adaptive MoR) | MoR overhead reduction | ENABLED | CLAIM-002 supports this | None |
| Patent-4 (Integrated IVERI) | Architectural integration | NOT_YET_ENABLED | All other patents must be enabled first | All MISSING items |

## Most Enabled Claim

**Patent-3, MoR overhead claim**: Evidence exists (CLAIM-002) that removing MoR
provides 2x throughput improvement with no statistically detected loss penalty at
short sequences. This is sufficient for a narrow enablement claim at seq_len=64.
The broader adaptive MoR claim requires MISSING-005.

## Earliest Patent Filing Recommendation

File provisional application now covering: architectural description + CLAIM-002 + CLAIM-003.
Complete utility application after MISSING-002 and MISSING-005 are resolved.