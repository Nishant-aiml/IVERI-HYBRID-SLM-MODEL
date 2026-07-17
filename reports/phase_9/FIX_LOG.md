# Phase 9 — Fix Log

**Date:** 2026-07-11

---

## Applied Fixes

### FIX-001 — Eval dataset registry null handling

**File:** `data/pipeline/registry.py`  
**Bug:** P9-BUG-001  
**Change:** Eval-only YAML specs with `quality_threshold: null` no longer crash registry discovery.  
**Before:** 6/11 specs discovered, 5 ERROR logs  
**After:** 11/11 specs discovered  
**Tests:** Not re-run (no dedicated registry test for eval null case)

---

## Fixes Not Applied (Require User Decision / Infrastructure)

| Bug | Reason deferred |
|-----|-----------------|
| P9-BUG-002 No certified data | Requires Phase 8 pipeline download (~GB disk, HF network) |
| P9-BUG-003 Mock certification | Requires full campaign implementation, not a one-line fix |
| P9-BUG-004 MoE balance | Needs architecture tuning or threshold review |
| P9-BUG-006 Phase 8 integration | Download backend returned 0 docs — needs HF/network investigation |
