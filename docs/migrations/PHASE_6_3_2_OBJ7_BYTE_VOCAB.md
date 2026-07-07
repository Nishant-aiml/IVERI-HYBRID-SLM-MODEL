# Phase 6.3.2 Objective 7 — Byte Vocabulary (Collision-Free)

**Date:** 2026-07-06  
**Scope:** Eliminate PAD/BOS/EOS collisions with real UTF-8 bytes; centralize constants.

## Summary

Structural tokens moved to extended IDs **256–258** (BOS, PAD, EOS). Content bytes remain **0–255** with a 1:1 UTF-8 mapping. Model embeddings expanded to **259**; legacy IDs **0–2** are retained only for checkpoint remap.

## Scientific Rationale

| Token | Old (colliding) | New (collision-free) |
|-------|-----------------|----------------------|
| PAD | 0 (`NUL`) | 257 |
| BOS | 1 (`U+0001`) | 256 |
| EOS | 2 (`U+0002`) | 258 |

NUL and control characters can appear in real corpora without being mistaken for structural tokens.

## Code Changes

| Module | Change |
|--------|--------|
| `core/constants.py` | Extended vocab; legacy constants; `ARCHITECTURE_VERSION=0.2.0-byte-vocab` |
| `core/byte_vocab.py` | Validation, strip, legacy remap helpers |
| `data/preprocessing.py` | `text_to_byte_ids`, `pad_byte_ids` return `list[int]` |
| `data/pipeline/byte_encoder.py` | Collision-free encode/decode + validation |
| `model/*` | Embeddings/logits use `BYTE_VOCAB_SIZE` |
| `training/*`, `evaluation/*` | Import `PAD_BYTE` from `core.constants` |
| `research/byte_vocab_audit.py` | Runtime audit + report generator |
| `tests/test_byte_vocab_audit.py` | OBJ7 tests |

## Validation

```powershell
python -m pytest tests/test_byte_vocab_audit.py tests/test_dataset.py -q
python -c "from research.byte_vocab_audit import write_byte_vocab_report; write_byte_vocab_report('reports/scientific_integrity_audit/Byte_Vocabulary_Report.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Byte_Vocabulary_Report.md`

## Backward Compatibility

Pre-v0.2.0 checkpoints require `remap_legacy_token_ids()` at load time. Full re-encode of training data is required for paper-profile campaigns.

## Not Changed (this objective)

- Documentation sync across README/CHANGELOG (Objective 8)
