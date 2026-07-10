# Byte Vocabulary Report (Phase 6.3.2 OBJ7)

**Generated:** 2026-07-09T02:58:47Z  
**Protocol:** Phase-6.3.2-OBJ7  
**Architecture:** 0.2.0-byte-vocab  

## Executive Verdict

**Collision-free byte vocabulary:** `PASS`

| Gate | Result | Detail |
|------|--------|--------|
| special_disjoint_from_raw | PASS | BOS=256, PAD=257, EOS=258, vocab=259 |
| legacy_collision_removed | PASS | legacy=[0, 1, 2]; active specials=[256, 257, 258] |
| encode_roundtrip | PASS | decoded='Hello world' |
| legacy_remap | PASS | remapped=[256, 65, 258] |
| centralized_constants | PASS | no local PAD_BYTE=0 |

## Scientific Rationale

1. Raw UTF-8 payload bytes occupy IDs **0–255** without reinterpretation.
2. Structural tokens **BOS=256**, **PAD=257**, **EOS=258** are outside the byte range.
3. NUL (0), U+0001, and U+0002 can appear in real text/binary without colliding with specials.
4. `remap_legacy_token_ids()` supports inference on pre-v0.2.0 checkpoints only.

## Proof: Runtime Gates

1. Content bytes map 1:1 to IDs 0–255; BOS/PAD/EOS use extended IDs 256–258.
2. Legacy colliding assignments (0, 1, 2) preserved only for checkpoint remap.
3. ByteEncoder validates token IDs and strips specials on decode.
4. Model embeddings expanded to BYTE_VOCAB_SIZE=259.
5. ARCHITECTURE_VERSION bumped to 0.2.0-byte-vocab.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ7",
  "timestamp_utc": "2026-07-09T02:58:47Z",
  "production_verdict": "PASS",
  "architecture_version": "0.2.0-byte-vocab",
  "gates": [
    {
      "gate_name": "special_disjoint_from_raw",
      "passed": true,
      "detail": "BOS=256, PAD=257, EOS=258, vocab=259"
    },
    {
      "gate_name": "legacy_collision_removed",
      "passed": true,
      "detail": "legacy=[0, 1, 2]; active specials=[256, 257, 258]"
    },
    {
      "gate_name": "encode_roundtrip",
      "passed": true,
      "detail": "decoded='Hello world'"
    },
    {
      "gate_name": "legacy_remap",
      "passed": true,
      "detail": "remapped=[256, 65, 258]"
    },
    {
      "gate_name": "centralized_constants",
      "passed": true,
      "detail": "no local PAD_BYTE=0"
    }
  ],
  "presence_proof": [
    "Content bytes map 1:1 to IDs 0\u2013255; BOS/PAD/EOS use extended IDs 256\u2013258.",
    "Legacy colliding assignments (0, 1, 2) preserved only for checkpoint remap.",
    "ByteEncoder validates token IDs and strips specials on decode.",
    "Model embeddings expanded to BYTE_VOCAB_SIZE=259.",
    "ARCHITECTURE_VERSION bumped to 0.2.0-byte-vocab."
  ]
}
```
