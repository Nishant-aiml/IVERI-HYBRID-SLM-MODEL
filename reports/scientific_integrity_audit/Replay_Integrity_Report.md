# Replay Integrity Report (Phase 6.3.2 OBJ6)

**Generated:** 2026-07-07T06:08:59Z  
**Protocol:** Phase-6.3.2-OBJ6  

## Executive Verdict

**Replay fail-closed framework:** `PASS`

- Full H1–H10 chain passes on measured seed DB: `True`
- Failure rows block replay: `True`
- Non-paper provenance blocked: `True`

## Proof: Replay Integrity Gates

1. replay_campaign.py pre-flights verify_replay_registry_integrity before publication.
2. Non-zero exit when registry, claim chain, or figure checks fail.
3. Claim chain requires MEASURED metrics per hypothesis experiment.
4. Disallowed tags (verification, pilot, mock, dry_run) block replay sign-off.
5. Figure verification rejects mock placeholder artifacts.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ6",
  "timestamp_utc": "2026-07-07T06:08:59Z",
  "production_verdict": "PASS",
  "full_chain_passes": true,
  "blocks_failure_rows": true,
  "blocks_pilot_provenance": true,
  "replay_integrity": {
    "protocol_version": "Phase-6.3.2-OBJ6",
    "registry_ok": true,
    "claims_ok": true,
    "figures_ok": true,
    "errors": []
  },
  "presence_proof": [
    "replay_campaign.py pre-flights verify_replay_registry_integrity before publication.",
    "Non-zero exit when registry, claim chain, or figure checks fail.",
    "Claim chain requires MEASURED metrics per hypothesis experiment.",
    "Disallowed tags (verification, pilot, mock, dry_run) block replay sign-off.",
    "Figure verification rejects mock placeholder artifacts."
  ]
}
```
