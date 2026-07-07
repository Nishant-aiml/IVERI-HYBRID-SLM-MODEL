# Checkpoint Report — Serialization Layout and Quality Ranking

This report validates checkpoint saving, loading, resume metadata, and ranking.

---

## 1. Serialization Schema

SFT checkpoints save the full training state:
- Model parameter weights (`state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Scheduler state (`scheduler_state_dict`)
- AMP Scaler (`scaler_state_dict`)
- Current iteration step and epoch
- Master configuration snapshot

---

## 2. Model Selection & Ranking

The `SFTCheckpointSelector` manages checkpoint metrics in `checkpoint_history.json`. It supports:
1. **Validation Loss & Perplexity**: Standard lower-is-better metric.
2. **Response Quality Score**: Shannon entropy and loop checking (higher-is-better).
3. **Joint Ranking**: Ranks models by a combined score: `joint = val_loss - 0.5 * response_quality_score`.

---

## 3. Resume / Recovery Validation

We verified that SFT training can resume seamlessly. The `resume_metadata.json` stores:
```json
{
  "step": 100,
  "epoch": 0,
  "latest_checkpoint": "logs/iveri_stage2_sft_lvl1/sft_checkpoint_100.pt",
  "timestamp": "2026-07-02T22:45:00"
}
```
All weights and states load cleanly with identical numerical outputs.
