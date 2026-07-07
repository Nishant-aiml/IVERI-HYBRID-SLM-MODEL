# Checkpoint Selector Report — IVERI CORE Phase 3.1

This report documents the checkpoint selector execution and saved checkpoints during pretraining.

## Saved Checkpoints

- **Experiment Directory**: `logs/iveri_stage1_lvl2/`
- **Checkpoint Frequency**: Every 50 steps
- **Saved Checkpoints**:
  - `checkpoint_50.pt` (Step 50): Loss = 4.6385, Perplexity = 103.39
  - `checkpoint_100.pt` (Step 100): *Saving pending step 100 completion*

## Checkpoint Selector Metrics

1. **Top Checkpoints Tracking**: Evaluates checkpoints based on validation loss and perplexity.
2. **Best Checkpoint Registration**:
   - `checkpoint_50.pt` is currently registered as the best checkpoint with validation loss = 5.5432 (from step 50 val pass).
3. **Resume Capability**: `resume_metadata.json` successfully records step 50 as the latest saved checkpoint, ensuring safe restart in case of worker failure.

## Conclusion

The checkpoint selector successfully manages checkpoint saving and indexing. Checkpoints are verified and compatible.
